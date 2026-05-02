from __future__ import annotations

import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import set_seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_JSON = PROJECT_ROOT / "data" / "ARCADE" / "processed" / "dataset.json"
EXTRA_ROOT = PROJECT_ROOT / "data" / "Extra"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "unext_pretrained"
LOG_DIR = PROJECT_ROOT / "runs" / "unext_pretrained"


# ============================================================================
# Dataset classes
# ============================================================================

class SyntaxDataset(Dataset):
    """Load Syntax segmentation data from ARCADE dataset.json.

    Combines train + validation splits for training, test split for evaluation.
    """

    def __init__(
        self,
        json_path: str = str(DATASET_JSON),
        train: bool = True,
        transform=None,
        root_dir: str = "",
    ):
        self.transform = transform
        self.root_dir = root_dir
        self.samples: List[Dict] = []

        with open(json_path, "r") as f:
            data = json.load(f)

        if train:
            for split in ("train", "validation"):
                source = data.get(split, {}).get("syntax", {})
                for sid in sorted(source.keys(), key=_natural_key):
                    self.samples.append(source[sid])
        else:
            source = data.get("test", {}).get("syntax", {})
            for sid in sorted(source.keys(), key=_natural_key):
                self.samples.append(source[sid])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        img_path = os.path.join(self.root_dir, item["data"])
        lbl_path = os.path.join(self.root_dir, item["label"])

        image = Image.open(img_path).convert("L")
        label = Image.open(lbl_path).convert("L")

        if self.transform is not None:
            return self.transform(image, label)
        return TF.to_tensor(image), _pil_to_binary_tensor(label)


class ExtraDataset(Dataset):
    """Load image/mask pairs from data/Extra/images/ and data/Extra/masks/."""

    def __init__(
        self,
        root: str = str(EXTRA_ROOT),
        transform=None,
        split_ratio: float = 0.2,
        train: bool = True,
        seed: int = 42,
    ):
        self.transform = transform
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        all_images = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in exts],
            key=lambda p: p.name,
        )

        paired = []
        for img_path in all_images:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                paired.append((str(img_path), str(mask_path)))

        rng = random.Random(seed)
        rng.shuffle(paired)
        split_idx = max(1, int(len(paired) * split_ratio))

        if train:
            self.pairs = paired[split_idx:]
        else:
            self.pairs = paired[:split_idx]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("L")
        label = Image.open(mask_path).convert("L")

        if self.transform is not None:
            return self.transform(image, label)
        return TF.to_tensor(image), _pil_to_binary_tensor(label)


# ============================================================================
# Heavy augmentation transforms
# ============================================================================

class HeavyAugmentation:
    """Heavy paired augmentations for vessel segmentation with 4-channel preprocessing.

    Channel 0: CLAHE (local contrast enhancement)
    Channel 1: White-hat top-hat (bright structures on dark)
    Channel 2: Black-hat (dark structures on bright)
    Channel 3: High-pass unsharp mask (edge emphasis)

    Geometry transforms applied to both image and mask at PIL level,
    then all preprocessing is done at numpy level.
    """

    def __init__(
        self,
        image_size: int = 256,
        training: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.3,
        rotation_degrees: float = 30.0,
        blur_prob: float = 0.3,
        blur_kernel: int = 5,
        blur_sigma_range: Tuple[float, float] = (0.5, 1.6),
        brightness_alpha_range: Tuple[float, float] = (0.85, 1.15),
        brightness_beta_range: Tuple[float, float] = (-12.0, 12.0),
        post_brightness_prob: float = 0.5,
        post_gamma_prob: float = 0.5,
        post_gamma_range: Tuple[float, float] = (0.75, 1.30),
        noise_std: float = 0.03,
        noise_prob: float = 0.5,
        artifact_line_prob: float = 0.0,
        artifact_circle_prob: float = 0.0,
    ):
        self.image_size = image_size
        self.training = training
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_degrees = rotation_degrees
        self.blur_prob = blur_prob
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.blur_sigma_range = blur_sigma_range
        self.brightness_alpha_range = brightness_alpha_range
        self.brightness_beta_range = brightness_beta_range
        self.post_brightness_prob = post_brightness_prob
        self.post_gamma_prob = post_gamma_prob
        self.post_gamma_range = post_gamma_range
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.artifact_line_prob = artifact_line_prob
        self.artifact_circle_prob = artifact_circle_prob

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def _resize(self, image, label):
        image = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        if label is not None:
            label = TF.resize(label, [self.image_size, self.image_size], InterpolationMode.NEAREST)
        return image, label

    def _inject_artifacts(self, image_np: np.ndarray):
        if random.random() < self.artifact_line_prob:
            num_lines = random.randint(1, 3)
            for _ in range(num_lines):
                x1 = random.randint(0, self.image_size)
                y1 = random.randint(0, self.image_size)
                x2 = x1 + random.randint(-150, 150)
                y2 = y1 + random.randint(-150, 150)
                thickness = random.randint(2, 6)
                color = random.randint(10, 80)
                cv2.line(image_np, (x1, y1), (x2, y2), color, thickness)

        if random.random() < self.artifact_circle_prob:
            cx = random.randint(0, self.image_size)
            cy = random.randint(0, self.image_size)
            radius = random.randint(15, 60)
            color = random.randint(30, 100)
            cv2.circle(image_np, (cx, cy), radius, color, -1)

    def __call__(self, image: Image.Image, label: Optional[Image.Image] = None):
        image, label = self._resize(image, label)

        if self.training:
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                label = TF.hflip(label) if label is not None else None

            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                label = TF.vflip(label) if label is not None else None

            if self.rotation_degrees > 0:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                image = TF.rotate(image, angle, InterpolationMode.BILINEAR, fill=0)
                if label is not None:
                    label = TF.rotate(label, angle, InterpolationMode.NEAREST, fill=0)

        image_np = np.array(image, dtype=np.uint8)

        if self.training:
            self._inject_artifacts(image_np)

            if random.random() < self.blur_prob:
                sigma = random.uniform(*self.blur_sigma_range)
                image_np = cv2.GaussianBlur(image_np, (self.blur_kernel, self.blur_kernel), sigmaX=sigma)

            if self.brightness_alpha_range != (1.0, 1.0) or self.brightness_beta_range != (0.0, 0.0):
                alpha = random.uniform(*self.brightness_alpha_range)
                beta = random.uniform(*self.brightness_beta_range)
                image_np = np.clip(alpha * image_np.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        c1 = self.clahe.apply(image_np)
        c2 = cv2.morphologyEx(image_np, cv2.MORPH_TOPHAT, self.morph_kernel)
        c3 = cv2.morphologyEx(image_np, cv2.MORPH_BLACKHAT, self.morph_kernel)
        blurred = cv2.GaussianBlur(image_np, (0, 0), sigmaX=10)
        c4 = cv2.addWeighted(image_np, 4.0, blurred, -4.0, 128)

        stacked = np.stack([c1, c2, c3, c4], axis=-1)
        img_t = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0

        if self.training:
            if random.random() < self.post_brightness_prob:
                brightness = random.uniform(0.85, 1.15)
                img_t = torch.clamp(img_t * brightness, 0.0, 1.0)

            if random.random() < self.post_gamma_prob:
                gamma = random.uniform(*self.post_gamma_range)
                img_t = torch.pow(img_t, gamma)

            if random.random() < self.noise_prob:
                noise = torch.randn_like(img_t) * self.noise_std
                img_t = torch.clamp(img_t + noise, 0.0, 1.0)

        if label is not None:
            label_tensor = _pil_to_binary_tensor(label)
            return img_t, label_tensor

        return img_t


# ============================================================================
# LoRA for Conv2d in UNeXt
# ============================================================================

class LoRAConv2d(nn.Module):
    """LoRA adapter injected into nn.Conv2d layers.

    Applies to kernel_size=1 (1x1) convolutions which act as channel-mixing linear layers.
    """

    def __init__(self, conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.conv = conv
        self.r = r
        self.scaling = alpha / r

        out_c, in_c, kh, kw = conv.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(r, in_c))
        self.lora_B = nn.Parameter(torch.zeros(out_c, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)
        delta = (self.lora_B @ self.lora_A).reshape(self.conv.weight.shape)
        lora_out = F.conv2d(
            x, delta, self.conv.bias, self.conv.stride,
            self.conv.padding, self.conv.dilation, self.conv.groups,
        )
        return base + self.scaling * self.dropout(lora_out)


def inject_lora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """Replace all 1x1 Conv2d layers in UNeXt with LoRAConv2d wrappers."""
    replaced: Set[str] = set()

    def _replace(parent: nn.Module, child_name: str, child: nn.Module):
        if isinstance(child, nn.Conv2d):
            kh, kw = child.kernel_size
            if kh == 1 and kw == 1 and child.groups == 1:
                setattr(parent, child_name, LoRAConv2d(child, r=r, alpha=alpha, dropout=dropout))
                replaced.add(child_name)
                return
        for sub_name, sub_mod in child.named_children():
            _replace(child, sub_name, sub_mod)

    for name, module in model.named_children():
        _replace(model, name, module)

    lora_count = sum(1 for m in model.modules() if isinstance(m, LoRAConv2d))
    print(f"[LoRA] Injected {lora_count} LoRA adapters into 1x1 Conv2d layers.")
    return model


def freeze_base_train_lora(model: nn.Module):
    """Freeze all base parameters; unfreeze only LoRA parameters."""
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name
    lora_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Trainable: {lora_trainable:,} / {total:,} ({100 * lora_trainable / total:.2f}%)")


# ============================================================================
# Loss helpers
# ============================================================================

def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    return (torch.pow(1.0 - pt, gamma) * bce).mean()


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    intersection = (prob * targets).sum()
    union = prob.sum() + targets.sum()
    return 1.0 - (2.0 * intersection + smooth) / (union + smooth)


def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
    preds = (torch.sigmoid(logits) > 0.5).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    denom_f1 = 2.0 * tp + fp + fn
    f1 = (2.0 * tp) / denom_f1 if denom_f1 > 0 else 1.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return f1, dice, iou


def _pil_to_binary_tensor(pil_image: Image.Image) -> torch.Tensor:
    t = TF.pil_to_tensor(pil_image).float()
    if t.max() > 1.0:
        t = t / 255.0
    return (t > 0.5).float()


def _natural_key(s: str) -> tuple:
    try:
        return (0, int(s))
    except (TypeError, ValueError):
        return (1, s)


def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _make_preview_grid(images, masks, probs, n: int = 4):
    n = min(images.shape[0], n)
    rows = []
    for i in range(n):
        img = images[i, 0:1].detach().float().cpu()
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        img = img.clamp(0, 1)
        m = masks[i].detach().float().cpu().clamp(0, 1)
        p = (probs[i].detach().float().cpu() > 0.5).float()
        rows.extend([img, m, p])
    grid = make_grid(torch.cat(rows, dim=0), nrow=3, padding=2)
    grid = (grid * 255).clamp(0, 255).to(torch.uint8)
    return grid


# ============================================================================
# Training epochs
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    tag: str = "Train",
) -> Dict[str, float]:
    model.train()
    total_focal, total_dice, total_loss = 0.0, 0.0, 0.0
    tp_all, fp_all, fn_all = 0.0, 0.0, 0.0

    pbar = tqdm(loader, desc=f"{tag} E{epoch:03d}", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device):
            logits = model(images)
            f_loss = _focal_loss(logits, masks)
            d_loss = _dice_loss(logits, masks)
            loss = f_loss + d_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            f1, dice, iou = _compute_metrics(logits, masks)

        bs = images.size(0)
        tp_all += f1 * bs
        fp_all += (1.0 - dice) * bs
        fn_all += (1.0 - iou) * bs
        total_loss += loss.item() * bs
        total_focal += f_loss.item() * bs
        total_dice += d_loss.item() * bs

        pbar.set_postfix(loss=total_loss / max(1, (batch_idx + 1) * bs))

        if writer is not None and batch_idx == 0 and epoch % 5 == 0:
            try:
                probs = torch.sigmoid(logits)
                writer.add_image(f"{tag}/preview", _make_preview_grid(images, masks, probs), epoch)
            except Exception:
                pass

    n = max(1, len(loader.dataset))
    return {
        "loss": total_loss / n,
        "focal": total_focal / n,
        "dice": total_dice / n,
        "f1": tp_all / n if n > 0 else 0.0,
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    tag: str = "Val",
) -> Dict[str, float]:
    model.eval()
    total_loss, total_focal, total_dice = 0.0, 0.0, 0.0
    tp_all, fp_all, fn_all = 0.0, 0.0, 0.0
    logged = False

    pbar = tqdm(loader, desc=f"{tag} E{epoch:03d}", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with _autocast_ctx(device):
            logits = model(images)
            f_loss = _focal_loss(logits, masks)
            d_loss = _dice_loss(logits, masks)
            loss = f_loss + d_loss

        with torch.no_grad():
            f1, dice, iou = _compute_metrics(logits, masks)

        bs = images.size(0)
        tp_all += f1 * bs
        fp_all += (1.0 - dice) * bs
        fn_all += (1.0 - iou) * bs
        total_loss += loss.item() * bs
        total_focal += f_loss.item() * bs
        total_dice += d_loss.item() * bs

        if writer is not None and not logged:
            try:
                probs = torch.sigmoid(logits)
                writer.add_image(f"{tag}/preview", _make_preview_grid(images, masks, probs), epoch)
            except Exception:
                pass
            logged = True

    n = max(1, len(loader.dataset))
    return {
        "loss": total_loss / n,
        "focal": total_focal / n,
        "dice": total_dice / n,
        "f1": tp_all / n if n > 0 else 0.0,
    }


# ============================================================================
# Phase 1: Train UNeXt on Syntax data
# ============================================================================

def phase1_train_syntax(
    image_size: int = 512,
    batch_size: int = 8,
    epochs: int = 150,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    log_dir: Path = LOG_DIR,
) -> str:
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- datasets ----
    train_aug = HeavyAugmentation(
        image_size=image_size,
        training=True,
        hflip_prob=0.5,
        vflip_prob=0.3,
        rotation_degrees=30.0,
        blur_prob=0.4,
        blur_kernel=7,
        brightness_alpha_range=(0.85, 1.15),
        brightness_beta_range=(-12.0, 12.0),
        post_gamma_range=(0.7, 1.5),
        noise_std=0.04,
        noise_prob=0.5,
    )
    val_aug = HeavyAugmentation(image_size=image_size, training=False)

    train_ds = SyntaxDataset(json_path=str(DATASET_JSON), train=True, transform=train_aug)
    test_ds = SyntaxDataset(json_path=str(DATASET_JSON), train=False, transform=val_aug)

    print(f"[Phase1] Syntax train: {len(train_ds)} samples (train+val combined)")
    print(f"[Phase1] Syntax test:  {len(test_ds)} samples")

    def seed_worker(wid: int):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    g = torch.Generator().manual_seed(seed)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **{k: v for k, v in loader_kwargs.items() if k != "generator"})

    # ---- model ----
    from zoo.unext import UNeXt_S
    model = UNeXt_S(in_channels=4, num_classes=1, base_channels=128, depths=[2, 1, 1],
                    mlp_ratio=4, drop_rate=0.1, attention=True, use_checkpoint=False)
    model = model.to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"[Phase1] UNeXt_S params: {total_p:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir=str(log_dir / "phase1"))
    best_f1 = -1.0
    best_path = str(checkpoint_dir / "phase1_best.pth")
    last_path = str(checkpoint_dir / "phase1_last.pth")

    for epoch in range(1, epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch, writer, "Phase1_Train")
        test_stats = validate_epoch(model, test_loader, device, epoch, writer, "Phase1_Test")
        scheduler.step()

        writer.add_scalar("Phase1/Train_Loss", train_stats["loss"], epoch)
        writer.add_scalar("Phase1/Train_F1", train_stats["f1"], epoch)
        writer.add_scalar("Phase1/Test_Loss", test_stats["loss"], epoch)
        writer.add_scalar("Phase1/Test_F1", test_stats["f1"], epoch)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": max(best_f1, test_stats["f1"]),
        }
        torch.save(ckpt, last_path)

        if test_stats["f1"] > best_f1:
            best_f1 = test_stats["f1"]
            torch.save(ckpt, best_path)
            print(f"  >> New best F1: {best_f1:.4f} @ epoch {epoch}")

        print(
            f"[Phase1] E{epoch:03d}/{epochs} | "
            f"train_loss={train_stats['loss']:.4f} f1={train_stats['f1']:.4f} | "
            f"test_loss={test_stats['loss']:.4f} f1={test_stats['f1']:.4f} | best={best_f1:.4f}"
        )

    writer.close()
    print(f"[Phase1] Done. Best F1: {best_f1:.4f} -> {best_path}")
    return best_path


# ============================================================================
# Phase 2: LoRA fine-tune on Extra data
# ============================================================================

def phase2_lora_extra(
    phase1_ckpt: str,
    image_size: int = 512,
    batch_size: int = 4,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    seed: int = 42,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    log_dir: Path = LOG_DIR,
) -> str:
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- datasets ----
    train_aug = HeavyAugmentation(
        image_size=image_size,
        training=True,
        hflip_prob=0.5,
        vflip_prob=0.3,
        rotation_degrees=30.0,
        blur_prob=0.5,
        blur_kernel=7,
        brightness_alpha_range=(0.80, 1.20),
        brightness_beta_range=(-15.0, 15.0),
        post_gamma_range=(0.6, 1.6),
        noise_std=0.06,
        noise_prob=0.5,
    )
    val_aug = HeavyAugmentation(image_size=image_size, training=False)

    train_ds = ExtraDataset(root=str(EXTRA_ROOT), transform=train_aug, split_ratio=0.2, train=True, seed=seed)
    val_ds = ExtraDataset(root=str(EXTRA_ROOT), transform=val_aug, split_ratio=0.2, train=False, seed=seed)

    print(f"[Phase2] Extra train: {len(train_ds)} samples")
    print(f"[Phase2] Extra val:   {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # ---- load base model ----
    from zoo.unext import UNeXt_S
    model = UNeXt_S(in_channels=4, num_classes=1, base_channels=64, depths=[2, 2, 2],
                    mlp_ratio=4, drop_rate=0.2, attention=True, use_checkpoint=False)
    state = torch.load(phase1_ckpt, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False)
    print(f"[Phase2] Loaded base model from epoch {state.get('epoch', '?')}")

    # ---- inject LoRA ----
    model = inject_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    model = model.to(device)

    # ---- freeze base, train LoRA only ----
    freeze_base_train_lora(model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir=str(log_dir / "phase2"))
    best_f1 = -1.0
    best_path = str(checkpoint_dir / "phase2_lora_best.pth")
    last_path = str(checkpoint_dir / "phase2_lora_last.pth")

    for epoch in range(1, epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch, writer, "Phase2_Train")
        val_stats = validate_epoch(model, val_loader, device, epoch, writer, "Phase2_Val")
        scheduler.step()

        writer.add_scalar("Phase2/Train_Loss", train_stats["loss"], epoch)
        writer.add_scalar("Phase2/Train_F1", train_stats["f1"], epoch)
        writer.add_scalar("Phase2/Val_Loss", val_stats["loss"], epoch)
        writer.add_scalar("Phase2/Val_F1", val_stats["f1"], epoch)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": max(best_f1, val_stats["f1"]),
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        }
        torch.save(ckpt, last_path)

        if val_stats["f1"] > best_f1:
            best_f1 = val_stats["f1"]
            torch.save(ckpt, best_path)
            print(f"  >> New best vessel F1: {best_f1:.4f} @ epoch {epoch}")

        print(
            f"[Phase2] E{epoch:03d}/{epochs} | "
            f"train_loss={train_stats['loss']:.4f} f1={train_stats['f1']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} f1={val_stats['f1']:.4f} | best={best_f1:.4f}"
        )

    writer.close()
    print(f"[Phase2] Done. Best F1: {best_f1:.4f} -> {best_path}")
    return best_path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="UNeXt Pretraining: Phase1 Syntax + Phase2 LoRA Extra")
    parser.add_argument("--phase1_only", action="store_true", help="Run only Phase 1")
    parser.add_argument("--phase2_only", action="store_true", help="Run only Phase 2 (requires --phase1_ckpt)")
    parser.add_argument("--phase1_ckpt", type=str, default=None,
                        help="Path to Phase 1 checkpoint for Phase 2 fine-tuning")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size_p1", type=int, default=4)
    parser.add_argument("--batch_size_p2", type=int, default=2)
    parser.add_argument("--epochs_p1", type=int, default=150)
    parser.add_argument("--epochs_p2", type=int, default=100)
    parser.add_argument("--lr_p1", type=float, default=3e-4)
    parser.add_argument("--lr_p2", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    run_both = not args.phase1_only and not args.phase2_only

    if args.phase1_only or run_both:
        print("=" * 60)
        print("PHASE 1: Training UNeXt on Syntax (vessel segmentation)")
        print("=" * 60)
        p1_ckpt = phase1_train_syntax(
            image_size=args.image_size,
            batch_size=args.batch_size_p1,
            epochs=args.epochs_p1,
            lr=args.lr_p1,
            seed=args.seed,
            num_workers=args.num_workers,
            device=device,
        )
    else:
        p1_ckpt = args.phase1_ckpt

    if args.phase2_only or run_both:
        if p1_ckpt is None:
            raise ValueError("Phase 2 requires a Phase 1 checkpoint (--phase1_ckpt)")
        print()
        print("=" * 60)
        print("PHASE 2: LoRA fine-tuning on Extra (full vessel segmentation)")
        print("=" * 60)
        phase2_lora_extra(
            phase1_ckpt=p1_ckpt,
            image_size=args.image_size,
            batch_size=args.batch_size_p2,
            epochs=args.epochs_p2,
            lr=args.lr_p2,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            seed=args.seed,
            num_workers=args.num_workers,
            device=device,
        )


if __name__ == "__main__":
    main()
