from __future__ import annotations

import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import set_seed

EXTRA_ROOT = PROJECT_ROOT / "data" / "Extra"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "unext_lora"
LOG_DIR = PROJECT_ROOT / "runs" / "unext_lora"


# ============================================================================
# Dataset
# ============================================================================

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
# Heavy augmentation (4-channel preprocessing)
# ============================================================================

class HeavyAugmentation:
    """4-channel preprocessing: CLAHE, white-hat, black-hat, high-pass."""

    def __init__(
        self,
        image_size: int = 512,
        training: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.3,
        rotation_degrees: float = 30.0,
        blur_prob: float = 0.5,
        blur_kernel: int = 7,
        blur_sigma_range: Tuple[float, float] = (0.5, 1.6),
        brightness_alpha_range: Tuple[float, float] = (0.80, 1.20),
        brightness_beta_range: Tuple[float, float] = (-15.0, 15.0),
        post_brightness_prob: float = 0.5,
        post_gamma_prob: float = 0.5,
        post_gamma_range: Tuple[float, float] = (0.6, 1.6),
        noise_std: float = 0.06,
        noise_prob: float = 0.5,
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

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def _resize(self, image, label):
        image = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        if label is not None:
            label = TF.resize(label, [self.image_size, self.image_size], InterpolationMode.NEAREST)
        return image, label

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
            if random.random() < self.blur_prob:
                sigma = random.uniform(*self.blur_sigma_range)
                image_np = cv2.GaussianBlur(image_np, (self.blur_kernel, self.blur_kernel), sigmaX=sigma)
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
                img_t = torch.clamp(img_t * random.uniform(0.85, 1.15), 0.0, 1.0)
            if random.random() < self.post_gamma_prob:
                img_t = torch.pow(img_t, random.uniform(*self.post_gamma_range))
            if random.random() < self.noise_prob:
                img_t = torch.clamp(img_t + torch.randn_like(img_t) * self.noise_std, 0.0, 1.0)

        if label is not None:
            return img_t, _pil_to_binary_tensor(label)
        return img_t


# ============================================================================
# LoRA
# ============================================================================

class LoRAConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.conv = conv
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
        lora_out = F.conv2d(x, delta, self.conv.bias, self.conv.stride,
                            self.conv.padding, self.conv.dilation, self.conv.groups)
        return base + self.scaling * self.dropout(lora_out)


def inject_lora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    replaced: set = set()
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
    print(f"[LoRA] Injected {len(replaced)} LoRA adapters into 1x1 Conv2d layers.")
    return model


def freeze_base_train_lora(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


# ============================================================================
# Loss / helpers
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
    denom = 2.0 * tp + fp + fn
    f1 = (2.0 * tp) / denom if denom > 0 else 1.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return f1, dice, iou


def _pil_to_binary_tensor(pil_image: Image.Image) -> torch.Tensor:
    t = TF.pil_to_tensor(pil_image).float()
    if t.max() > 1.0:
        t = t / 255.0
    return (t > 0.5).float()


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
    return (grid * 255).clamp(0, 255).to(torch.uint8)


# ============================================================================
# Training loops
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
    total_loss, total_focal, total_dice = 0.0, 0.0, 0.0
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
    return {"loss": total_loss / n, "focal": total_focal / n, "dice": total_dice / n, "f1": tp_all / n if n > 0 else 0.0}


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
    return {"loss": total_loss / n, "focal": total_focal / n, "dice": total_dice / n, "f1": tp_all / n if n > 0 else 0.0}


# ============================================================================
# Phase 2: LoRA fine-tune on Extra
# ============================================================================

def phase2_lora_extra(
    phase1_ckpt: str,
    image_size: int = 512,
    batch_size: int = 2,
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
    resume_ckpt: Optional[str] = None,
) -> str:
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- datasets ----
    train_aug = HeavyAugmentation(image_size=image_size, training=True)
    val_aug = HeavyAugmentation(image_size=image_size, training=False)

    train_ds = ExtraDataset(root=str(EXTRA_ROOT), transform=train_aug, split_ratio=0.2, train=True, seed=seed)
    val_ds = ExtraDataset(root=str(EXTRA_ROOT), transform=val_aug, split_ratio=0.2, train=False, seed=seed)
    print(f"[Phase2] Extra train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ---- build model ----
    from zoo.unext import UNeXt_S
    model = UNeXt_S(in_channels=4, num_classes=1, base_channels=64, depths=[2, 2, 2],
                    mlp_ratio=4, drop_rate=0.2, attention=True, use_checkpoint=False)

    start_epoch = 1
    best_f1 = -1.0
    optimizer = None
    scheduler = None

    if resume_ckpt:
        # Resume from previous Phase 2 checkpoint (model already has LoRA injected)
        state = torch.load(resume_ckpt, map_location="cpu")
        model = inject_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        model.load_state_dict(state["model_state_dict"], strict=False)
        model = model.to(device)
        freeze_base_train_lora(model)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state.get("epoch", 0) + 1
        best_f1 = state.get("best_f1", -1.0)
        print(f"[Phase2] Resumed from epoch {state.get('epoch', '?')}, best F1: {best_f1:.4f}")
    else:
        # Load Phase 1 base weights, then inject LoRA
        state = torch.load(phase1_ckpt, map_location="cpu")
        model.load_state_dict(state["model_state_dict"], strict=False)
        model = inject_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        model = model.to(device)
        freeze_base_train_lora(model)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        print(f"[Phase2] Loaded Phase 1 from epoch {state.get('epoch', '?')}")

    if scheduler is None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir=str(log_dir / "phase2"))
    best_path = str(checkpoint_dir / "phase2_lora_best.pth")
    last_path = str(checkpoint_dir / "phase2_lora_last.pth")

    for epoch in range(start_epoch, epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch, writer, "Train")
        val_stats = validate_epoch(model, val_loader, device, epoch, writer, "Val")
        scheduler.step()

        writer.add_scalar("Train/Loss", train_stats["loss"], epoch)
        writer.add_scalar("Train/F1", train_stats["f1"], epoch)
        writer.add_scalar("Val/Loss", val_stats["loss"], epoch)
        writer.add_scalar("Val/F1", val_stats["f1"], epoch)

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
            print(f"  >> New best F1: {best_f1:.4f} @ epoch {epoch}")

        print(
            f"E{epoch:03d}/{epochs} | "
            f"train_loss={train_stats['loss']:.4f} f1={train_stats['f1']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} f1={val_stats['f1']:.4f} | best={best_f1:.4f}"
        )

    writer.close()
    print(f"[Phase2] Done. Best F1: {best_f1:.4f} -> {best_path}")
    return best_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UNeXt Phase 2: LoRA fine-tune on Extra data")
    parser.add_argument("--phase1_ckpt", type=str, required=True, help="Path to Phase 1 checkpoint (.pth)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from Phase 2 checkpoint (.pth)")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    phase2_lora_extra(
        phase1_ckpt=args.phase1_ckpt,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
        resume_ckpt=args.resume,
    )
