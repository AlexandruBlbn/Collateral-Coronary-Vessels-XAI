import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import set_seed
from data.vessel_consensus import compute_consensus_mask
from zoo.backbones import get_backbone

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# BLOCK MASKING
# ---------------------------------------------------------------------------

def generate_block_mask(
    h: int, w: int,
    mask_scale: Tuple[float, float] = (0.15, 0.7),
    aspect_ratio: Tuple[float, float] = (0.75, 1.5),
    num_blocks: int = 8,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate a random block mask for CNNs.
    Returns a binary mask [1, H, W] where 1 = masked (to be predicted), 0 = visible.
    """
    mask = torch.zeros(1, h, w, device=device)
    mask_ratio = random.uniform(*mask_scale)
    ar = random.uniform(*aspect_ratio)

    total_pixels = h * w
    target_masked = int(total_pixels * mask_ratio)

    # Divide into blocks and randomly mask them
    block_h = max(4, h // num_blocks)
    block_w = max(4, w // num_blocks)

    current_masked = 0
    attempts = 0
    while current_masked < target_masked and attempts < 200:
        attempts += 1
        bh = int(block_h * random.uniform(0.5, 1.5))
        bw = int(block_w * random.uniform(0.5, 1.5) * ar)
        bh = min(bh, h // 2)
        bw = min(bw, w // 2)
        y = random.randint(0, max(0, h - bh - 1))
        x = random.randint(0, max(0, w - bw - 1))
        mask[0, y:y+bh, x:x+bw] = 1.0
        current_masked = int(mask.sum().item())

    return mask


def apply_mask(image: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Apply a binary mask to an image. Masked regions are filled with fill_value."""
    return image * (1.0 - mask) + fill_value * mask


# ---------------------------------------------------------------------------
# PREDICTOR
# ---------------------------------------------------------------------------

class ConvPredictor(nn.Module):
    """
    Lightweight convolutional predictor that operates on a spatial feature map.
    Takes encoder features and tries to reconstruct the target features.
    """
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# JEPA MODEL (Architecture-Agnostic)
# ---------------------------------------------------------------------------

class JEPAModel(nn.Module):
    """
    Joint-Embedding Predictive Architecture.

    Works with ANY backbone (EfficientNet, ConvNeXt, ViT, etc.)
    by extracting multi-scale spatial features and applying
    lightweight conv predictors per layer.

    Key design:
    - Encoder processes masked image → multi-scale features
    - Same encoder processes full image (stop-grad) → target features
    - Conv predictors per layer try to reconstruct target features
    - Loss is weighted by vessel confidence at masked positions
    - Deep self-supervision: loss applied at 4 intermediate layers + final
    - No EMA teacher (same encoder, stop-gradient on target)
    """
    def __init__(
        self,
        backbone_name: str = "tf_efficientnetv2_s",
        in_channels: int = 1,
        pretrained: bool = False,
        predictor_hidden_dim: Optional[int] = None,
        deep_layers: Tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.deep_layers = deep_layers

        # Build encoder that returns multi-scale features
        # out_indices for deep supervision: we want features at different strides
        # EfficientNetV2 stages: stride 2, 4, 8, 16 (4 stages)
        self.encoder = get_backbone(
            model_name=backbone_name,
            in_channels=in_channels,
            pretrained=pretrained,
            return_intermediates=True,
            out_indices=(0, 1, 2, 3),  # 4 intermediate stages
        )

        # Determine feature dimensions per layer
        # Run a dummy forward to get channel counts
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 256, 256)
            dummy_out = self.encoder(dummy)
            self.enc_dims = [f.shape[1] for f in dummy_out]
            self.enc_strides = []
            for f in dummy_out:
                stride = 256 // f.shape[-1]
                self.enc_strides.append(stride)

        print(f"[JEPA] Encoder stages: {len(self.enc_dims)} layers")
        for i, (d, s) in enumerate(zip(self.enc_dims, self.enc_strides)):
            print(f"       Layer {i}: {d} channels, stride {s}")

        # Build predictors for each deep supervision layer + final layer
        self.predictors = nn.ModuleList()
        for i in range(len(self.enc_dims)):
            if i in self.deep_layers or i == len(self.enc_dims) - 1:
                self.predictors.append(ConvPredictor(self.enc_dims[i], predictor_hidden_dim))
            else:
                self.predictors.append(nn.Identity())

    def forward(
        self,
        masked_image: torch.Tensor,
        full_image: torch.Tensor,
        mask: torch.Tensor,
        vessel_conf: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            masked_image: [B, C, H, W] image with masked regions
            full_image: [B, C, H, W] original image (target, stop-grad)
            mask: [B, 1, H, W] binary mask (1 = masked, 0 = visible)
            vessel_conf: [B, 1, H', W'] vessel confidence map (optional)
            alpha: vessel confidence weighting strength

        Returns:
            dict with 'loss', 'loss_per_layer', 'predicted', 'target'
        """
        # 1. Encode masked image → predictions
        pred_features = self.encoder(masked_image)  # list of tensors

        # 2. Encode full image → targets (stop-gradient)
        with torch.no_grad():
            target_features = self.encoder(full_image)

        # 3. Apply predictors to selected layers
        losses = {}
        total_loss = 0.0
        n_layers = 0

        for i in range(len(pred_features)):
            if i not in self.deep_layers and i != len(pred_features) - 1:
                continue

            pred = self.predictors[i](pred_features[i])
            target = target_features[i]

            # Resize mask and vessel confidence to match feature resolution
            feat_h, feat_w = pred.shape[-2:]
            mask_down = F.interpolate(mask, size=(feat_h, feat_w), mode="nearest")

            # Build per-pixel weight
            weight = torch.ones_like(mask_down, dtype=torch.float32)
            if vessel_conf is not None:
                conf_down = F.interpolate(vessel_conf, size=(feat_h, feat_w), mode="bilinear", align_corners=False)
                weight = weight + alpha * conf_down

            # MSE at all positions, weighted
            mse = (pred - target) ** 2
            weighted_mse = (weight * mse).mean()

            losses[f"layer_{i}"] = weighted_mse.detach().item()
            total_loss = total_loss + weighted_mse
            n_layers += 1

        avg_loss = total_loss / max(1, n_layers)

        return {
            "loss": avg_loss,
            "loss_per_layer": losses,
            "predicted": pred_features,
            "target": target_features,
        }


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class XCADensePretrainDataset(Dataset):
    """
    Dataset for JEPA pretraining on XCA images.

    Loads grayscale XCA images and precomputed vessel confidence maps.
    The vessel confidence maps are expected as .npy files alongside the images,
    or computed on-the-fly (slow, for prototyping only).
    """
    def __init__(
        self,
        json_path: str,
        split: str = "train",
        root_dir: str = ".",
        img_size: int = 256,
        conf_dir: Optional[str] = None,
        compute_conf_on_the_fly: bool = False,
        sources: Tuple[str, ...] = ("syntax", "stenoza", "cadica", "extra", "coronarydominance"),
        max_samples: Union[int, str] = "all",
    ):
        self.root_dir = Path(root_dir)
        self.img_size = int(img_size)
        self.conf_dir = Path(conf_dir) if conf_dir else None
        self.compute_conf_on_the_fly = compute_conf_on_the_fly
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        with open(json_path, "r") as f:
            data = json.load(f)

        split_key = split
        if split_key not in data:
            if split == "val" and "validation" in data:
                split_key = "validation"
            elif split == "validation" and "val" in data:
                split_key = "val"
            else:
                raise ValueError(f"Split '{split}' not found in {list(data.keys())}")

        self.samples = []
        split_data = data[split_key]
        for source_name in sources:
            source_data = split_data.get(source_name, {})
            for sample_id in sorted(source_data.keys()):
                info = source_data[sample_id]
                img_path = info.get("data")
                if not img_path:
                    continue
                self.samples.append({
                    "image_path": img_path,
                    "source": source_name,
                    "id": sample_id,
                })

        if isinstance(max_samples, int) and max_samples > 0 and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        print(f"[Dataset] Loaded {len(self.samples)} samples for split '{split}'")

    def __len__(self):
        return len(self.samples)

    def _load_conf_map(self, img_path: str) -> np.ndarray:
        """Load precomputed vessel confidence map, or compute on-the-fly."""
        if self.conf_dir is not None:
            conf_path = self.conf_dir / Path(img_path).with_suffix(".npy").name
            if conf_path.exists():
                return np.load(str(conf_path))

        if self.compute_conf_on_the_fly:
            abs_path = str(self.root_dir / img_path)
            img_np = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
            if img_np is None:
                return np.zeros((self.img_size // 8, self.img_size // 8), dtype=np.float32)
            img_np = cv2.resize(img_np, (self.img_size, self.img_size))
            res = compute_consensus_mask(img_np)
            conf = res["consensus"]
            # Pool to 8x8 patches (90th percentile)
            h, w = conf.shape
            hs, ws = h // 8, w // 8
            pooled = np.zeros((hs, ws), dtype=np.float32)
            for i in range(hs):
                for j in range(ws):
                    pooled[i, j] = np.percentile(conf[i*8:(i+1)*8, j*8:(j+1)*8], 90)
            return pooled

        return np.zeros((self.img_size // 8, self.img_size // 8), dtype=np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        abs_path = str(self.root_dir / sample["image_path"])

        # Load and preprocess image
        img_np = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            img_np = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img_np = cv2.resize(img_np, (self.img_size, self.img_size))

        # CLAHE normalization
        img_np = self.clahe.apply(img_np)

        # Normalize to [0, 1]
        img_t = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0

        # Load vessel confidence map
        conf_np = self._load_conf_map(sample["image_path"])
        conf_t = torch.from_numpy(conf_np).float().unsqueeze(0)  # [1, H/8, W/8]

        return {
            "image": img_t,
            "vessel_conf": conf_t,
            "path": sample["image_path"],
        }


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def train_epoch(
    model: JEPAModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    config: Dict,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    accum_steps = int(config["training"].get("accum_steps", 1))
    clip_grad = float(config["training"].get("clip_grad_norm", 1.0))
    alpha = float(config["loss"].get("vessel_alpha", 1.0))
    mask_scale = tuple(config["training"].get("mask_scale", [0.15, 0.7]))

    running_loss = 0.0
    running_layers = {}

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1} [Train]")

    for batch_idx, batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        conf = batch["vessel_conf"].to(device, non_blocking=True)
        B, C, H, W = images.shape

        # Generate block masks
        masks = torch.stack([
            generate_block_mask(H, W, mask_scale=mask_scale, device=images.device)
            for _ in range(B)
        ], dim=0)

        # Apply masks
        masked_images = apply_mask(images, masks)

        with torch.amp.autocast("cuda", enabled=config["training"].get("use_amp", True),
                                dtype=torch.bfloat16 if config["training"].get("precision", "bfloat16") == "bfloat16" else torch.float16):
            output = model(masked_images, images, masks, conf, alpha=alpha)
            loss = output["loss"]

        if not torch.isfinite(loss):
            continue

        loss_scaled = loss / accum_steps

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(dataloader)):
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_val = float(loss.detach().item())
        running_loss += loss_val
        for k, v in output["loss_per_layer"].items():
            running_layers[k] = running_layers.get(k, 0.0) + v

        denom = batch_idx + 1
        pbar.set_postfix({"loss": running_loss / denom})

    n = max(1, len(dataloader))
    stats = {"loss": running_loss / n}
    for k, v in running_layers.items():
        stats[k] = v / n
    return stats


def validate_epoch(
    model: JEPAModel,
    dataloader: DataLoader,
    epoch: int,
    config: Dict,
) -> Dict[str, float]:
    model.eval()
    alpha = float(config["loss"].get("vessel_alpha", 1.0))
    mask_scale = tuple(config["training"].get("mask_scale", [0.15, 0.7]))

    running_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1} [Val]")
        for batch_idx, batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            conf = batch["vessel_conf"].to(device, non_blocking=True)
            B, C, H, W = images.shape

            masks = torch.stack([
                generate_block_mask(H, W, mask_scale=mask_scale, device=images.device)
                for _ in range(B)
            ], dim=0)
            masked_images = apply_mask(images, masks)

            with torch.amp.autocast("cuda", enabled=config["training"].get("use_amp", True),
                                    dtype=torch.bfloat16 if config["training"].get("precision", "bfloat16") == "bfloat16" else torch.float16):
                output = model(masked_images, images, masks, conf, alpha=alpha)
                loss = output["loss"]

            running_loss += float(loss.item())
            pbar.set_postfix({"loss": running_loss / max(1, batch_idx + 1)})

    n = max(1, len(dataloader))
    return {"loss": running_loss / n}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(config: Dict):
    experiment_name = config["experiment_name"]
    log_dir = config["logging"]["log_dir"].format(experiment_name=experiment_name)
    checkpoint_dir = config["logging"]["checkpoint_dir"].format(experiment_name=experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.dump(config, f, sort_keys=False)

    writer = SummaryWriter(log_dir=log_dir)

    # Dataset
    train_dataset = XCADensePretrainDataset(
        json_path=config["data"]["json_path"],
        split="train",
        root_dir=config["data"].get("root_dir", "."),
        img_size=config["data"]["img_size"],
        conf_dir=config["data"].get("conf_dir"),
        compute_conf_on_the_fly=config["data"].get("compute_conf_on_the_fly", False),
        sources=tuple(config["data"].get("sources", ["syntax", "stenoza", "cadica", "extra", "coronarydominance"])),
        max_samples=config["data"].get("max_samples", "all"),
    )
    val_dataset = XCADensePretrainDataset(
        json_path=config["data"]["json_path"],
        split="validation" if "validation" in json.load(open(config["data"]["json_path"])) else "val",
        root_dir=config["data"].get("root_dir", "."),
        img_size=config["data"]["img_size"],
        conf_dir=config["data"].get("conf_dir"),
        compute_conf_on_the_fly=config["data"].get("compute_conf_on_the_fly", False),
        sources=tuple(config["data"].get("sources", ["syntax", "stenoza", "cadica", "extra", "coronarydominance"])),
        max_samples=config["data"].get("max_samples", "all"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", torch.cuda.is_available()),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", torch.cuda.is_available()),
        drop_last=False,
    )

    # Model
    model = JEPAModel(
        backbone_name=config["model"]["backbone"],
        in_channels=config["model"].get("in_channels", 1),
        pretrained=config["model"].get("pretrained", False),
        predictor_hidden_dim=config["model"].get("predictor_hidden_dim"),
        deep_layers=tuple(config["model"].get("deep_layers", [0, 1, 2, 3])),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[JEPA] Model: {config['model']['backbone']}")
    print(f"       Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"       Trainable:    {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.04)),
    )

    num_epochs = int(config["training"]["epochs"])
    warmup_epochs = min(int(config["training"].get("warmup_epochs", 5)), max(1, num_epochs - 1))
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler("cuda", enabled=(config["training"].get("use_amp", True) and config["training"].get("precision", "bfloat16") == "float16"))

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    last_checkpoint = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    if os.path.isfile(last_checkpoint):
        ckpt = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Resume] Epoch {start_epoch}, best val loss {best_val_loss:.6f}")

    # Training loop
    patience = int(config["training"].get("patience", 0))
    epochs_no_improve = 0

    for epoch in range(start_epoch, num_epochs):
        train_stats = train_epoch(model, train_loader, optimizer, epoch, config, scaler)
        val_stats = validate_epoch(model, val_loader, epoch, config)

        scheduler.step()

        # Logging
        writer.add_scalar("Loss/train", train_stats["loss"], epoch)
        writer.add_scalar("Loss/val", val_stats["loss"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        for k, v in train_stats.items():
            if k != "loss":
                writer.add_scalar(f"Layers/train_{k}", v, epoch)

        # Checkpoint
        is_best = val_stats["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_stats["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        else:
            epochs_no_improve += 1

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }, last_checkpoint)

        patience_msg = f" | Patience: {epochs_no_improve}/{patience}" if patience > 0 else ""
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_stats['loss']:.6f} | val_loss={val_stats['loss']:.6f} | "
            f"best_val_loss={best_val_loss:.6f}{patience_msg}"
        )

        if patience > 0 and epochs_no_improve >= patience:
            print(f"[Early stopping] No improvement for {patience} epochs.")
            break

    writer.close()
    print(f"\nDone! Best val loss: {best_val_loss:.6f}")
    print(f"Best model: {os.path.join(checkpoint_dir, 'best_model.pth')}")


if __name__ == "__main__":
    config = {
        "experiment_name": "jepa_efficientnetv2_s",
        "logging": {
            "log_dir": "runs/{experiment_name}",
            "checkpoint_dir": "checkpoints/{experiment_name}",
        },
        "data": {
            "json_path": "data/ARCADE/processed/dataset.json",
            "root_dir": ".",
            "conf_dir": None,  # Set to "data/conf_maps" if precomputed
            "compute_conf_on_the_fly": False,  # Set True for prototyping
            "img_size": 256,
            "batch_size": 16,
            "num_workers": 4,
            "pin_memory": True,
            "sources": ["syntax", "stenoza", "cadica", "extra", "coronarydominance"],
            "max_samples": "all",
        },
        "model": {
            "backbone": "tf_efficientnetv2_s",
            "in_channels": 1,
            "pretrained": False,
            "predictor_hidden_dim": None,
            "deep_layers": [0, 1, 2, 3],
        },
        "training": {
            "epochs": 100,
            "learning_rate": 5e-4,
            "weight_decay": 0.04,
            "warmup_epochs": 5,
            "scheduler": "Warmup + CosineAnnealingLR",
            "use_amp": True,
            "precision": "bfloat16",
            "accum_steps": 2,
            "clip_grad_norm": 1.0,
            "patience": 20,
            "mask_scale": [0.15, 0.7],
        },
        "loss": {
            "vessel_alpha": 1.0,
        },
    }

    main(config)