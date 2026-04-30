#!/usr/bin/env python3
"""
Train UNeXt on Extra dataset (all-vessel labels) as a distillation teacher.

Extra contains full coronary vessel segmentations (not just main arteries).
The trained model replaces Frangi for SSL pretraining.

Usage:
    python engine/train_unext_arcade.py
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.unext import get_model
from engine.UnExT_train import (
    train_one_epoch,
    validate_one_epoch,
    _build_focal_loss,
    _build_dice_loss,
)
from data.dataloader import Transforms
from utils.helpers import set_seed


# ── Paths ──
EXTRA_IMAGE_DIR = "data/Extra/images"
EXTRA_MASK_DIR = "data/Extra/masks"
LOG_DIR = "runs/unext_extra"
CHECKPOINT_DIR = "checkpoints/unext_extra"
TEACHER_EXPORT = "checkpoints/unext_extra/teacher_model.pth"


class ExtraSegDataset(Dataset):
    """Loads Extra image/mask pairs (all-vessel labels)."""

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        # Find matching pairs (same filename in both dirs)
        self.samples = []
        for fname in sorted(os.listdir(image_dir)):
            img_path = self.image_dir / fname
            msk_path = self.mask_dir / fname
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif") and msk_path.exists():
                self.samples.append((str(img_path), str(msk_path)))

        self.transform = transform
        print(f"ExtraSegDataset: {len(self.samples)} image/mask pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, msk_path = self.samples[idx]
        image = Image.open(img_path).convert("L")
        mask = Image.open(msk_path).convert("L")

        if self.transform is not None:
            return self.transform(image, mask)

        return image, mask


def train_unext_extra(
    image_size: int = 256,
    batch_size: int = 8,
    epochs: int = 300,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    num_workers: int = 4,
    val_split: float = 0.2,
    focal_gamma: float = 2.0,
    device: torch.device = None,
):
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Transforms ──
    train_transform = Transforms(
        image_size=image_size,
        training=True,
        hflip_prob=0.5,
        vflip_prob=0.2,
        rotation_degrees=15.0,
        brightness=0.12,
        contrast=0.12,
        blur_prob=0.15,
        blur_kernel_size=5,
        normalize=True,
    )
    val_transform = Transforms(
        image_size=image_size,
        training=False,
        normalize=True,
    )

    # ── Full dataset → train/val split ──
    full = ExtraSegDataset(EXTRA_IMAGE_DIR, EXTRA_MASK_DIR)
    indices = list(range(len(full)))
    random.shuffle(indices)
    val_count = max(1, int(len(indices) * val_split))
    train_idx, val_idx = indices[val_count:], indices[:val_count]

    train_dataset = torch.utils.data.Subset(
        [(full[i][0], full[i][1]) for i in train_idx], ...  # won't work
    )

    # Better: manual split
    train_samples = [(full.samples[i][0], full.samples[i][1]) for i in train_idx]
    val_samples = [(full.samples[i][0], full.samples[i][1]) for i in val_idx]

    class SplitDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            img_path, msk_path = self.samples[idx]
            image = Image.open(img_path).convert("L")
            mask = Image.open(msk_path).convert("L")
            if self.transform is not None:
                return self.transform(image, mask)
            return image, mask

    train_dataset = SplitDataset(train_samples, train_transform)
    val_dataset = SplitDataset(val_samples, val_transform)

    # ── DataLoaders ──
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_dataset)}   Val: {len(val_dataset)}")

    # ── Model ──
    model = get_model(in_channels=1, num_classes=1, device=device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    focal_loss = _build_focal_loss(gamma=focal_gamma)
    dice_loss = _build_dice_loss()

    # ── Logging ──
    log_dir = Path(LOG_DIR)
    checkpoint_dir = Path(CHECKPOINT_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_f1 = -1.0
    history = []

    # ── Training Loop ──
    for epoch in range(epochs):
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            focal_loss=focal_loss,
            dice_loss=dice_loss,
            device=device,
            optimizer=optimizer,
        )
        val_stats = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            focal_loss=focal_loss,
            dice_loss=dice_loss,
            device=device,
            writer=writer,
            epoch=epoch,
        )

        scheduler.step()

        writer.add_scalar("Loss/train", train_stats["loss"], epoch)
        writer.add_scalar("Loss/val", val_stats["loss"], epoch)
        writer.add_scalar("F1/train", train_stats["f1"], epoch)
        writer.add_scalar("F1/val", val_stats["f1"], epoch)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_f1": train_stats["f1"],
            "val_f1": val_stats["f1"],
        })

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train_loss={train_stats['loss']:.4f} | val_loss={val_stats['loss']:.4f} | "
            f"train_f1={train_stats['f1']:.4f} | val_f1={val_stats['f1']:.4f}"
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_f1": best_val_f1,
            },
            checkpoint_dir / "last.pth",
        )

        if val_stats["f1"] >= best_val_f1:
            best_val_f1 = val_stats["f1"]
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            torch.save(model.state_dict(), TEACHER_EXPORT)
            print(f"  >>> New best F1={best_val_f1:.4f} — teacher exported to {TEACHER_EXPORT}")

    writer.close()
    torch.save(model.state_dict(), TEACHER_EXPORT)
    print(f"\nDone. Best val F1: {best_val_f1:.4f}")
    print(f"Teacher: {TEACHER_EXPORT}")
    return history


if __name__ == "__main__":
    train_unext_extra()
