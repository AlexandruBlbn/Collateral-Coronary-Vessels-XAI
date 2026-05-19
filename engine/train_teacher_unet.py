import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Extra"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "teacher_unet"


class VesselDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return image, mask.float()


def get_training_augmentation(size=512):
    return albu.Compose([
        albu.Resize(size, size),

        albu.OneOf([
            albu.RandomCrop(int(size * 0.9), int(size * 0.9)),
            albu.CenterCrop(int(size * 0.85), int(size * 0.85)),
            albu.RandomResizedCrop(size, size, scale=(0.7, 1.0)),
        ], p=0.6),

        albu.OneOf([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
        ], p=0.7),

        albu.RandomRotate90(p=0.7),

        albu.OneOf([
            albu.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            albu.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            albu.OpticalDistortion(distort_limit=0.05, p=0.5),
        ], p=0.4),

        albu.OneOf([
            albu.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            albu.RandomGamma(gamma_limit=(80, 120)),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        ], p=0.8),

        albu.OneOf([
            albu.GaussNoise(var_limit=(10.0, 50.0)),
            albu.GaussianBlur(blur_limit=(3, 7)),
            albu.MedianBlur(blur_limit=5),
        ], p=0.5),

        albu.CoarseDropout(max_holes=4, max_height=int(size * 0.15), max_width=int(size * 0.15), fill_value=0, p=0.3),

        albu.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])


def get_validation_augmentation(size=512):
    return albu.Compose([
        albu.Resize(size, size),
        albu.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred_probs = torch.sigmoid(pred)
        pred_flat = pred_probs.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice)


def dice_score(pred, target, smooth=1e-5):
    pred_probs = torch.sigmoid(pred)
    pred_binary = (pred_probs > 0.5).float()

    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )
    return dice


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, writer):
    model.train()
    total_loss = 0
    total_dice = 0

    loop = tqdm(loader, desc=f"Epoch {epoch}")
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        d = dice_score(logits, masks).item()
        total_loss += loss.item()
        total_dice += d

        loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)

    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/dice", avg_dice, epoch)
        if scheduler:
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

    return avg_loss, avg_dice


def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Val {epoch}")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            d = dice_score(logits, masks).item()
            total_loss += loss.item()
            total_dice += d

            loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)

    if writer:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/dice", avg_dice, epoch)

    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--weights", type=str, default="imagenet")
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([str(p) for p in (DATA_DIR / "images").glob("*.png")])
    mask_paths = sorted([str(p) for p in (DATA_DIR / "masks").glob("*.png")])

    assert len(image_paths) == len(mask_paths) > 0, "No images/masks found"
    print(f"Found {len(image_paths)} samples")

    indices = np.random.permutation(len(image_paths))
    split = int(len(indices) * (1 - args.val_split))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    train_dataset = VesselDataset(train_images, train_masks, get_training_augmentation(args.size))
    val_dataset = VesselDataset(val_images, val_masks, get_validation_augmentation(args.size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.weights,
        in_channels=3,
        classes=1,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(CHECKPOINT_DIR / "logs" / timestamp)

    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, writer)
        val_loss, val_dice = validate(model, val_loader, criterion, device, epoch, writer)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"  >>> Saved best model (dice: {val_dice:.4f})")

        if epoch % 25 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"model_epoch_{epoch}.pth")

    print(f"Done. Best val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
