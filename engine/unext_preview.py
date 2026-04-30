#!/usr/bin/env python3
"""
UNeXt teacher preview — plot segmentations on Extra images (trained domain)
and ARCADE images (cross-domain generalization test).

Usage:
    python engine/unext_preview.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.unext import get_model
from data.dataloader import Transforms


# ── Paths ──
DATASET_JSON = "data/ARCADE/processed/dataset.json"
EXTRA_IMAGE_DIR = "data/Extra/images"
EXTRA_MASK_DIR = "data/Extra/masks"
WEIGHTS_PATH = "Unext_Weights/teacher_model.pth"
OUTPUT_DIR = "unext_preview"
N_IMAGES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_teacher(weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained UNeXt teacher model."""
    model = get_model(in_channels=1, num_classes=1, device=device)
    model.eval()

    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    print(f"Teacher loaded from {weights_path}")
    return model


def predict(
    model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Run UNeXt inference, return probability map [0,1]."""
    with torch.no_grad():
        input_batch = img_tensor.unsqueeze(0).to(device)  # [1, 1, H, W]
        logits = model(input_batch)
        probs = torch.sigmoid(logits.float())
        return probs.squeeze().cpu().numpy()  # [H, W]


def save_preview(
    orig_display: np.ndarray,
    prob_map: np.ndarray,
    title: str,
    out_path: str,
    ground_truth: np.ndarray = None,
):
    """Save 3-panel (or 4-panel if ground truth available) preview."""
    binary = (prob_map > 0.5).astype(np.uint8) * 255
    pred_pct = (prob_map > 0.5).mean() * 100

    if ground_truth is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        gt_binary = (ground_truth > 0).astype(np.uint8) * 255
        axes[3].imshow(gt_binary, cmap="gray")
        axes[3].set_title("Ground Truth")
        axes[3].axis("off")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_display, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    im1 = axes[1].imshow(prob_map, cmap="hot", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"UNeXt [{pred_pct:.1f}% >0.5]")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(binary, cmap="gray")
    axes[2].set_title("Binary (thresh=0.5)")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}  ({pred_pct:.1f}% pixels > 0.5)")


def plot_teacher_preview(num_images: int = N_IMAGES):
    """Run UNeXt on Extra + ARCADE images, save previews."""
    device = torch.device(DEVICE)
    model = load_teacher(WEIGHTS_PATH, device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Transform for inference (resize + normalize only)
    transform = Transforms(image_size=256, training=False, normalize=True)

    count = 0

    # ── 1. Extra images (trained domain, has ground truth) ──
    print("\n=== Extra (training domain) ===")
    extra_files = sorted(Path(EXTRA_IMAGE_DIR).glob("*.png"))[:num_images // 2]
    for img_path in extra_files:
        if count >= num_images:
            break

        msk_path = Path(EXTRA_MASK_DIR) / img_path.name
        if not msk_path.exists():
            continue

        pil_img = Image.open(img_path).convert("L")
        pil_mask = Image.open(msk_path).convert("L")
        img_tensor, _ = transform(pil_img, pil_mask)

        prob_map = predict(model, img_tensor, device)
        orig_display = (img_tensor.squeeze().cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        gt_mask = np.array(pil_mask.resize((256, 256), Image.NEAREST))

        out_path = os.path.join(OUTPUT_DIR, f"extra_{img_path.stem}.png")
        save_preview(orig_display, prob_map, f"Extra: {img_path.name}", out_path, gt_mask)
        count += 1

    # ── 2. ARCADE train images (cross-domain) ──
    print("\n=== ARCADE (cross-domain) ===")
    with open(DATASET_JSON, "r") as f:
        data = json.load(f)

    train = data.get("train", {})
    for source in sorted(train.keys()):
        if count >= num_images:
            break
        entries = sorted(train[source].items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        for key, info in entries:
            if count >= num_images:
                break
            img_path = os.path.normpath(info["data"])
            if not os.path.exists(img_path):
                continue

            pil_img = Image.open(img_path).convert("L")
            img_tensor, _ = transform(pil_img, pil_img)

            prob_map = predict(model, img_tensor, device)
            orig_display = (img_tensor.squeeze().cpu().numpy() * 0.5 + 0.5).clip(0, 1)

            out_path = os.path.join(OUTPUT_DIR, f"{source}_{key}.png")
            save_preview(orig_display, prob_map, f"{source}: {key}", out_path)
            count += 1

    print(f"\nDone. Saved {count} previews to {OUTPUT_DIR}/")


if __name__ == "__main__":
    plot_teacher_preview(num_images=N_IMAGES)
