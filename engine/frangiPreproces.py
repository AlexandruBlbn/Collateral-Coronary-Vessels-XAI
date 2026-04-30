# engine/frangiPreproces.py
from PIL import Image
import numpy as np
import cv2
from skimage.filters import frangi
from skimage.morphology import reconstruction
import matplotlib.pyplot as plt
import json
import os


def _auto_crop_black_borders(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Detect and crop black borders from XCA images."""
    mask = img > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        # Add small margin so we don't cut into content
        margin = 8
        y_min = max(0, y_min - margin)
        y_max = min(img.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(img.shape[1], x_max + margin)
        return img[y_min:y_max, x_min:x_max]
    return img


def _soft_fill_hollow_vessels(vesselness: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Soft morphological filling of hollow vessel interiors.

    Uses reconstruction by dilation: seed = eroded, mask = original.
    Peaks in the mask propagate into valleys of the seed, filling
    hollow vessel centers while preserving the continuous [0,1] output.
    """
    # Erode the vesselness to create a seed (remove thin edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed = cv2.erode(vesselness, kernel, iterations=1)

    # Reconstruction by dilation: seed values propagate into mask
    # seed <= mask is required (seed is eroded, so it's ≤ original)
    rec = reconstruction(seed, vesselness, method='dilation')

    # Blend: where rec > original, we filled a hollow region
    fill_amount = np.clip(rec - vesselness, 0.0, None)
    fill_amount = cv2.GaussianBlur(fill_amount, (0, 0), sigmaX=sigma)
    filled = vesselness + fill_amount * 0.5
    filled = np.clip(filled, 0.0, 1.0)

    return filled


def frangi_filter(
    img_path="data/ARCADE/processed/syntax/data/1.png",
    img_size=512,
    sigmas=range(1, 14, 1),
    alpha=0.2,
    beta=0.9,
    gamma=25,
    black_ridges=True,
    # --- Pre-processing ---
    auto_crop_borders=True,
    border_threshold=10,
    # --- Noise handling ---
    clahe_clip_limit=2.0,
    pre_blur_sigma=15.0,
    # --- Soft fill hollow vessels ---
    fill_hollow=True,
    fill_sigma=3.0,
):
    """
    Frangi vesselness returning a SOFT continuous map [0, 1].

    Fixes:
    1. Black borders → auto-crop before processing
    2. Noise → CLAHE + stronger blur
    3. Hollow thick vessels → morphological reconstruction fill

    No hard thresholding, no binary masking — returns soft targets.
    """
    # --- Load ---
    img = Image.open(img_path).convert('L')
    img = np.array(img, dtype=np.float32)

    # --- 1. Auto-crop black borders ---
    if auto_crop_borders:
        img = _auto_crop_black_borders(img, threshold=border_threshold)

    # --- 2. CLAHE for contrast normalization (reduces noise impact) ---
    if clahe_clip_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        img_uint8 = img.astype(np.uint8)
        img = clahe.apply(img_uint8).astype(np.float32)

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Resize
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    # Strong Gaussian blur to suppress noise
    img = cv2.GaussianBlur(img, (5, 5), sigmaX=pre_blur_sigma)

    # --- Frangi vesselness ---
    vesselness = frangi(
        img, sigmas=sigmas,
        alpha=alpha, beta=beta, gamma=gamma,
        black_ridges=black_ridges,
    )

    # --- 3. Soft fill hollow vessel interiors ---
    if fill_hollow:
        vesselness = _soft_fill_hollow_vessels(vesselness, sigma=fill_sigma)

    # Soft contrast stretch
    p_low, p_high = np.percentile(vesselness, (1, 99.9))
    vesselness = np.clip(vesselness, 0.0, p_high)

    # Normalize to [0, 1]
    v_max = vesselness.max()
    if v_max > 0:
        vesselness = vesselness / v_max

    return vesselness


# ── Plot N train samples as soft-target PNG previews ──
DATASET_JSON = "data/ARCADE/processed/dataset.json"
OUTPUT_DIR = "frangi_preview"
N_IMAGES = 10


def plot_frangi_preview(num_images: int = N_IMAGES):
    with open(DATASET_JSON, "r") as f:
        data = json.load(f)

    train = data.get("train", {})
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    count = 0
    for source in sorted(train.keys()):
        if count >= num_images:
            break
        entries = sorted(train[source].items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        for key, info in entries:
            if count >= num_images:
                break
            img_path = info["data"]
            abs_path = os.path.normpath(img_path)

            if not os.path.exists(abs_path):
                print(f"  SKIP: {img_path} not found")
                continue

            print(f"[{count+1}/{num_images}] Processing {img_path} ...")

            vesselness = frangi_filter(img_path=abs_path)
            orig = np.array(Image.open(abs_path).convert('L'), dtype=np.uint8)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(orig, cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis('off')

            im = axes[1].imshow(vesselness, cmap='hot', vmin=0.0, vmax=1.0)
            axes[1].set_title("Soft Frangi Target [0,1]")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            out_name = f"{source}_{key}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            count += 1

    print(f"\nDone. Saved {count} soft-target previews to {OUTPUT_DIR}/")


if __name__ == "__main__":
    plot_frangi_preview(num_images=N_IMAGES)
