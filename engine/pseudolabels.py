from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.filters import frangi
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

XA170K_ROOT = os.path.join(project_root, "XA-170K", "dataset")
OUTPUT_HESSIAN = os.path.join(project_root, "data", "XA-170K_hessian")
OUTPUT_GROWCUT = os.path.join(project_root, "data", "XA-170K_growcut")


# ============================================================================
# Stage 1: Multi-scale Hessian vesselness (Frangi)
# ============================================================================

def compute_vesselness(
    image_np: np.ndarray,
    sigmas: Tuple[float, ...] = (0.8, 1.2, 1.6, 2.0, 2.5, 3.0),
    black_ridges: bool = True,
) -> np.ndarray:
    """Multi-scale Frangi vesselness filter on enhanced X-ray angiography.

    Args:
        image_np: uint8 enhanced grayscale image.
        sigmas: Gaussian sigma values — at 512px, 0.8≈1.5px, 3.0≈6px vessel.
        black_ridges: True if vessels are dark on bright background (standard XA).

    Returns:
        vesselness: float32 [0, 1] probability map.
    """
    image_f = image_np.astype(np.float64)
    all_scales = []

    for sigma in sigmas:
        try:
            v = frangi(image_f, sigmas=(sigma,), alpha=0.1, beta=0.1, gamma=5,
                       black_ridges=black_ridges)
            all_scales.append(v)
        except Exception:
            continue

    if not all_scales:
        return np.zeros_like(image_np, dtype=np.float32)

    vesselness = np.max(all_scales, axis=0)
    valid = vesselness > 0
    if valid.any():
        vmax = vesselness.max()
        vesselness[valid] = vesselness[valid] / vmax
    vesselness = np.clip(vesselness, 0, 1)

    return vesselness.astype(np.float32)


# ============================================================================
# Stage 2: Grow Cut refinement (binary masks)
# ============================================================================

def grow_cut(
    image_np: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    max_iter: int = 80,
    beta: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Grow Cut algorithm as described in the paper (Section III-A-3).

    "Grow Cut [1] is an alternative to Graph Cut, yet with much better
    performance. Each image pixel is formulated as a cell of certain type
    (foreground, background, or undefined). As the algorithm proceeds,
    these cells compete to dominate the image domain."

    Implementation detail:
      - Labels: 0=background, 1=foreground, 2=undetermined.
      - Attack strength:  F(p, q) = 1.0 - beta * |I(p) - I(q)| / 255
      - A cell q attacks cell p if: attack = F(p,q) * strength(q) > strength(p)

    Args:
        image_np: uint8 CLAHE-enhanced grayscale image.
        fg_mask: bool array of reliable foreground seed pixels.
        bg_mask: bool array of reliable background seed pixels.
        max_iter: maximum iterations.
        beta: attack attenuation factor (default 1.0).

    Returns:
        fg_result: bool array, True = foreground (vessel).
    """
    h, w = image_np.shape
    img_f = image_np.astype(np.float32)

    labels = np.full((h, w), 2, dtype=np.int8)
    labels[fg_mask] = 1
    labels[bg_mask] = 0

    strength = np.zeros((h, w), dtype=np.float32)
    strength[fg_mask | bg_mask] = 1.0

    max_diff = 255.0

    for it in range(max_iter):
        changed = False

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue

                y0_cur = max(0, -dy)
                y1_cur = h - max(0, dy)
                x0_cur = max(0, -dx)
                x1_cur = w - max(0, dx)

                y0_nb = y0_cur + dy
                y1_nb = y1_cur + dy
                x0_nb = x0_cur + dx
                x1_nb = x1_cur + dx

                cur = (slice(y0_cur, y1_cur), slice(x0_cur, x1_cur))
                nb = (slice(y0_nb, y1_nb), slice(x0_nb, x1_nb))

                cur_labels = labels[cur]
                cur_strength = strength[cur]
                nb_labels = labels[nb]
                nb_strength = strength[nb]

                undet = cur_labels == 2
                if not undet.any():
                    continue

                labeled = nb_labels != 2
                if not labeled.any():
                    continue

                cur_intensity = img_f[cur]
                nb_intensity = img_f[nb]
                diff = np.abs(cur_intensity - nb_intensity) / max_diff
                attack = (1.0 - beta * diff) * nb_strength

                eligible = undet & labeled & (attack > cur_strength)
                if eligible.any():
                    strength[cur][eligible] = attack[eligible]
                    labels[cur][eligible] = nb_labels[eligible]
                    changed = True

        if not changed:
            break

    # Filter small segments (paper: "segments whose number of points are
    # smaller than a given value are omitted")
    fg = labels == 1
    fg = _remove_small_components(fg, min_size=50)

    return fg, it + 1  # return (mask, iterations), it + 1  # return (mask, iterations)


def _remove_small_components(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove connected components smaller than min_size pixels."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    result = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            result |= labels == i
    return result


# ============================================================================
# Pipeline: Hessian + Grow Cut
# ============================================================================

def process_image(
    image_path: str,
    image_size: int = 512,
    black_ridges: bool = True,
    fg_pct: float = 97,
    bg_pct: float = 55,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full pipeline on a single image.

    Returns:
        original: uint8 CLAHE-enhanced image.
        hessian: float32 [0, 1] vesselness map.
        growcut: bool binary vessel mask.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load {image_path}")

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # Stage 1: Hessian vesselness
    hessian_map = compute_vesselness(img_clahe, black_ridges=black_ridges)

    # Stage 2: Grow Cut with percentile-based seeds
    fg_thresh = np.percentile(hessian_map, fg_pct)
    bg_thresh = np.percentile(hessian_map, bg_pct)
    fg_seed = hessian_map >= fg_thresh
    bg_seed = hessian_map <= bg_thresh
    binary_mask, gc_iters = grow_cut(img_clahe, fg_seed, bg_seed)

    return img_clahe, hessian_map, binary_mask


# ============================================================================
# Batch processing
# ============================================================================

def batch_process(
    input_dir: str = XA170K_ROOT,
    image_size: int = 512,
    max_images: Optional[int] = None,
):
    """Run full pipeline on all XA-170K images and save results."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_paths = sorted(
        [str(p) for p in Path(input_dir).rglob("*") if p.suffix.lower() in exts],
        key=lambda p: p,
    )
    if max_images is not None:
        all_paths = all_paths[:max_images]

    out_hessian = Path(OUTPUT_HESSIAN)
    out_growcut = Path(OUTPUT_GROWCUT)

    print(f"Processing {len(all_paths)} images from {input_dir}")
    print(f"  Hessian output: {out_hessian}")
    print(f"  GrowCut output: {out_growcut}")

    for img_path in tqdm(all_paths, desc="Pipeline"):
        try:
            img_clahe, hessian_map, binary_mask = process_image(img_path, image_size)

            rel = os.path.relpath(img_path, input_dir)
            base, _ = os.path.splitext(rel)

            # Save Hessian soft map
            dst_h = out_hessian / f"{base}_hessian.png"
            dst_h.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_h), (hessian_map * 255).astype(np.uint8))

            # Save Grow Cut binary mask
            dst_g = out_growcut / f"{base}_growcut.png"
            dst_g.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_g), binary_mask.astype(np.uint8) * 255)

        except Exception as e:
            print(f"  Failed {img_path}: {e}")

    print(f"Done. Results in {out_hessian} and {out_growcut}")


# ============================================================================
# Visualization
# ============================================================================

def visualize_samples(
    input_dir: str = XA170K_ROOT,
    image_size: int = 512,
    num_samples: int = 8,
    output_path: str = "pseudolabels_preview.png",
):
    """Process random images and plot Data | Hessian | Grow Cut."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_paths = sorted(
        [str(p) for p in Path(input_dir).rglob("*") if p.suffix.lower() in exts],
        key=lambda p: p,
    )
    if len(all_paths) > num_samples * 3:
        selected = sorted(random.sample(all_paths, num_samples))
    else:
        selected = all_paths[:num_samples]

    n = len(selected)
    cols = 3
    fig, axes = plt.subplots(n, cols, figsize=(4 * cols, 4 * n))

    col_titles = ["Data (CLAHE)", "Hessian (Frangi soft)", "Grow Cut (binary)"]
    if n == 1:
        axes = axes.reshape(1, -1)
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14, fontweight="bold")

    for i, img_path in enumerate(selected):
        try:
            img_clahe, hessian_map, binary_mask = process_image(img_path, image_size)

            axes[i, 0].imshow(img_clahe, cmap="gray", vmin=0, vmax=255)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(hessian_map, cmap="hot", vmin=0, vmax=1)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].axis("off")

            label = os.path.basename(img_path)[:30]
            axes[i, 0].set_ylabel(label, fontsize=9)
        except Exception as e:
            print(f"  Failed {img_path}: {e}")

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Preview saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pseudolabel generation: Frangi + Grow Cut")
    parser.add_argument("--input", type=str, default=XA170K_ROOT)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--mode", type=str, default="visualize",
                        choices=["visualize", "batch"],
                        help="visualize: show random samples; batch: process all")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit in batch mode (None = all)")
    parser.add_argument("--output", type=str, default="pseudolabels_preview.png",
                        help="Output path for visualization grid")
    args = parser.parse_args()

    if args.mode == "visualize":
        visualize_samples(
            input_dir=args.input,
            image_size=args.image_size,
            num_samples=args.num_samples,
            output_path=args.output,
        )
    else:
        batch_process(
            input_dir=args.input,
            image_size=args.image_size,
            max_images=args.max_images,
        )
