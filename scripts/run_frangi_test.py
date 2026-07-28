#!/usr/bin/env python
"""Run Frangi vesselness pipeline on a random ARCADE syntax sample and compare with GT."""

import sys, os, warnings
warnings.filterwarnings('ignore')

# Add project root to path so we can import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import *
from skimage import filters as skfilters

# --------------------------------------------------------------------------
# 1. Pick a random patient index (1–1500)
# --------------------------------------------------------------------------
set_seed(69)  # for reproducibility; remove for true randomness
idx = 168
print(f"Patient index: {idx}")

# --------------------------------------------------------------------------
# 2. Load image as grayscale
# --------------------------------------------------------------------------
img_path = f"data/ARCADE/processed/syntax/data/{idx}.png"
image = np.array(Image.open(img_path).convert('L'))
print(f"Image shape: {image.shape}")

# --------------------------------------------------------------------------
# 3. Load ground truth syntax mask as grayscale
# --------------------------------------------------------------------------
gt_path = f"data/ARCADE/processed/syntax/label/{idx}.png"
gt_mask = np.array(Image.open(gt_path).convert('L'))

# --------------------------------------------------------------------------
# 4. VasoMIM Frangi pipeline
# --------------------------------------------------------------------------

# Step A: Sato filter
sato = skfilters.sato(
    image.astype(np.uint8),
    sigmas=[1, 2, 3, 4],
    black_ridges=True,
    mode="reflect",
    cval=0
)

# Step B: Zero out borders (border=20)
BORDER = 20
h, w = sato.shape
sato[:BORDER, :] = 0
sato[-BORDER:, :] = 0
sato[:, :BORDER] = 0
sato[:, -BORDER:] = 0

# Step C: Convert to uint8
sato_u8 = sato.astype(np.uint8)

# Step D: Percentile thresholding at 92%
thresh_val = np.percentile(sato_u8, 92.0)
thresh = np.where(sato_u8 >= thresh_val, sato_u8, 0)

# Step E: Region growing from maximum intensity pixel
# (4-connected flood fill on nonzero pixels, output binary 0/255)
if np.any(thresh):
    seed_idx = np.unravel_index(np.argmax(thresh), thresh.shape)
    seed = (int(seed_idx[0]), int(seed_idx[1]))
else:
    seed = None

visited = np.zeros_like(thresh, dtype=bool)
frangi_mask = np.zeros_like(thresh, dtype=np.uint8)
if seed is not None:
    stack = [seed]
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while stack:
        x, y = stack.pop()
        if not (0 <= x < h and 0 <= y < w):
            continue
        if visited[x, y] or thresh[x, y] == 0:
            continue
        visited[x, y] = True
        frangi_mask[x, y] = 255
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and thresh[nx, ny] > 0:
                stack.append((nx, ny))

# --------------------------------------------------------------------------
# 5. Compute statistics
# --------------------------------------------------------------------------
total_pixels = image.size

# Frangi mask stats
frangi_positive = int(np.sum(frangi_mask > 0))
frangi_pct = 100.0 * frangi_positive / total_pixels

# GT syntax mask (binarize: > 0)
gt_binary = (gt_mask > 0).astype(np.uint8) * 255
gt_positive = int(np.sum(gt_binary > 0))
gt_pct = 100.0 * gt_positive / total_pixels

# Dice score
intersection = int(np.sum((frangi_mask > 0) & (gt_binary > 0)))
dice = (2.0 * intersection) / (frangi_positive + gt_positive) if (frangi_positive + gt_positive) > 0 else 0.0

# Overlap analysis
frangi_in_gt = int(np.sum((frangi_mask > 0) & (gt_binary > 0)))  # same as intersection
gt_in_frangi = frangi_in_gt  # same

print()
print("=" * 55)
print("Frangi Pipeline Statistics")
print("=" * 55)
print(f"  Patient index used:               {idx}")
print(f"  Image shape:                      {image.shape}")
print()
print(f"  Frangi mask:")
print(f"    Positive pixels:                {frangi_positive} / {total_pixels} ({frangi_pct:.3f}%)")
print()
print(f"  Ground truth syntax mask:")
print(f"    Positive pixels:                {gt_positive} / {total_pixels} ({gt_pct:.3f}%)")
print()
print(f"  Dice score:                       {dice:.6f}")
print()
print(f"  Overlap analysis:")
print(f"    GT pixels also in Frangi:       {gt_in_frangi} / {gt_positive} ({100.0 * gt_in_frangi / gt_positive:.3f}%)" if gt_positive > 0 else "    GT pixels also in Frangi:       0 / 0 (N/A)")
print(f"    Frangi pixels also in GT:       {frangi_in_gt} / {frangi_positive} ({100.0 * frangi_in_gt / frangi_positive:.3f}%)" if frangi_positive > 0 else "    Frangi pixels also in GT:       0 / 0 (N/A)")
print("=" * 55)

# --------------------------------------------------------------------------
# 6. Side-by-side visualization
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Frangi Pipeline vs GT Syntax — Patient {idx}", fontsize=14)

axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(frangi_mask, cmap='gray')
axes[1].set_title(f"Frangi Mask (Dice={dice:.4f})")
axes[1].axis('off')

axes[2].imshow(gt_binary, cmap='gray')
axes[2].set_title("GT Syntax Mask")
axes[2].axis('off')

plt.tight_layout()
out_path = f"scripts/frangi_comparison_{idx}.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved comparison: {out_path}")

# --------------------------------------------------------------------------
# 7. Save Frangi mask
# --------------------------------------------------------------------------
mask_path = f"scripts/frangi_mask_{idx}.png"
Image.fromarray(frangi_mask).save(mask_path)
print(f"Saved Frangi mask: {mask_path}")
print()

# --------------------------------------------------------------------------
# 8. Quick sanity check — what raw Frangi values look like
# --------------------------------------------------------------------------
print(f"Raw Sato stats (after border zeroing): min={sato.min():.4f}, max={sato.max():.4f}, mean={sato.mean():.4f}")
print(f"After uint8 convert:                  min={sato_u8.min()}, max={sato_u8.max()}")
print(f"92nd percentile threshold:            {thresh_val:.4f}")
print(f"Thresholded nonzero pixels:           {int(np.sum(thresh > 0))}")
