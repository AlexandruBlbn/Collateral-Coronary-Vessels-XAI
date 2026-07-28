#!/usr/bin/env python
"""
precompute_frangi_gpu.py

GPU-accelerated Frangi vesselness prior computation using PyTorch + CUDA.

Pipeline (per batch on GPU):
  1. Gaussian smoothing at 4 scales: sigma = [1, 2, 3, 4]
  2. Second derivatives via conv2d (dxx, dyy, dxy) matching np.gradient stencil
  3. Hessian eigenvalues → Sato vesselness (σ² * max(λ_largest, 0))
  4. Max across scales → zero out border (20 px) → percentile threshold (92%)
  5. Multi-seed region growing on CPU (n_seeds=8, min_seed_dist=50, flood-fill union)
  6. F.avg_pool2d to 14×14 → [0,1] float16 .npy

SMOKE TEST (run first — only 5 images):
  Compares multi-seed GPU Frangi vs single-seed GPU Frangi vs RCP priors.
  DO NOT run full dataset until smoke test passes (more nonzeros than single-seed).

Usage:
  python scripts/precompute_frangi_gpu.py            # runs smoke test only
  python scripts/precompute_frangi_gpu.py --full     # full precomputation
"""

import sys
import os
import time
import warnings
import traceback

warnings.filterwarnings("ignore")

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import *  # np, cv2, Image, tqdm, Path, torch, F, etc.
from skimage.filters import sato as sato_filter
from scipy.stats import pearsonr

# ── Globals / configuration ─────────────────────────────────────────────────
RCP_DIR = os.path.join(PROJECT_ROOT, "data", "pretrain", "priors")
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "pretrain", "dataset")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "pretrain", "priors_frangi")

BORDER = 20
PERCENTILE = 92.0
SIGMAS = [1, 2, 3, 4]
TARGET_SIZE = (14, 14)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMOKE_DIR = os.path.join(PROJECT_ROOT, "scripts", "smoke_test_frangi_multiseed")

# Process smallest → largest
DATASETS = [
    ("xcad", 1621),
    ("arcade", 2000),
    ("syntax", 2943),
    ("cadica", 6594),
    ("coronarydominance", 160320),
]


# ═══════════════════════════════════════════════════════════════════════════════
# GPU FRANGI (SATO) FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def _make_gaussian_kernel_2d(sigma: float, device: torch.device) -> torch.Tensor:
    """Create a normalized 2D Gaussian kernel for depthwise conv2d.
    Returns (1, 1, k, k) where k = 2*ceil(3*sigma)+1."""
    k_size = 2 * int(math.ceil(3 * sigma)) + 1
    x = torch.arange(k_size, dtype=torch.float32, device=device) - k_size // 2
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    return gauss_2d.view(1, 1, k_size, k_size)


def _make_derivative_kernels(device: torch.device):
    """Second derivative kernels matching np.gradient-style stencil.
    
    Returns dxx_k, dyy_k, dxy_k as (1, 1, kH, kW) tensors.
    
    dxx: central second difference in x (1×3): [1, -2, 1]
    dyy: central second difference in y (3×1): [1, -2, 1]^T
    dxy: cross derivative (3×3): [[-1, 0, 1], [0, 0, 0], [1, 0, -1]] / 4
    """
    # dxx = [1, -2, 1] as 1×3 kernel (applied as 2D conv along width)
    dxx = torch.tensor([[[[1.0, -2.0, 1.0]]]], device=device)
    # dyy = [1, -2, 1]^T as 3×1 kernel
    dyy = torch.tensor([[[[1.0], [-2.0], [1.0]]]], device=device)
    # dxy = cross-derivative 3×3 kernel / 4, shape (1, 1, 3, 3)
    dxy_raw = torch.tensor(
        [[[[-1.0, 0.0, 1.0],
           [0.0, 0.0, 0.0],
           [1.0, 0.0, -1.0]]]],
        device=device,
    )
    dxy = dxy_raw / 4.0
    return dxx, dyy, dxy


@torch.no_grad()
def gpu_sato_filter(
    images: torch.Tensor,  # (B, 1, H, W), float32 on GPU
    sigmas: list = SIGMAS,
    black_ridges: bool = True,
) -> torch.Tensor:
    """
    GPU Sato vesselness filter. Output: (B, 1, H, W) float32.

    Matches skimage.filters.sato(..., use_gaussian_derivatives=True) formula:
      For 2D, black_ridges=True:
        V_sigma = sigma^2 * max(lambda_largest, 0)
        V = max across sigmas
    """
    B, C, H, W = images.shape
    assert C == 1, f"Expected single-channel images, got {C}"

    # If black_ridges=False, negate image (as skimage does internally)
    if not black_ridges:
        images = -images

    # Precompute derivative kernels (same for all scales)
    dxx_k, dyy_k, dxy_k = _make_derivative_kernels(images.device)

    vesselness_max = torch.zeros(B, 1, H, W, device=images.device, dtype=torch.float32)

    for sigma in sigmas:
        # --- Gaussian smoothing ---
        gk = _make_gaussian_kernel_2d(sigma, images.device)
        pad = gk.shape[-1] // 2
        # Use reflect padding to match skimage mode='reflect'
        smoothed = F.pad(images, (pad, pad, pad, pad), mode="reflect")
        smoothed = F.conv2d(smoothed, gk)

        # --- Second derivatives ---
        # dxx: second derivative in x (horizontal)
        dxx = F.pad(smoothed, (1, 1, 0, 0), mode="reflect")
        dxx = F.conv2d(dxx, dxx_k)

        # dyy: second derivative in y (vertical)
        dyy = F.pad(smoothed, (0, 0, 1, 1), mode="reflect")
        dyy = F.conv2d(dyy, dyy_k)

        # dxy: cross derivative
        dxy = F.pad(smoothed, (1, 1, 1, 1), mode="reflect")
        dxy = F.conv2d(dxy, dxy_k)

        # --- Hessian eigenvalues (2D analytical formula) ---
        # lambda_largest  = (dxx + dyy + sqrt((dxx-dyy)^2 + 4*dxy^2)) / 2
        # lambda_smallest = (dxx + dyy - sqrt((dxx-dyy)^2 + 4*dxy^2)) / 2
        disc = torch.sqrt(
            (dxx - dyy) ** 2 + 4 * dxy ** 2 + 1e-10
        )  # add epsilon for numerical stability
        lambda_largest = (dxx + dyy + disc) / 2.0

        # --- Sato vesselness: sigma^2 * max(lambda_largest, 0) ---
        vals = (sigma ** 2) * torch.clamp(lambda_largest, min=0.0)

        # --- Max across scales ---
        vesselness_max = torch.maximum(vesselness_max, vals)

    return vesselness_max


# ═══════════════════════════════════════════════════════════════════════════════
# REGION GROWING (CPU)
# ═══════════════════════════════════════════════════════════════════════════════

def region_grow_mask(thresh: np.ndarray) -> np.ndarray:
    """
    4-connected flood-fill from the max-intensity pixel on a 2D thresholded array.

    Args:
        thresh: (H, W) uint8 or int array. Pixels > 0 are candidates.
    Returns:
        mask: (H, W) uint8, values 0 or 255.
    """
    H, W = thresh.shape

    if not np.any(thresh):
        return np.zeros((H, W), dtype=np.uint8)

    seed_idx = np.unravel_index(np.argmax(thresh), thresh.shape)
    seed = (int(seed_idx[0]), int(seed_idx[1]))

    visited = np.zeros((H, W), dtype=bool)
    mask = np.zeros((H, W), dtype=np.uint8)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    stack = [seed]
    while stack:
        x, y = stack.pop()
        if not (0 <= x < H and 0 <= y < W):
            continue
        if visited[x, y] or thresh[x, y] == 0:
            continue
        visited[x, y] = True
        mask[x, y] = 255
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < H
                and 0 <= ny < W
                and not visited[nx, ny]
                and thresh[nx, ny] > 0
            ):
                stack.append((nx, ny))

    return mask


def region_grow_multi_seed(
    thresh: np.ndarray,
    n_seeds: int = 8,
    min_seed_dist: int = 50,
) -> np.ndarray:
    """
    Region growing from multiple spatially diverse seeds.

    1. Find all nonzero pixel coordinates
    2. Sort by intensity (descending)
    3. Greedily select top-N seeds that are at least min_seed_dist pixels apart
    4. Run flood fill (4-connected) from each seed on the SAME thresholded image
    5. Union all results into one binary mask

    Args:
        thresh: (H, W) uint8 or int array. Pixels > 0 are candidates.
        n_seeds: Maximum number of seeds to use.
        min_seed_dist: Minimum L∞ distance between seeds (pixels).
    Returns:
        mask: (H, W) uint8, values 0 or 255.
    """
    H, W = thresh.shape

    if not np.any(thresh):
        return np.zeros((H, W), dtype=np.uint8)

    # Find candidate seed points: all nonzero pixels, sorted by intensity
    coords = np.argwhere(thresh > 0)
    intensities = thresh[coords[:, 0], coords[:, 1]]
    sorted_idx = np.argsort(-intensities)  # descending
    coords = coords[sorted_idx]

    # Greedily select seeds that are spatially diverse
    seeds = []
    for y, x in coords:
        too_close = False
        for sy, sx in seeds:
            if abs(y - sy) < min_seed_dist and abs(x - sx) < min_seed_dist:
                too_close = True
                break
        if not too_close:
            seeds.append((int(y), int(x)))
        if len(seeds) >= n_seeds:
            break

    # Flood fill from all seeds simultaneously (union)
    visited = np.zeros((H, W), dtype=bool)
    mask = np.zeros((H, W), dtype=np.uint8)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    stack = list(seeds)
    while stack:
        x, y = stack.pop()
        if not (0 <= x < H and 0 <= y < W):
            continue
        if visited[x, y] or thresh[x, y] == 0:
            continue
        visited[x, y] = True
        mask[x, y] = 255
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < H
                and 0 <= ny < W
                and not visited[nx, ny]
                and thresh[nx, ny] > 0
            ):
                stack.append((nx, ny))

    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# CPU REFERENCE (skimage Sato)
# ═══════════════════════════════════════════════════════════════════════════════

def cpu_sato_full_pipeline(
    image: np.ndarray,
    sigmas: list = SIGMAS,
    border: int = BORDER,
    percentile: float = PERCENTILE,
    target_size: tuple = TARGET_SIZE,
) -> np.ndarray:
    """Full CPU Frangi pipeline matching precompute_frangi_priors.py.
    Returns (14, 14) float16 array in [0, 1]."""
    # 1. Sato filter
    sato = sato_filter(
        image.astype(np.uint8),
        sigmas=sigmas,
        black_ridges=True,
        mode="reflect",
        cval=0,
    )

    # 2. Zero out borders
    h, w = sato.shape
    sato[:border, :] = 0
    sato[-border:, :] = 0
    sato[:, :border] = 0
    sato[:, -border:] = 0

    # 3. Cast to uint8 (matching existing pipeline)
    sato_u8 = sato.astype(np.uint8)

    # 4. Percentile threshold
    if np.any(sato_u8):
        thresh_val = np.percentile(sato_u8, percentile)
        thresh = np.where(sato_u8 >= thresh_val, sato_u8, 0)
    else:
        thresh = np.zeros_like(sato_u8)

    # 5. Region growing
    mask = region_grow_mask(thresh)

    # 6. Downsample to 14×14
    frangi_14 = cv2.resize(
        mask.astype(np.float32), target_size, interpolation=cv2.INTER_AREA
    )
    frangi_14 /= 255.0

    return frangi_14.astype(np.float16)


# ═══════════════════════════════════════════════════════════════════════════════
# GPU BATCH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_frangi_prior_gpu_batch(
    image_paths: list,
    batch_size: int = 32,
    multi_seed: bool = True,
) -> list:
    """
    Full GPU pipeline for a batch of images → 14×14 float16 priors.

    Steps:
      1. Load batch of images → (B, 1, 512, 512) float32 on GPU
      2. GPU Sato filter → vesselness (B, 1, 512, 512)
      3. Zero border, percentile threshold on GPU → (B, 1, 512, 512)
      4. Move to CPU, region grow each image → (B, 512, 512) binary
      5. GPU/F.avg_pool2d → (B, 1, 14, 14)
      6. Convert to [0, 1] float16 numpy arrays

    Args:
        image_paths: List of image file paths.
        batch_size: How many images to load/process per GPU batch.
        multi_seed: If True, use multi-seed region growing; else single-seed.

    Returns list of (14, 14) float16 numpy arrays.
    """
    results = []
    total = len(image_paths)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_paths = image_paths[start:end]

        # --- Step 1: Load batch onto GPU ---
        batch_images = []
        for p in batch_paths:
            img = np.array(Image.open(p).convert("L"), dtype=np.float32)
            batch_images.append(img)
        batch_tensor = torch.from_numpy(np.stack(batch_images)).unsqueeze(1).to(DEVICE)
        B, C, H_img, W_img = batch_tensor.shape

        # --- Step 2: GPU Sato vesselness ---
        vesselness = gpu_sato_filter(batch_tensor, sigmas=SIGMAS, black_ridges=True)
        # vesselness shape: (B, 1, H, W)

        # --- Step 3: Zero border + percentile threshold ---
        vesselness[:, :, :BORDER, :] = 0.0
        vesselness[:, :, -BORDER:, :] = 0.0
        vesselness[:, :, :, :BORDER] = 0.0
        vesselness[:, :, :, -BORDER:] = 0.0

        # Cast to uint8 (matching existing CPU pipeline quantization)
        vesselness_u8 = vesselness.clamp(
            0, 255
        ).to(torch.uint8)  # (B, 1, H, W)

        for b in range(B):
            # Per-image percentile threshold (on uint8 values)
            single = vesselness_u8[b, 0].float()  # (H, W)
            flat = single.flatten()
            nonzero_mask = flat > 0

            if nonzero_mask.any():
                thresh_val = torch.quantile(
                    flat[nonzero_mask].float(), PERCENTILE / 100.0
                )
                thresh_single = torch.where(single >= thresh_val, single, torch.tensor(0.0, device=DEVICE))
            else:
                thresh_single = torch.zeros_like(single)

            # --- Step 4: Move to CPU, region grow ---
            thresh_cpu = thresh_single.cpu().numpy().astype(np.uint8)
            if multi_seed:
                mask_cpu = region_grow_multi_seed(thresh_cpu, n_seeds=8, min_seed_dist=50)
            else:
                mask_cpu = region_grow_mask(thresh_cpu)  # (H, W) uint8 0/255

            # --- Step 5: Back to GPU for downsampling ---
            mask_gpu = (
                torch.from_numpy(mask_cpu.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )  # (1, 1, H, W)

            # avg_pool2d → 14×14
            # kernel_size approx H/14, W/14
            kh = H_img // 14
            kw = W_img // 14
            # Use avg_pool2d with kernel_size matching the block size
            pooled = F.avg_pool2d(mask_gpu, kernel_size=(kh, kw))
            # pooled: (1, 1, ≈14, ≈14)

            # Ensure exactly 14×14 via interpolate (robust to rounding)
            if pooled.shape[-2:] != (14, 14):
                pooled = F.interpolate(
                    pooled, size=(14, 14), mode="area"
                )

            # --- Step 6: Convert to [0,1] float16 numpy ---
            prior = pooled.squeeze().cpu().numpy()  # (14, 14), float32
            prior = prior / 255.0
            prior = prior.astype(np.float16)
            results.append(prior)

        # Cleanup
        del batch_tensor, vesselness, vesselness_u8
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST (run first!)
# ═══════════════════════════════════════════════════════════════════════════════

def run_smoke_test(num_images: int = 5):
    """Compare GPU Frangi (multi-seed vs single-seed) vs RCP on first N images of arcade."""
    print("=" * 70)
    print("SMOKE TEST: GPU Frangi (Multi-Seed vs Single-Seed) vs RCP")
    print("=" * 70)
    print(f"  Device         : {DEVICE}")
    print(f"  Images          : {num_images} from arcade dataset")
    print(f"  Sigmas          : {SIGMAS}")
    print(f"  Border          : {BORDER} px")
    print(f"  Percentile      : {PERCENTILE} %")
    print(f"  Target size     : {TARGET_SIZE}")
    print(f"  Multi-seed      : n_seeds=8, min_seed_dist=50")
    print()

    # Prepare output directory
    os.makedirs(SMOKE_DIR, exist_ok=True)

    # Get image paths
    arcade_img_dir = os.path.join(IMG_DIR, "arcade")
    arcade_rcp_dir = os.path.join(RCP_DIR, "arcade")
    indices = [1, 2, 3, 4, 5]

    image_paths = []
    for idx in indices:
        p = os.path.join(arcade_img_dir, f"{idx}.png")
        if not os.path.exists(p):
            print(f"WARNING: missing image {p}")
            continue
        image_paths.append(p)

    if len(image_paths) == 0:
        print("ERROR: No images found. Smoke test aborted.")
        return

    print(f"  Found {len(image_paths)} images")
    print()

    # ── 1. GPU Frangi (multi-seed) ─────────────────────────────────────────
    print("--- GPU Frangi MULTI-SEED computation ---")
    t_ms_start = time.perf_counter()
    ms_priors = compute_frangi_prior_gpu_batch(image_paths, batch_size=len(image_paths), multi_seed=True)
    t_ms = time.perf_counter() - t_ms_start
    print(f"  Multi-seed time: {t_ms:.3f}s for {len(image_paths)} images ({t_ms / len(image_paths):.3f}s/img)")
    print()

    # ── 2. GPU Frangi (single-seed) ───────────────────────────────────────
    print("--- GPU Frangi SINGLE-SEED computation ---")
    t_ss_start = time.perf_counter()
    ss_priors = compute_frangi_prior_gpu_batch(image_paths, batch_size=len(image_paths), multi_seed=False)
    t_ss = time.perf_counter() - t_ss_start
    print(f"  Single-seed time: {t_ss:.3f}s for {len(image_paths)} images ({t_ss / len(image_paths):.3f}s/img)")
    print()

    # ── 3. Load RCP priors ────────────────────────────────────────────────
    print("--- RCP priors ---")
    rcp_priors = []
    for idx in indices[:len(image_paths)]:
        rcp_path = os.path.join(arcade_rcp_dir, f"{idx}.npy")
        if os.path.exists(rcp_path):
            rcp = np.load(rcp_path).astype(np.float32)
            rcp_priors.append(rcp)
            print(f"  Loaded RCP prior {idx}: shape={rcp.shape}, mean={rcp.mean():.4f}, max={rcp.max():.4f}")
        else:
            print(f"  WARNING: RCP prior {idx} not found")
            rcp_priors.append(np.zeros((14, 14), dtype=np.float32))
    print()

    # ── 4. Comparison ─────────────────────────────────────────────────────
    n = len(ms_priors)
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    ms_ss_corrs = []
    ms_rcp_corrs = []
    ss_rcp_corrs = []

    for i in range(n):
        ms_p = ms_priors[i].astype(np.float32)
        ss_p = ss_priors[i].astype(np.float32)
        rcp_p = rcp_priors[i].astype(np.float32)

        assert ms_p.shape == (14, 14), f"Multi-seed prior shape {ms_p.shape}"
        assert ss_p.shape == (14, 14), f"Single-seed prior shape {ss_p.shape}"

        ms_flat = ms_p.flatten()
        ss_flat = ss_p.flatten()
        rcp_flat = rcp_p.flatten()

        # Correlations
        if np.std(ms_flat) > 0 and np.std(ss_flat) > 0:
            ms_ss_corr, _ = pearsonr(ms_flat, ss_flat)
        else:
            ms_ss_corr = 0.0
        if np.std(ms_flat) > 0 and np.std(rcp_flat) > 0:
            ms_rcp_corr, _ = pearsonr(ms_flat, rcp_flat)
        else:
            ms_rcp_corr = 0.0
        if np.std(ss_flat) > 0 and np.std(rcp_flat) > 0:
            ss_rcp_corr, _ = pearsonr(ss_flat, rcp_flat)
        else:
            ss_rcp_corr = 0.0

        ms_ss_corrs.append(ms_ss_corr)
        ms_rcp_corrs.append(ms_rcp_corr)
        ss_rcp_corrs.append(ss_rcp_corr)

        # Print per-image stats
        idx_label = indices[i]
        ms_nz = int((ms_p > 0).sum())
        ss_nz = int((ss_p > 0).sum())
        rcp_nz = int((rcp_p > 0).sum())
        print(f"\n  Image {idx_label}:")
        print(
            f"    Multi-seed  — mean={ms_p.mean():.5f}  max={ms_p.max():.5f}  "
            f"nonzero={ms_nz}"
        )
        print(
            f"    Single-seed — mean={ss_p.mean():.5f}  max={ss_p.max():.5f}  "
            f"nonzero={ss_nz}"
        )
        print(
            f"    RCP prior   — mean={rcp_p.mean():.5f}  max={rcp_p.max():.5f}  "
            f"nonzero={rcp_nz}"
        )
        print(
            f"    Corr(MS,SS)={ms_ss_corr:.5f}  "
            f"Corr(MS,RCP)={ms_rcp_corr:.5f}  "
            f"Corr(SS,RCP)={ss_rcp_corr:.5f}"
        )
        print(
            f"    Delta nonzero: MS vs SS = {ms_nz - ss_nz:+d}  "
            f"MS vs RCP = {ms_nz - rcp_nz:+d}"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    print(
        f"  Multi-seed vs Single-seed correlation:  mean={np.mean(ms_ss_corrs):.5f}  "
        f"std={np.std(ms_ss_corrs):.5f}"
    )
    print(
        f"  Multi-seed vs RCP correlation:          mean={np.mean(ms_rcp_corrs):.5f}  "
        f"std={np.std(ms_rcp_corrs):.5f}"
    )
    print(
        f"  Single-seed vs RCP correlation:         mean={np.mean(ss_rcp_corrs):.5f}  "
        f"std={np.std(ss_rcp_corrs):.5f}"
    )

    # Nonzero counts summary
    ms_nz_total = sum(int((p > 0).sum()) for p in ms_priors)
    ss_nz_total = sum(int((p > 0).sum()) for p in ss_priors)
    rcp_nz_total = sum(int((p > 0).sum()) for p in rcp_priors)
    print()
    print(f"  Total nonzero patches (14x14):")
    print(f"    Multi-seed:  {ms_nz_total}  ({ms_nz_total/n:.1f}/image)")
    print(f"    Single-seed: {ss_nz_total}  ({ss_nz_total/n:.1f}/image)")
    print(f"    RCP:         {rcp_nz_total}  ({rcp_nz_total/n:.1f}/image)")
    print()
    print(f"  Multi-seed time: {t_ms:.3f}s for {n} images ({t_ms/n:.3f}s/img)")
    print(f"  Single-seed time: {t_ss:.3f}s for {n} images ({t_ss/n:.3f}s/img)")

    # Verdict
    mean_ms_rcp = np.mean(ms_rcp_corrs)
    ms_improvement = ms_nz_total > ss_nz_total
    print()
    if ms_improvement and mean_ms_rcp >= 0.3:
        print(
            f"  [PASS] SMOKE TEST PASSED "
            f"(more nonzeros than single-seed, MS-RCP correlation {mean_ms_rcp:.4f})"
        )
    else:
        if not ms_improvement:
            print(
                f"  [WARN] Multi-seed produced FEWER nonzeros than single-seed. "
                f"Check parameters."
            )
        if mean_ms_rcp < 0.3:
            print(
                f"  [WARN] Multi-seed vs RCP correlation {mean_ms_rcp:.4f} < 0.3. "
                f"May indicate issues."
            )

    # Save smoke test outputs
    print(f"\n  Smoke test outputs saved to: {SMOKE_DIR}")
    for i in range(n):
        idx_label = indices[i]
        np.save(
            os.path.join(SMOKE_DIR, f"multiseed_{idx_label}.npy"), ms_priors[i]
        )
        np.save(
            os.path.join(SMOKE_DIR, f"singleseed_{idx_label}.npy"), ss_priors[i]
        )

    # Save summary as text
    with open(os.path.join(SMOKE_DIR, "summary.txt"), "w") as f:
        f.write("SMOKE TEST RESULTS (MULTI-SEED vs SINGLE-SEED vs RCP)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Multi-seed vs Single-seed correlation: {np.mean(ms_ss_corrs):.5f}\n")
        f.write(f"Multi-seed vs RCP correlation: {np.mean(ms_rcp_corrs):.5f}\n")
        f.write(f"Single-seed vs RCP correlation: {np.mean(ss_rcp_corrs):.5f}\n\n")
        f.write(f"Total nonzero patches (14x14):\n")
        f.write(f"  Multi-seed:  {ms_nz_total}\n")
        f.write(f"  Single-seed: {ss_nz_total}\n")
        f.write(f"  RCP:         {rcp_nz_total}\n\n")
        f.write(f"Multi-seed time: {t_ms:.3f}s for {n} images\n")
        f.write(f"Single-seed time: {t_ss:.3f}s for {n} images\n\n")
        for i in range(n):
            f.write(f"Image {indices[i]}:\n")
            f.write(f"  MS-SS corr: {ms_ss_corrs[i]:.5f}\n")
            f.write(f"  MS-RCP corr: {ms_rcp_corrs[i]:.5f}\n")
            f.write(f"  SS-RCP corr: {ss_rcp_corrs[i]:.5f}\n")

    return mean_ms_rcp, t_ms, t_ss, n


# ═══════════════════════════════════════════════════════════════════════════════
# FULL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def process_dataset_gpu(dataset: str, batch_size: int = 32):
    """Process one dataset end-to-end with GPU batching.

    Steps:
      1. Scan dataset directory for all .png files
      2. Skip files that already exist in priors_frangi/{dataset}/
      3. Process in batches on GPU
      4. Handle missing images gracefully (skip + log)
    """
    img_subdir = os.path.join(IMG_DIR, dataset)
    out_subdir = os.path.join(OUT_DIR, dataset)

    if not os.path.isdir(img_subdir):
        print(f"  SKIP: image directory not found: {img_subdir}")
        return 0.0

    os.makedirs(out_subdir, exist_ok=True)

    # Gather all indices from image files
    all_files = sorted(
        [f for f in os.listdir(img_subdir) if f.endswith(".png")],
        key=lambda f: int(os.path.splitext(f)[0]),
    )
    all_indices = [int(os.path.splitext(f)[0]) for f in all_files]

    # Determine already-done indices
    existing = set()
    if os.path.isdir(out_subdir):
        for f in os.listdir(out_subdir):
            if f.endswith(".npy"):
                try:
                    existing.add(int(os.path.splitext(f)[0]))
                except ValueError:
                    pass

    skipped = len(existing)
    remaining_indices = [idx for idx in all_indices if idx not in existing]
    remaining = len(remaining_indices)

    print(f"\n{'='*65}")
    print(f"Dataset: {dataset}")
    print(f"  Total images: {len(all_indices)}  Already done: {skipped}  Remaining: {remaining}")
    print(f"{'='*65}")

    if remaining == 0:
        print("  -> All files already computed, skipping.")
        return 0.0

    t_start = time.perf_counter()
    ok_count = 0
    skip_count = skipped
    error_count = 0
    missing_count = 0
    error_log = []

    # Process in batches
    pbar = tqdm(range(0, remaining, batch_size), desc=f"  {dataset}", unit="batch")
    for start in pbar:
        end = min(start + batch_size, remaining)
        batch_indices = remaining_indices[start:end]

        # Build image paths; skip missing images
        batch_paths = []
        batch_valid_indices = []
        for idx in batch_indices:
            img_path = os.path.join(img_subdir, f"{idx}.png")
            if os.path.exists(img_path):
                batch_paths.append(img_path)
                batch_valid_indices.append(idx)
            else:
                missing_count += 1
                error_log.append(f"[MISSING] {dataset}/{idx}.png")

        if not batch_paths:
            continue

        try:
            priors = compute_frangi_prior_gpu_batch(
                batch_paths, batch_size=len(batch_paths), multi_seed=True
            )
            for idx, prior in zip(batch_valid_indices, priors):
                out_path = os.path.join(out_subdir, f"{idx}.npy")
                np.save(out_path, prior.astype(np.float16))
                ok_count += 1
        except Exception:
            error_count += len(batch_valid_indices)
            error_log.append(
                f"[ERROR] {dataset}/batch indices {batch_valid_indices[0]}-{batch_valid_indices[-1]}: {traceback.format_exc().strip()}"
            )

        # Update progress bar
        pbar.set_postfix(ok=ok_count, err=error_count, miss=missing_count)

    elapsed = time.perf_counter() - t_start

    # Print summary for this dataset
    print(f"\n  --- {dataset} summary ---")
    print(f"  Processed: {ok_count}  Skipped (existing): {skipped}  Errors: {error_count}  Missing: {missing_count}")
    if ok_count > 0:
        print(f"  Time: {elapsed:.1f}s  ({ok_count/elapsed:.1f} img/s)")
    else:
        print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {out_subdir}")

    # Save error log if any
    if error_log:
        log_path = os.path.join(OUT_DIR, f"{dataset}_errors.log")
        with open(log_path, "w") as f:
            f.write("\n".join(error_log))
        print(f"  Error log: {log_path}")

    return elapsed


def run_full_computation():
    """Run full GPU Frangi precomputation on all datasets, smallest first."""
    print("=" * 65)
    print("GPU FRANGI PRIOR PRECOMPUTATION (FULL) — Multi-Seed")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")
    print(f"Multi-seed: n_seeds=8, min_seed_dist=50")
    print()

    total_start = time.perf_counter()
    dataset_times = {}

    for dataset, n_expected in DATASETS:
        elapsed = process_dataset_gpu(dataset, batch_size=32)
        dataset_times[dataset] = elapsed

    total_elapsed = time.perf_counter() - total_start

    # Final summary
    print()
    print("=" * 65)
    print("ALL DATASETS COMPLETE")
    print("=" * 65)
    for dataset, n_expected in DATASETS:
        et = dataset_times.get(dataset, 0)
        print(f"  {dataset:>20s}: {et:.1f}s")
    print(f"  {'TOTAL':>20s}: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU Frangi vesselness prior precomputation"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full precomputation on all datasets (only after smoke test passes)",
    )
    parser.add_argument(
        "--smoke-n",
        type=int,
        default=5,
        help="Number of images for smoke test (default: 5)",
    )
    args = parser.parse_args()

    if args.full:
        print("Running FULL COMPUTATION on all datasets.\n")
        run_full_computation()
    else:
        print("Running SMOKE TEST (multi-seed vs single-seed vs RCP).\n")
        run_smoke_test(num_images=args.smoke_n)


if __name__ == "__main__":
    main()
