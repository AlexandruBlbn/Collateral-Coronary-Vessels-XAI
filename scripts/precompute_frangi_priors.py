#!/usr/bin/env python
"""
precompute_frangi_priors.py

Precompute Frangi vesselness priors at 14x14 for every image that has
an RCP prior in data/pretrain/priors/. Processes datasets from smallest
to largest with multiprocessing and resume support.

Source: RCP priors in data/pretrain/priors/{dataset}/{idx}.npy
Images: data/pretrain/dataset/{dataset}/{idx}.png
Output: data/pretrain/priors_frangi/{dataset}/{idx}.npy  (float16, 14x14, [0,1])
"""

import sys
import os
import time
import traceback
import warnings
from functools import partial
from multiprocessing import Pool

warnings.filterwarnings("ignore")

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import *  # np, cv2, Image, tqdm, Path, torch, etc.
from skimage.filters import sato as sato_filter

# ── Globals / configuration ─────────────────────────────────────────────────
RCP_DIR = os.path.join(PROJECT_ROOT, "data", "pretrain", "priors")
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "pretrain", "dataset")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "pretrain", "priors_frangi")

BORDER = 20
PERCENTILE = 92.0
SIGMAS = [1, 2, 3, 4]
TARGET_SIZE = (14, 14)

ERROR_LOG = os.path.join(PROJECT_ROOT, "scripts", "frangi_precompute_errors.log")

# Process smallest → largest
DATASETS = [
    ("xcad",              1621),
    ("arcade",            2000),
    ("syntax",            2943),
    ("cadica",            6594),
    ("coronarydominance", 160320),
]


# ── Core Frangi pipeline (single image → 14×14 float16 array) ───────────────
def compute_frangi_prior_14x14(image_path: str) -> np.ndarray:
    """
    Exact VasoMIM implementation (matching scripts/run_frangi_test.py):
      1. Load grayscale via PIL ('L')
      2. Sato filter (sigmas=[1,2,3,4], black_ridges=True, mode='reflect')
      3. Zero out 20-pixel border
      4. Convert to uint8
      5. Percentile threshold at 92 %
      6. 4-connected region growing from max-intensity seed (binary 0/255)
      7. cv2.INTER_AREA downsample to 14×14, divide by 255 → [0, 1]
    Returns (14, 14) float16 array.
    """
    # 1. Load as grayscale
    image = np.array(Image.open(image_path).convert("L"))

    # 2. Sato vesselness filter
    sato = sato_filter(
        image.astype(np.uint8),
        sigmas=SIGMAS,
        black_ridges=True,
        mode="reflect",
        cval=0,
    )

    # 3. Zero out borders (border=20)
    h, w = sato.shape
    sato[:BORDER, :] = 0
    sato[-BORDER:, :] = 0
    sato[:, :BORDER] = 0
    sato[:, -BORDER:] = 0

    # 4. Convert to uint8
    sato_u8 = sato.astype(np.uint8)

    # 5. Percentile thresholding at 92 %
    thresh_val = np.percentile(sato_u8, PERCENTILE)
    thresh = np.where(sato_u8 >= thresh_val, sato_u8, 0)

    # 6. Region growing from maximum-intensity pixel (4-connected)
    if np.any(thresh):
        seed_idx = np.unravel_index(np.argmax(thresh), thresh.shape)
        seed = (int(seed_idx[0]), int(seed_idx[1]))
    else:
        seed = None

    visited = np.zeros_like(thresh, dtype=bool)
    mask = np.zeros_like(thresh, dtype=np.uint8)

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
            mask[x, y] = 255
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < h
                    and 0 <= ny < w
                    and not visited[nx, ny]
                    and thresh[nx, ny] > 0
                ):
                    stack.append((nx, ny))

    # 7. Downsample to 14×14 with INTER_AREA → continuous [0, 1]
    frangi_14 = cv2.resize(
        mask.astype(np.float32), TARGET_SIZE, interpolation=cv2.INTER_AREA
    )
    frangi_14 /= 255.0  # scale to [0, 1]

    return frangi_14.astype(np.float16)


# ── Single-file worker (module-level so multiprocessing can pickle it) ──────
def _process_one(args):
    """Worker function: compute & save Frangi prior for one index."""
    idx, dataset = args
    out_path = os.path.join(OUT_DIR, dataset, f"{idx}.npy")

    # Skip if already exists (resume)
    if os.path.exists(out_path):
        return ("skip", idx, "")

    img_path = os.path.join(IMG_DIR, dataset, f"{idx}.png")

    if not os.path.exists(img_path):
        return ("error", idx, f"missing image: {img_path}")

    try:
        frangi_14 = compute_frangi_prior_14x14(img_path)
        np.save(out_path, frangi_14)
        return ("ok", idx, "")
    except Exception:
        return ("error", idx, traceback.format_exc().strip())


# ── Dataset-level runner ─────────────────────────────────────────────────────
def process_dataset(dataset: str, n_expected: int) -> float:
    """
    Process all images for one dataset using a multiprocessing Pool.
    Returns elapsed wall-clock seconds.
    """
    rcp_subdir = os.path.join(RCP_DIR, dataset)
    out_subdir = os.path.join(OUT_DIR, dataset)

    # Collect indices from RCP prior files
    rcp_files = [f for f in os.listdir(rcp_subdir) if f.endswith(".npy")]
    indices = sorted(
        int(os.path.splitext(f)[0]) for f in rcp_files
    )

    # Ensure output directory exists
    os.makedirs(out_subdir, exist_ok=True)

    # Count how many already exist (resume)
    already_done = sum(
        1 for idx in indices
        if os.path.exists(os.path.join(out_subdir, f"{idx}.npy"))
    )
    remaining = len(indices) - already_done

    print(f"\n{'=' * 65}")
    print(f"Dataset: {dataset}")
    print(f"  RCP priors found : {len(indices)}")
    print(f"  Already computed  : {already_done}")
    print(f"  Remaining         : {remaining}")
    print(f"  Image dir         : {os.path.join(IMG_DIR, dataset)}")
    print(f"  Output dir        : {out_subdir}")
    print(f"{'=' * 65}")

    if remaining == 0:
        print("  → All files already computed, skipping.\n")
        return 0.0

    # Build task list (only indices not yet done)
    tasks = [
        (idx, dataset) for idx in indices
        if not os.path.exists(os.path.join(out_subdir, f"{idx}.npy"))
    ]

    num_workers = os.cpu_count()
    print(f"  Workers: {num_workers}")

    t_start = time.perf_counter()

    ok_count = 0
    skip_count = 0
    error_count = 0
    errors = []

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc=f"  {dataset}", unit="img") as pbar:
            for status, idx, msg in pool.imap_unordered(_process_one, tasks):
                if status == "ok":
                    ok_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    error_count += 1
                    errors.append((idx, msg))
                pbar.update(1)

    elapsed = time.perf_counter() - t_start
    rate = ok_count / elapsed if elapsed > 0 else 0

    print(f"\n  ── {dataset} results ──")
    print(f"    OK     : {ok_count}")
    print(f"    Skipped: {skip_count}")
    print(f"    Errors : {error_count}")
    print(f"    Time   : {elapsed:.1f}s  ({ok_count / elapsed:.1f} img/s)" if ok_count > 0 else f"    Time   : {elapsed:.1f}s")

    if errors:
        print(f"    (errors logged to {ERROR_LOG})")
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            for idx, msg in errors:
                f.write(f"[{dataset}/{idx}] {msg}\n")
            f.flush()

    # Estimate for full dataset at this rate
    if ok_count > 0 and rate > 0:
        est_seconds = n_expected / rate
        est_minutes = est_seconds / 60
        print(f"    Est for full {n_expected}: {est_seconds:.0f}s = {est_minutes:.1f} min")

    return elapsed


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("FRANGI PRIOR PRECOMPUTATION")
    print("=" * 65)
    print(f"Source priors  : {RCP_DIR}")
    print(f"Source images  : {IMG_DIR}")
    print(f"Output dir     : {OUT_DIR}")
    print(f"Datasets       : {[d for d, _ in DATASETS]}")
    print(f"Sato sigmas    : {SIGMAS}")
    print(f"Border crop    : {BORDER} px")
    print(f"Percentile     : {PERCENTILE} %")
    print(f"Target size    : {TARGET_SIZE}")
    print(f"CPU cores      : {os.cpu_count()}")
    print()

    total_start = time.perf_counter()
    dataset_times = {}

    for dataset, n_expected in DATASETS:
        t_ds = process_dataset(dataset, n_expected)
        dataset_times[dataset] = t_ds

        # Estimate time for remaining datasets
        if len(dataset_times) < len(DATASETS):
            remaining = DATASETS[len(dataset_times):]
            # Use the rate from the first completed dataset
            first_ds = list(dataset_times.keys())[0]
            first_t = dataset_times[first_ds]
            first_n = dict(DATASETS)[first_ds]
            if first_t > 0:
                rate = first_n / first_t
                total_remaining = sum(n for _, n in remaining)
                est = total_remaining / rate
                print(f"\n  >>> Estimated time for remaining {len(remaining)} datasets: {est:.0f}s = {est / 60:.1f} min")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 65}")
    print(f"ALL DONE in {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 65}")

    # ── Sample stats: compare 5 random Frangi priors vs RCP priors ──────────
    print("\n" + "=" * 65)
    print("SAMPLE COMPARISON: Frangi vs RCP priors (14×14)")
    print("=" * 65)

    import random as _random
    _random.seed(42)

    for dataset, _ in DATASETS:
        rcp_subdir = os.path.join(RCP_DIR, dataset)
        frangi_subdir = os.path.join(OUT_DIR, dataset)
        rcp_files = [f for f in os.listdir(rcp_subdir) if f.endswith(".npy")]
        if len(rcp_files) < 5:
            continue

        samples = _random.sample(rcp_files, min(5, len(rcp_files)))
        rcp_stats = []
        frangi_stats = []

        for f in samples:
            try:
                rcp = np.load(os.path.join(rcp_subdir, f)).astype(np.float32)
                frangi = np.load(os.path.join(frangi_subdir, f)).astype(np.float32)
                rcp_stats.append((rcp.mean(), rcp.max(), rcp.std()))
                frangi_stats.append((frangi.mean(), frangi.max(), frangi.std()))
            except Exception:
                pass

        if rcp_stats:
            rcp_m, rcp_x, rcp_s = np.mean(rcp_stats, axis=0)
            f_m, f_x, f_s = np.mean(frangi_stats, axis=0)
            print(f"\n  {dataset}:")
            print(f"    RCP    — mean={rcp_m:.4f}  max={rcp_x:.4f}  std={rcp_s:.4f}")
            print(f"    Frangi — mean={f_m:.4f}  max={f_x:.4f}  std={f_s:.4f}")


if __name__ == "__main__":
    main()
