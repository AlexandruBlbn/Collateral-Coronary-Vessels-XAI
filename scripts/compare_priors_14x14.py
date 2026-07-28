"""
compare_priors_14x14.py
Compare three vessel priors at 14x14 resolution on ARCADE syntax data:
  (1) Frangi (Sato filter)  -- computed on-the-fly
  (2) RCP (Ridge Count Prior) -- precomputed .npy files at 14x14
  (3) GT syntax (expert) masks -- downsampled to 14x14

Metrics at 14x14: Dice (binary & continuous), MAE, Pearson correlation,
positive patch count.
"""

import sys
import os

# Add project root to path so we can use utils.helpers and data.data
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import *
from skimage.filters import sato
from skimage.measure import label, regionprops
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ARCADE", "processed", "syntax", "data")
LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "ARCADE", "processed", "syntax", "label")
RCP_DIR = os.path.join(PROJECT_ROOT, "data", "pretrain", "priors", "syntax")

# ---------------------------------------------------------------------------
# Utility: region growing from seed
# ---------------------------------------------------------------------------
def select_cc_with_max_pixel(response: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """
    From a binary mask, select only the connected component that contains the
    pixel with the maximum response value (within the binary foreground).
    If binary is empty, return zeros.
    """
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    # Find the max pixel in the response (only within binary foreground)
    masked_response = response * binary.astype(np.float32)
    seed = np.unravel_index(np.argmax(masked_response), masked_response.shape)

    # Label connected components
    num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    cc_label = labels[seed]

    result = (labels == cc_label).astype(np.uint8) * 255
    return result


# ---------------------------------------------------------------------------
# Compute Frangi (Sato) mask at 512x512
# ---------------------------------------------------------------------------
def compute_frangi_mask_512(
    image: np.ndarray,
    sigmas=(1, 2, 3, 4),
    border_zero: int = 20,
    percentile_thresh: int = 92,
) -> np.ndarray:
    """
    Compute Frangi vesselness via Sato filter at 512x512, then post-process:
      1. Sato filter (black_ridges=True, mode='reflect')
      2. Zero out border of `border_zero` pixels
      3. Threshold at given percentile (keep top (100 - pct)%)
      4. Region growing from the max pixel of the thresholded response
    Returns binary mask (0/255) at 512x512.
    """
    h, w = image.shape

    # 1. Sato vesselness
    sato_response = sato(image.astype(np.float64), sigmas=sigmas, black_ridges=True, mode='reflect')
    # sato returns positive values for ridges (dark on bright), higher = more vessel-like
    sato_response = np.maximum(sato_response, 0.0)  # clip negatives

    # 2. Zero border
    sato_response[:border_zero, :] = 0
    sato_response[-border_zero:, :] = 0
    sato_response[:, :border_zero] = 0
    sato_response[:, -border_zero:] = 0

    # 3. Percentile threshold
    nonzero = sato_response[sato_response > 0]
    if len(nonzero) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    thresh_val = np.percentile(nonzero, percentile_thresh)
    binary = (sato_response > thresh_val).astype(np.uint8)

    # 4. Select connected component containing the max pixel
    if binary.sum() > 0:
        binary = select_cc_with_max_pixel(sato_response, binary)

    return binary.astype(np.uint8)  # 0 or 255


# ---------------------------------------------------------------------------
# Downsample 512x512 to 14x14
# ---------------------------------------------------------------------------
def downsample_to_14x14(mask_512: np.ndarray) -> np.ndarray:
    """
    Downsample a 512x512 binary mask (values 0/255 or 0/1) to 14x14.
    Uses cv2.INTER_AREA to compute the average fraction of positive pixels
    in each patch. Returns float32 array shape (14, 14) with values in [0, 1].

    For binary masks, this gives the fraction of vessel pixels per patch.
    """
    # Ensure float [0, 1] before resize
    if mask_512.max() > 1.0:
        mask_f = mask_512.astype(np.float32) / 255.0
    else:
        mask_f = mask_512.astype(np.float32)

    resized = cv2.resize(mask_f, (14, 14), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)  # [0, 1] continuous


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def dice_binary(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> tuple:
    """Dice coefficient for binary masks. Returns (dice, both_empty_flag)."""
    a_bin = (a >= threshold).astype(np.uint8)
    b_bin = (b >= threshold).astype(np.uint8)
    inter = (a_bin & b_bin).sum()
    total = a_bin.sum() + b_bin.sum()
    if total == 0:
        return 1.0, True  # both empty -> perfect agreement (but flag it)
    return 2.0 * inter / total, False


def dice_continuous(a: np.ndarray, b: np.ndarray) -> float:
    """Dice-like overlap for continuous values in [0,1]:
       2 * sum(min(a,b)) / sum(a + b)."""
    num = 2.0 * np.sum(np.minimum(a, b))
    den = np.sum(a) + np.sum(b)
    if den == 0:
        return 1.0
    return float(num / den)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(a - b)))


def positive_patch_count(arr: np.ndarray, threshold: float = 0.5) -> int:
    """Number of 14x14 patches above threshold."""
    return int((arr >= threshold).sum())


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def main():
    set_seed(42)

    # Number of patients to sample
    N_PATIENTS = 30
    MAX_IDX = 1500

    # Random sample of indices
    all_indices = list(range(1, MAX_IDX + 1))
    random.shuffle(all_indices)
    sample_indices = sorted(all_indices[:N_PATIENTS])

    print("=" * 70)
    print("PRIOR COMPARISON AT 14x14")
    print("  Patients sampled:", N_PATIENTS, "/", MAX_IDX)
    print("  RCP prior source: data/pretrain/priors/syntax/{idx}.npy  [14x14, float16]")
    print("  Frangi prior: Sato filter computed at 512x512, downsample to 14x14")
    print("  GT syntax: label at 512x512, downsample to 14x14")
    print("=" * 70)

    # Accumulators
    # Thresholds for binary Dice: 0.5 is the standard "vessel present" threshold;
    # but at 14x14 vessel density per patch is low, so also report lower thresholds.
    THRESHOLDS = [0.01, 0.05, 0.10, 0.50]
    results = {}
    for t in THRESHOLDS:
        results[f"frangi_bin_dice_{t}"] = []
        results[f"rcp_bin_dice_{t}"] = []
        results[f"frangi_both_empty_{t}"] = 0  # count
        results[f"rcp_both_empty_{t}"] = 0

    results.update({
        "frangi_cont_dice": [],
        "rcp_cont_dice": [],
        "frangi_mae": [],
        "rcp_mae": [],
        "frangi_corr": [],
        "rcp_corr": [],
        "frangi_ppos": [],
        "rcp_ppos": [],
        "gt_ppos": [],
        "frangi_max": [],
        "rcp_max": [],
        "gt_max": [],
    })

    errors = []

    for idx in tqdm(sample_indices, desc="Processing patients"):
        try:
            # --- 1. Load image (512x512) ---
            img_path = os.path.join(DATA_DIR, f"{idx}.png")
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                errors.append(f"Idx {idx}: cannot load image {img_path}")
                continue

            # --- 2. Load GT syntax mask (512x512, 0/255) ---
            gt_path = os.path.join(LABEL_DIR, f"{idx}.png")
            gt_512 = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_512 is None:
                errors.append(f"Idx {idx}: cannot load GT label {gt_path}")
                continue

            # --- 3. Compute Frangi mask at 512x512 ---
            frangi_512 = compute_frangi_mask_512(image)

            # --- 4. Load RCP prior (already 14x14) ---
            rcp_path = os.path.join(RCP_DIR, f"{idx}.npy")
            rcp_14 = np.load(rcp_path).astype(np.float32)  # (14, 14), [0, 1]

            # --- 5. Downsample Frangi and GT to 14x14 ---
            frangi_14 = downsample_to_14x14(frangi_512)  # [0, 1] continuous
            gt_14 = downsample_to_14x14(gt_512)           # [0, 1] continuous

            # --- 6. Metrics ---
            # Binary Dice at multiple thresholds
            for t in THRESHOLDS:
                fd, fempty = dice_binary(frangi_14, gt_14, threshold=t)
                rd, rempty = dice_binary(rcp_14, gt_14, threshold=t)
                results[f"frangi_bin_dice_{t}"].append(fd)
                results[f"rcp_bin_dice_{t}"].append(rd)
                results[f"frangi_both_empty_{t}"] += int(fempty)
                results[f"rcp_both_empty_{t}"] += int(rempty)

            # Continuous Dice
            f_cont_dice = dice_continuous(frangi_14, gt_14)
            r_cont_dice = dice_continuous(rcp_14, gt_14)

            # MAE
            f_mae = mae(frangi_14, gt_14)
            r_mae = mae(rcp_14, gt_14)

            # Pearson correlation (flatten to 1D, use all valid patches)
            f_corr, _ = pearsonr(frangi_14.flatten(), gt_14.flatten())
            r_corr, _ = pearsonr(rcp_14.flatten(), gt_14.flatten())

            # Positive patch count (at 0.5)
            f_ppos = positive_patch_count(frangi_14)
            r_ppos = positive_patch_count(rcp_14)
            g_ppos = positive_patch_count(gt_14)

            # Max value in the 14x14 grid
            f_max = float(frangi_14.max())
            r_max = float(rcp_14.max())
            g_max = float(gt_14.max())

            # Store
            results["frangi_cont_dice"].append(f_cont_dice)
            results["rcp_cont_dice"].append(r_cont_dice)
            results["frangi_mae"].append(f_mae)
            results["rcp_mae"].append(r_mae)
            results["frangi_corr"].append(f_corr)
            results["rcp_corr"].append(r_corr)
            results["frangi_ppos"].append(f_ppos)
            results["rcp_ppos"].append(r_ppos)
            results["gt_ppos"].append(g_ppos)
            results["frangi_max"].append(f_max)
            results["rcp_max"].append(r_max)
            results["gt_max"].append(g_max)

        except Exception as e:
            errors.append(f"Idx {idx}: {e}")

    # --- Summary ---
    n = len(results["frangi_cont_dice"])
    print(f"\nProcessed {n} patients successfully ({len(errors)} errors)")

    if errors:
        print("\nERRORS:")
        for err in errors:
            print(f"  - {err}")

    if n == 0:
        print("\nNo results to aggregate.")
        return

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (mean ± std across patients)")
    print("=" * 70)

    # --- Binary Dice at multiple thresholds ---
    print("\n--- Binary Dice (threshold at 0.5) ---")
    print(f"{'Metric':<55} {'Mean':>10} {'Std':>10}")
    print("-" * 77)
    for t in THRESHOLDS:
        for name, prefix in [("Frangi", "frangi"), ("RCP   ", "rcp")]:
            key = f"{prefix}_bin_dice_{t}"
            vals = results[key]
            mu = np.mean(vals)
            sd = np.std(vals)
            empty_key = f"{prefix}_both_empty_{t}"
            n_empty = results[empty_key]
            # Filter out "both empty" cases for a fair Dice
            non_empty = [v for v, e in zip(vals, [None]*len(vals))]  # placeholder
            # We can't easily know per-patient, so just note count
            label = f"Dice Bin (th={t:.2f}) [{name} vs GT]"
            print(f"{label:<55} {mu:>10.4f} {sd:>10.4f}  "
                  f"(both-empty in {n_empty}/{n} pts)")

    # --- Continuous Metrics ---
    print("\n--- Continuous Overlap & Error ---")
    metrics_to_print = [
        ("Dice Continuous (Frangi vs GT)",       "frangi_cont_dice"),
        ("Dice Continuous (RCP    vs GT)",       "rcp_cont_dice"),
        ("MAE           (Frangi vs GT)",         "frangi_mae"),
        ("MAE           (RCP    vs GT)",         "rcp_mae"),
        ("Pearson r     (Frangi vs GT)",         "frangi_corr"),
        ("Pearson r     (RCP    vs GT)",         "rcp_corr"),
    ]

    print(f"{'Metric':<45} {'Mean':>10} {'Std':>10}")
    print("-" * 67)
    for name, key in metrics_to_print:
        vals = results[key]
        mu = np.mean(vals)
        sd = np.std(vals)
        print(f"{name:<45} {mu:>10.4f} {sd:>10.4f}")

    # --- Patch statistics ---
    print("\n--- Positive Patch Count (threshold=0.5, of 196 patches) ---")
    print(f"{'Metric':<45} {'Mean':>10} {'Std':>10}")
    print("-" * 67)
    for name, key in [("Positive patches (Frangi)", "frangi_ppos"),
                       ("Positive patches (RCP)",    "rcp_ppos"),
                       ("Positive patches (GT)",     "gt_ppos")]:
        vals = results[key]
        mu = np.mean(vals)
        sd = np.std(vals)
        print(f"{name:<45} {mu:>10.1f} {sd:>10.1f}")

    # --- Max value per 14x14 grid ---
    print("\n--- Max Value per 14x14 Patch Grid ---")
    print(f"{'Metric':<45} {'Mean':>10} {'Std':>10}")
    print("-" * 67)
    for name, key in [("Max value (Frangi)", "frangi_max"),
                       ("Max value (RCP)",    "rcp_max"),
                       ("Max value (GT)",     "gt_max")]:
        vals = results[key]
        mu = np.mean(vals)
        sd = np.std(vals)
        print(f"{name:<45} {mu:>10.4f} {sd:>10.4f}")

    # --- Per-patient table ---
    print("\n" + "=" * 70)
    print("PER-PATIENT RESULTS")
    header = (f"{'Idx':>5} {'F_Dice.5':>9} {'R_Dice.5':>9} "
              f"{'F_ContD':>8} {'R_ContD':>8} {'F_MAE':>7} {'R_MAE':>7} "
              f"{'F_r':>6} {'R_r':>6} {'F_pos':>5} {'R_pos':>5} {'G_pos':>5} "
              f"{'F_max':>6} {'R_max':>6} {'G_max':>6}")
    print(header)
    print("-" * len(header))

    for i, idx in enumerate(sample_indices[:n]):
        fd05 = results[f"frangi_bin_dice_0.5"][i]
        rd05 = results[f"rcp_bin_dice_0.5"][i]
        print(f"{idx:>5} "
              f"{fd05:>9.4f} "
              f"{rd05:>9.4f} "
              f"{results['frangi_cont_dice'][i]:>8.4f} "
              f"{results['rcp_cont_dice'][i]:>8.4f} "
              f"{results['frangi_mae'][i]:>7.4f} "
              f"{results['rcp_mae'][i]:>7.4f} "
              f"{results['frangi_corr'][i]:>6.3f} "
              f"{results['rcp_corr'][i]:>6.3f} "
              f"{results['frangi_ppos'][i]:>5d} "
              f"{results['rcp_ppos'][i]:>5d} "
              f"{results['gt_ppos'][i]:>5d} "
              f"{results['frangi_max'][i]:>6.4f} "
              f"{results['rcp_max'][i]:>6.4f} "
              f"{results['gt_max'][i]:>6.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
