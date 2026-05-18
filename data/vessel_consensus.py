import argparse
import os
import cv2
import numpy as np
from typing import Dict, Sequence
import skimage.filters
import skimage.morphology

def pad_black_borders(img: np.ndarray) -> np.ndarray:
    """
    Extrapolates the valid tissue area into the black borders (collimator edges).
    This prevents Frangi and Coherence from seeing massive artificial step-gradients.
    """
    # 1. Identify the purely black background areas (usually < 15 intensity)
    _, fg = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    
    # 2. Inpaint (extrapolate) the tissue colors into the black area
    # INPAINT_TELEA is extremely fast and effective for this kind of smooth continuation
    extrapolated = cv2.inpaint(img, 255 - fg, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    return extrapolated

def get_fov_mask(img: np.ndarray) -> np.ndarray:
    """Detect the circular/rectangular XCA field of view to mask out borders."""
    _, fg = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    # Erode aggressively to ignore the massive gradients exactly at the border
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.erode(fg, kernel, iterations=2)
    
    # Force a 15-pixel black border around the absolute edges of the image
    # to kill any filter artifacts that happen at array boundaries.
    mask[0:15, :] = 0
    mask[-15:, :] = 0
    mask[:, 0:15] = 0
    mask[:, -15:] = 0
    
    return mask

def hessian_eigenvalue_ratio(img: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    v = skimage.filters.frangi(img, sigmas=sigmas, black_ridges=True, alpha=0.5, beta=0.5, gamma=20)
    if v.max() > 0:
        v = v / v.max()
    return v.astype(np.float32)

def structure_tensor_coherence(img: np.ndarray, inner_sigma: float = 1.5, outer_sigma: float = 3.0) -> np.ndarray:
    img_f = img.astype(np.float64) / 255.0
    gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)

    r_in = int(np.ceil(inner_sigma * 3))
    k_inner = np.exp(-0.5 * (np.arange(-r_in, r_in + 1) / inner_sigma) ** 2)
    k_inner /= k_inner.sum()
    
    Jxx = cv2.sepFilter2D(gx * gx, -1, k_inner, k_inner)
    Jxy = cv2.sepFilter2D(gx * gy, -1, k_inner, k_inner)
    Jyy = cv2.sepFilter2D(gy * gy, -1, k_inner, k_inner)

    r_out = int(np.ceil(outer_sigma * 3))
    k_outer = np.exp(-0.5 * (np.arange(-r_out, r_out + 1) / outer_sigma) ** 2)
    k_outer /= k_outer.sum()

    Jxx = cv2.sepFilter2D(Jxx, -1, k_outer, k_outer)
    Jxy = cv2.sepFilter2D(Jxy, -1, k_outer, k_outer)
    Jyy = cv2.sepFilter2D(Jyy, -1, k_outer, k_outer)

    tr = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    sqrt_disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))

    mu1 = 0.5 * (tr + sqrt_disc)
    mu2 = 0.5 * (tr - sqrt_disc)

    coherence = (mu1 - mu2) / (mu1 + mu2 + 1e-5) 
    
    magnitude_weight = tr / (tr.max() + 1e-8)
    coherence = coherence * (magnitude_weight > 0.05).astype(np.float32)

    return np.clip(coherence, 0, 1).astype(np.float32)

def dark_valley_map(img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Morphological Black-Hat to find local dark valleys (vessels)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return (bh.astype(np.float32) / 255.0)

def extract_vessel_from_blackhat(blackhat_img: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    """Pure Black-Hat vessel extraction using Seeded Morphological Reconstruction."""
    valid_pixels = blackhat_img[fov_mask > 0]
    if len(valid_pixels) == 0:
        return np.zeros_like(blackhat_img, dtype=np.uint8)

    # 1. Two Thresholds
    thresh_low = np.percentile(valid_pixels, 85)
    thresh_high = np.percentile(valid_pixels, 98)

    mask = (blackhat_img >= thresh_low).astype(np.uint8)
    seed = (blackhat_img >= thresh_high).astype(np.uint8)

    # 2. Morphological Reconstruction (Region Growing)
    reconstructed = skimage.morphology.reconstruction(seed, mask, method='dilation')
    reconstructed = (reconstructed * 255).astype(np.uint8)

    # 3. Filter out background noise blobs (Connected Components)
    clean_vessels = np.zeros_like(reconstructed)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(reconstructed, connectivity=8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] > 150:
            clean_vessels[labels == i] = 255

    # 4. Closing to fill small holes inside the thick vessels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_vessels = cv2.morphologyEx(clean_vessels, cv2.MORPH_CLOSE, kernel)

    return clean_vessels

def compute_consensus_mask(img: np.ndarray, sigmas=( 2,3,4,5,6), signal_pct=85.0) -> Dict[str, np.ndarray]:
    fov_mask = get_fov_mask(img)
    
    # Pad out the black collimator borders with artificial tissue (inpaint)
    # so that the filters don't "trip" on the massive black-to-white edge gradient.
    img_padded = pad_black_borders(img)
    
    # Apply CLAHE to improve contrast of the vessels against the background
    img_u8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_padded.astype(np.uint8))
    
    # Smooth the image to reduce background speckle/noise before the derivative-heavy Frangi filter.
    # 1. Bilateral filter to smooth flat tissue areas while keeping main vessel edges.
    img_smoothed = cv2.bilateralFilter(img_u8, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 2. Gaussian blur to soften the remaining "shaky" high-frequency pixels and pixel-level jitter.
    img_smoothed = cv2.GaussianBlur(img_smoothed, (5, 5), 1.0)
    
    s1 = hessian_eigenvalue_ratio(img_smoothed, sigmas=sigmas)
    s2 = structure_tensor_coherence(img_smoothed, 1.5, 3.0)
    s3 = dark_valley_map(img_smoothed, kernel_size=15)

    s1[fov_mask == 0] = 0
    s2[fov_mask == 0] = 0
    s3[fov_mask == 0] = 0

    soft_consensus = s1 * s2 * s3

    b1 = s1 > np.percentile(s1[fov_mask > 0], signal_pct)
    b2 = s2 > np.percentile(s2[fov_mask > 0], signal_pct)
    b3 = s3 > np.percentile(s3[fov_mask > 0], signal_pct)
    
    thick_consensus = (b1 & b2 & b3).astype(np.uint8) * 255

    # --- SKELETONIZATION & RECONSTRUCTION ---
    # 1. Skeletonize the dark valley (b3) directly for perfect centerline alignment
    dv_skeleton = skimage.morphology.skeletonize(b3)
    
    # 2. Filter using binary consensus mask (b1 & b2 & b3)
    filtered_skeleton = dv_skeleton & (thick_consensus > 0)
    
    # 3. Remove tiny unconnected components (<5px)
    clean_skeleton = np.zeros_like(filtered_skeleton, dtype=np.uint8)
    if filtered_skeleton.any():
        n_skel, labels_skel, stats_skel, _ = cv2.connectedComponentsWithStats(filtered_skeleton.astype(np.uint8), connectivity=8)
        for i in range(1, n_skel):
            if stats_skel[i, cv2.CC_STAT_AREA] >= 5:
                clean_skeleton[labels_skel == i] = 1

    # 4. Reconstruct full vessel thickness by using the cleaned skeleton to seed the thick consensus
    reconstructed_thick = np.zeros_like(thick_consensus)
    components = np.zeros_like(thick_consensus, dtype=np.int32)
    if thick_consensus.any():
        n_thick, labels_thick, stats_thick, _ = cv2.connectedComponentsWithStats(thick_consensus, connectivity=8)
        components = labels_thick.copy()
        for i in range(1, n_thick):
            component_mask = (labels_thick == i)
            # If this thick component intersects with a valid skeleton, keep it
            if (component_mask & (clean_skeleton > 0)).any():
                reconstructed_thick[component_mask] = 255
            else:
                components[labels_thick == i] = 0

    # 5. Dilate the 1-pixel skeleton slightly (3x3) so the neural network
    #    has a stable 3-pixel wide centerline target instead of a vanishingly sparse one.
    skel_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skeleton_thick = cv2.dilate(clean_skeleton, skel_kernel) * 255

    # 6. Extract using standalone Black-Hat method for comparison
    pure_bh_vessels = extract_vessel_from_blackhat(s3, fov_mask)

    return {
        "hessian_ratio": s1, "coherence": s2, "dark_valley": s3,
        "consensus": soft_consensus, "consensus_binary": reconstructed_thick,
        "skeleton_mask": skeleton_thick,
        "pure_bh_vessels": pure_bh_vessels,
        "components": components
    }

def visualise_consensus(result, image, save_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    maps = [image, result["hessian_ratio"], result["coherence"], 
            result["dark_valley"], result["pure_bh_vessels"], result["consensus_binary"], 
            result["skeleton_mask"]]
    titles = ["Original", "Frangi Tubularity", "Orientation Coherence", 
              "Dark Valley (BlackHat)", "Pure BH Recon", "Consensus (Thick)", 
              "Centerline Skeleton"]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[-1, -1].axis("off")  # Hide the 8th empty subplot
    
    for ax, title, data in zip(axes.flatten()[:7], titles, maps):
        cmap = "gray" if ("Original" in title or "Consensus" in title or "Skeleton" in title or "BH Recon" in title) else "hot"
        vmax = 1.0 if cmap == "hot" else None
        ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title); ax.axis("off")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--signal_pct", type=float, default=85.0)
    parser.add_argument("--output_viz", type=str, default="data/consensus_preview.png")
    args = parser.parse_args()

    img = cv2.resize(cv2.imread(args.input, cv2.IMREAD_GRAYSCALE), (512, 512))
    res = compute_consensus_mask(img, signal_pct=args.signal_pct)
    visualise_consensus(res, img, args.output_viz)
    print(f"Done! Skeleton pixels: {(res['skeleton_mask'] > 0).sum()} / {img.size}")
