import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vessel_consensus import compute_consensus_mask

def extract_patch_heatmap(img_path, patch_size=8, out_path="data/patch_heatmap.png"):
    # 1. Read and resize image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512))
    
    # 2. Get the continuous, soft physical signals
    res = compute_consensus_mask(img)
    
    # soft_consensus is Frangi * Coherence * Blackhat (values 0 to 1)
    # It strongly suppresses background noise while keeping continuous vessel structures
    continuous_signal = res["consensus"]
    
    # Normalize the continuous signal to 0-1 for the heatmap
    if continuous_signal.max() > 0:
        continuous_signal = continuous_signal / continuous_signal.max()
        
    # 3. Patch-ify the signal
    h, w = continuous_signal.shape
    h_patches = h // patch_size
    w_patches = w // patch_size
    
    patch_scores = np.zeros((h_patches, w_patches), dtype=np.float32)
    
    for i in range(h_patches):
        for j in range(w_patches):
            patch_data = continuous_signal[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            # Use the 90th percentile within the patch. 
            patch_scores[i, j] = np.percentile(patch_data, 90)
            
    # 4. Resize the patch scores back to original image size using NEAREST neighbor 
    # to keep the blocky grid aesthetic
    blocky_heatmap = cv2.resize(patch_scores, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 5. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot A: Original
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot B: The Blocky Patch Heatmap alone
    axes[1].imshow(blocky_heatmap, cmap="hot", vmin=0, vmax=1.0)
    axes[1].set_title(f"Patch-wise Vessel Confidence ({patch_size}x{patch_size})")
    axes[1].axis("off")
    
    # Plot C: Overlay
    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(blocky_heatmap, cmap="hot", alpha=0.4, vmin=0, vmax=1.0)
    axes[2].set_title("Overlay (Feedback Guidance)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved patch heatmap to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/extra/trainB/0.png")
    parser.add_argument("--patch", type=int, default=8)
    args = parser.parse_args()
    extract_patch_heatmap(args.input, args.patch)
