import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

#Pathuri pentru dataseturile de stage 1 training
cadica = r"../XA-170K/dataset/cadica"
syntax = r"../XA-170K/dataset/syntax"
xcad = r"../XA-170K/dataset/xcad"
arcadeStenoza = r"ARCADE/processed/stenoza/data"
arcadeSyntax = r"ARCADE/processed/syntax/data"
#Stage 2 - coronaryDominance
coronaryDominance = r"../XA-170K/dataset/coronarydominance"

def plot_distribution():
    import matplotlib.pyplot as plt
    import cv2
    base = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Gather all image paths
    paths = [os.path.abspath(os.path.join(base, p)) for p in [cadica, syntax, xcad, arcadeStenoza, arcadeSyntax]]
    images = [os.path.join(r, f) for p in paths if os.path.exists(p) 
              for r, _, files in os.walk(p) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
              
    if not images:
        return

    # Image Processing Pipeline (CLAHE + Bilateral smoothing)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    def process_image(p):
        img_np = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        clahe = clahe_obj.apply(img_np)
        return cv2.bilateralFilter(clahe, d=9, sigmaColor=25, sigmaSpace=9.0)

    # 2. Extract stats over all pixels
    raw_sum = raw_sq = proc_sum = proc_sq = total = 0
    for p in images:
        try:
            raw = np.array(Image.open(p).convert("L"), dtype=np.float32)
            proc = process_image(p).astype(np.float32)
            
            raw_sum += raw.sum()
            raw_sq += (raw**2).sum()
            proc_sum += proc.sum()
            proc_sq += (proc**2).sum()
            total += raw.size
        except Exception:
            pass
            
    raw_mean = raw_sum / total
    raw_std = np.sqrt((raw_sq / total) - (raw_mean ** 2))
    proc_mean = proc_sum / total
    proc_std = np.sqrt((proc_sq / total) - (proc_mean ** 2))
    
    print(f"Raw Stats - Mean: {raw_mean:.4f}, Std: {raw_std:.4f}")
    print(f"Processed Stats - Mean: {proc_mean:.4f}, Std: {proc_std:.4f}")
    
    # 3. Accumulate Histograms
    raw_counts, bins = np.histogram([], bins=50, range=(0, 256))
    proc_counts, _ = np.histogram([], bins=50, range=(0, 256))
    
    for p in images:
        try:
            raw = np.array(Image.open(p).convert("L"), dtype=np.float32)
            proc = process_image(p).astype(np.float32)
            raw_counts += np.histogram(raw, bins=50, range=(0, 256))[0]
            proc_counts += np.histogram(proc, bins=50, range=(0, 256))[0]
        except Exception:
            pass

    # 4. Plot & Save
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(bins[:-1], raw_counts, width=np.diff(bins), align="edge", color='blue', alpha=0.7)
    axes[0].set(title=f"Raw Data\nMean: {raw_mean:.2f} | Std: {raw_std:.2f}", xlabel="Pixel Value", ylabel="Frequency")
    
    axes[1].bar(bins[:-1], proc_counts, width=np.diff(bins), align="edge", color='purple', alpha=0.7)
    axes[1].set(title=f"CLAHE + Bilateral\nMean: {proc_mean:.2f} | Std: {proc_std:.2f}", xlabel="Pixel Value", ylabel="Frequency")
    
    fig_dir = os.path.abspath(os.path.join(base, "../figs"))
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "stage1_distribution.png"), bbox_inches='tight')
    plt.close()
    print("Saved distribution plot with stats to ../figs/stage1_distribution.png")

def plot_vessel_vs_background():
    import matplotlib.pyplot as plt
    base = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Match images and labels
    img_dir = os.path.abspath(os.path.join(base, "ARCADE/processed/syntax/data"))
    lbl_dir = os.path.abspath(os.path.join(base, "ARCADE/processed/syntax/label"))
    
    if not os.path.exists(lbl_dir):
        print("Label directory not found.")
        return
        
    pairs = []
    for f in os.listdir(lbl_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(img_dir, f)
            lbl_path = os.path.join(lbl_dir, f)
            if os.path.exists(img_path):
                pairs.append((img_path, lbl_path))
                
    if not pairs:
        print("No matching image-label pairs found.")
        return
        
    # 2. Accumulate stats and histograms
    num_bins = 50
    bins = np.linspace(0, 256, num_bins + 1)
    
    v_sum = v_sq_sum = v_total = 0
    b_sum = b_sq_sum = b_total = 0
    
    v_counts = np.zeros(num_bins, dtype=np.int64)
    b_counts = np.zeros(num_bins, dtype=np.int64)
    
    for img_p, lbl_p in pairs:
        try:
            img = np.array(Image.open(img_p).convert("L"), dtype=np.float32)
            mask = np.array(Image.open(lbl_p).convert("L"))
            
            v_mask = mask > 127
            b_mask = ~v_mask
            
            v_pix = img[v_mask]
            b_pix = img[b_mask]
            
            # Update Vessel Stats
            v_sum += v_pix.sum()
            v_sq_sum += (v_pix**2).sum()
            v_total += v_pix.size
            v_counts += np.histogram(v_pix, bins=bins)[0]
            
            # Update Background Stats
            b_sum += b_pix.sum()
            b_sq_sum += (b_pix**2).sum()
            b_total += b_pix.size
            b_counts += np.histogram(b_pix, bins=bins)[0]
        except Exception:
            pass
            
    v_mean = v_sum / v_total
    v_std = np.sqrt((v_sq_sum / v_total) - (v_mean ** 2))
    
    b_mean = b_sum / b_total
    b_std = np.sqrt((b_sq_sum / b_total) - (b_mean ** 2))
    
    print(f"Vessels - Mean: {v_mean:.2f}, Std: {v_std:.2f}")
    print(f"Background - Mean: {b_mean:.2f}, Std: {b_std:.2f}")
    
    # 3. Plot Overlaid histograms
    plt.figure(figsize=(10, 5))
    plt.bar(bins[:-1], b_counts, width=np.diff(bins), align="edge", color='blue', alpha=0.5, label=f'Background (Mean: {b_mean:.1f})')
    plt.bar(bins[:-1], v_counts, width=np.diff(bins), align="edge", color='red', alpha=0.7, label=f'Vessels (Mean: {v_mean:.1f})')
    
    plt.title("Vessel vs. Background Pixel Intensity Distribution")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")
    plt.legend()
    
    fig_dir = os.path.abspath(os.path.join(base, "../figs"))
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "vessel_vs_background_distribution.png"), bbox_inches='tight')
    plt.close()
    print("Saved comparison plot to ../figs/vessel_vs_background_distribution.png") 