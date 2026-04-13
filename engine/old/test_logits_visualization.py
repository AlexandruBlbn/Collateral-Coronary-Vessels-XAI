import os
import sys
import numpy as np
import torch
import cv2
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.filters import frangi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Adăugăm rădăcina proiectului în path pentru importuri
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from engine.train_targeted_vessel_segmentation import TargetedSyntaxSegmentationDataset, MultiTaskTargetedUNet
from engine.train_lca_refinement import GlobalRefinementUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Helper: detect long straight line-like structures (typical catheter prior)
# ------------------------------------------------------------------------------
def _build_catheter_prior(img_np: np.ndarray) -> np.ndarray:
    h, w = img_np.shape
    img_u8 = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    edges = cv2.Canny(img_blur, 40, 120)

    min_line_length = int(0.18 * min(h, w))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=60,
        minLineLength=max(60, min_line_length),
        maxLineGap=12,
    )

    catheter_prior = np.zeros((h, w), dtype=np.uint8)
    if lines is None:
        return catheter_prior

    border_margin = int(0.05 * min(h, w))
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)
        length = np.hypot(x2 - x1, y2 - y1)
        if length < max(80, 0.2 * min(h, w)):
            continue

        # Catheters are typically line-like structures entering from the border.
        touches_border = (
            (x1 < border_margin) or (x1 > w - border_margin) or
            (x2 < border_margin) or (x2 > w - border_margin) or
            (y1 < border_margin) or (y1 > h - border_margin) or
            (y2 < border_margin) or (y2 > h - border_margin)
        )
        if not touches_border:
            continue

        cv2.line(catheter_prior, (x1, y1), (x2, y2), 1, thickness=6)

    catheter_prior = cv2.dilate(
        catheter_prior,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    return catheter_prior


def _prune_straight_border_spurs(binary_mask: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
    skel = skeletonize(binary_mask > 0).astype(np.uint8)
    if skel.sum() == 0:
        return binary_mask

    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = ndi.convolve(skel, kernel, mode='constant', cval=0)
    endpoints = np.argwhere(filtered == 11)
    branchpoints = {tuple(p) for p in np.argwhere(filtered >= 13)}

    h, w = binary_mask.shape
    border_margin = int(0.05 * min(h, w))
    min_path_len = int(0.12 * min(h, w))
    keep_prob_gate = 0.985

    skel_coords = set(map(tuple, np.argwhere(skel > 0)))
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def _neighbors(p):
        r, c = p
        out = []
        for dr, dc in offsets:
            q = (r + dr, c + dc)
            if q in skel_coords:
                out.append(q)
        return out

    remove_skel = np.zeros_like(binary_mask, dtype=np.uint8)

    for ep in endpoints:
        start = (int(ep[0]), int(ep[1]))
        r, c = start
        near_border = (
            r < border_margin or r >= h - border_margin or
            c < border_margin or c >= w - border_margin
        )
        if not near_border:
            continue

        path = [start]
        prev = None
        curr = start
        ambiguous = False

        for _ in range(4 * min(h, w)):
            nbrs = _neighbors(curr)
            if prev is not None:
                nbrs = [n for n in nbrs if n != prev]

            if len(nbrs) == 0:
                break
            if len(nbrs) > 1:
                ambiguous = True
                break

            nxt = nbrs[0]
            path.append(nxt)
            prev, curr = curr, nxt

            if curr in branchpoints:
                break

        if ambiguous or len(path) < min_path_len:
            continue

        p0 = np.array(path[0], dtype=np.float32)
        p1 = np.array(path[-1], dtype=np.float32)
        euclid = float(np.linalg.norm(p1 - p0))
        geodesic = float(max(1, len(path) - 1))
        straightness = euclid / geodesic

        path_arr = np.array(path, dtype=np.int32)
        mean_prob = float(prob_map[path_arr[:, 0], path_arr[:, 1]].mean())

        # Catheter-like spur: long, straight, border-connected branch.
        if straightness >= 0.92 and mean_prob < keep_prob_gate:
            remove_skel[path_arr[:, 0], path_arr[:, 1]] = 1

    if remove_skel.sum() == 0:
        return binary_mask

    remove_mask = cv2.dilate(
        remove_skel,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    pruned = binary_mask.copy()
    pruned[remove_mask > 0] = 0
    return pruned

# ==============================================================================
# 🛠️ AICI IMPLEMENTEZI TU POST-PROCESAREA
# ==============================================================================
def apply_my_custom_postprocessing(prob_map: np.ndarray, img_np: np.ndarray) -> np.ndarray:
    threshold = 0.5
    binary_mask = (prob_map >= threshold).astype(np.uint8)
    
    # --- 0. Eliminare Zgomot ---
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned_mask = np.zeros_like(binary_mask)
    for label_id in range(1, n_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= 30:
            cleaned_mask[labels == label_id] = 1
    binary_mask = cleaned_mask

    # --- 0b. Catheter suppression with image prior + confidence gating ---
    catheter_prior = _build_catheter_prior(img_np)
    preserve_high_conf = prob_map >= 0.85
    remove_mask = np.logical_and(catheter_prior == 1, np.logical_not(preserve_high_conf))
    binary_mask[remove_mask] = 0
    
    # --- 1. Eliminare Catetere ---
    labeled = label(binary_mask)
    props = regionprops(labeled)
    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox
        length = max(max_row - min_row, max_col - min_col)
        avg_width = prop.area / max(1, length)
        
        if prop.eccentricity > 0.80 and avg_width < 8.0 and length > 50:
            binary_mask[labeled == prop.label] = 0

    # Re-clean after catheter suppression and shape filtering.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned_mask = np.zeros_like(binary_mask)
    for label_id in range(1, n_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= 30:
            cleaned_mask[labels == label_id] = 1
    binary_mask = cleaned_mask

    # --- 1b. Prune long straight border-connected branches (catheter-like spurs) ---
    binary_mask = _prune_straight_border_spurs(binary_mask, prob_map)

    # Final cleanup after spur pruning.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned_mask = np.zeros_like(binary_mask)
    for label_id in range(1, n_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= 30:
            cleaned_mask[labels == label_id] = 1
    binary_mask = cleaned_mask
            
    # --- 2. Reconectare Vase (Frangi) ---
    skel = skeletonize(binary_mask > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = ndi.convolve(skel, kernel, mode='constant', cval=0)
    endpoints = np.argwhere(filtered == 11)
    
    f_map = frangi(img_np, sigmas=range(1, 4, 1), black_ridges=True)
    if f_map.max() > 0:
        f_map = f_map / f_map.max()
        
    if len(endpoints) >= 2:
        dists = cdist(endpoints, endpoints, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                if dists[i, j] <= 30:
                    p1, p2 = endpoints[i], endpoints[j]
                    line_mask = np.zeros_like(binary_mask)
                    cv2.line(line_mask, (p1[1], p1[0]), (p2[1], p2[0]), 1, 3)
                    
                    if f_map[line_mask == 1].mean() > 0.05:
                        cv2.line(binary_mask, (p1[1], p1[0]), (p2[1], p2[0]), 1, 3)
    
    return binary_mask

# ==============================================================================
# ⚙️ CONFIGURAȚII MODELE
# ==============================================================================
BASE_MODEL_CKPT = "/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/GridSearch_efficientnetv2_s_fpn_Base/best_model.pth"
REF_MODEL_CKPT = "/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/GridSearch_efficientnetv2_s_fpn_Refined/best_model.pth"
ENCODER_NAME = "tu-efficientnetv2_s"
DECODER_ARCH = "fpn"

def main():
    print(f"[{device.type.upper()}] Pornire script de vizualizare LOGITS (fara Sigmoid)...")

    test_ds = TargetedSyntaxSegmentationDataset(
        target_csv="results/arcade_patient_tables/patient_main_artery_targets.csv",
        syntax_root="data/ARCADE/Unprocessed/arcade/syntax",
        split="test",
        img_size=512,
        mode="test"
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print("[INFO] Încărcare Base Model...")
    base_model = MultiTaskTargetedUNet(arch=DECODER_ARCH, encoder_name=ENCODER_NAME, in_channels=4, img_size=512).to(device)
    if os.path.exists(BASE_MODEL_CKPT):
        base_model.load_state_dict(torch.load(BASE_MODEL_CKPT, map_location=device))
    base_model.eval()

    print("[INFO] Încărcare Refinement Model...")
    ref_model = GlobalRefinementUNet(encoder_name=ENCODER_NAME, in_channels=5).to(device)
    if os.path.exists(REF_MODEL_CKPT):
        ref_model.load_state_dict(torch.load(REF_MODEL_CKPT, map_location=device))
    ref_model.eval()

    plots_dir = "rezultate/test_logits_plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"[INFO] Imaginile comparative for fi salvate in '{plots_dir}'")

    col_titles = [
        "Data (Input)", 
        "Refined Prob (CU Sigmoid)", 
        "Refined Logits (FĂRĂ Sigmoid)", 
        "Refined + Postprocessing", 
        "Ground Truth"
    ]

    with torch.no_grad():
        for idx, (inputs, masks, _, _, _, target_ids, file_names) in enumerate(tqdm(test_loader, desc="Testare")):
            # Limităm la primii 50 de pacienți ca să nu umplem partiția inutil
            if idx >= 50:
                break
                
            inputs = inputs.to(device)
            t_id = int(target_ids.item())
            
            # --- A. Inferență Model Bază ---
            _, _, seg_both, _, _, _ = base_model(inputs)
            target_idx = torch.tensor([t_id], device=device).long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both.shape[-2], seg_both.shape[-1])
            seg_logits_base = torch.gather(seg_both, dim=1, index=target_idx).squeeze(1)
            base_prob_t = torch.sigmoid(seg_logits_base)
            
            # --- B. Inferență Model Rafinare (Cascada) ---
            ref_logits = ref_model(inputs, base_prob_t)
            ref_prob_t = torch.sigmoid(ref_logits)
            
            # --- C. Trecerea în format Imagine (NumPy) ---
            img_np = inputs[0, 0].cpu().numpy() 
            gt_np = masks[0, 0].cpu().numpy().astype(np.uint8)
            
            # Probabilitățile Refined (Turtite între 0 și 1 de Sigmoid)
            ref_prob_np = ref_prob_t[0, 0].cpu().numpy()
            
            # Logits Refined (Brut: de la infinit negativ la infinit pozitiv)
            ref_logits_np = ref_logits[0, 0].cpu().numpy()
            
            # --- D. Aplicăm Post-Procesarea ---
            final_mask_np = apply_my_custom_postprocessing(ref_prob_np, img_np)
            
            # --- E. Calcul F1 per pacient ---
            p_tp = np.logical_and(final_mask_np == 1, gt_np == 1).sum()
            p_fp = np.logical_and(final_mask_np == 1, gt_np == 0).sum()
            p_fn = np.logical_and(final_mask_np == 0, gt_np == 1).sum()
            p_f1 = (2.0 * p_tp) / max(1e-8, 2.0 * p_tp + p_fp + p_fn)
            
            # --- F. Generare PLOT COMPARAȚIE ---
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            target_name = "RCA" if t_id == 0 else "LCA"
            
            # 0. Data
            axes[0].imshow(img_np, cmap="gray")
            axes[0].axis("off")
            axes[0].text(10, 30, f"{file_names[0]} ({target_name})", color="white", fontsize=12, backgroundcolor="black")
            axes[0].set_title(col_titles[0], fontsize=12, fontweight="bold")
            
            # 1. Refined Prob (CU Sigmoid) - Aici poți vedea cum "dispare" informația
            axes[1].imshow(ref_prob_np, cmap="gray", vmin=0, vmax=1)
            axes[1].axis("off")
            axes[1].set_title(col_titles[1], fontsize=12, fontweight="bold")
            
            # 2. Refined Logits (FĂRĂ Sigmoid) 
            # Folosim 'coolwarm' centrat in zero: Rosu = Vas de sange, Albastru = Fundal
            vmax = max(abs(ref_logits_np.max()), abs(ref_logits_np.min()))
            im_log = axes[2].imshow(ref_logits_np, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            axes[2].axis("off")
            axes[2].set_title(col_titles[2], fontsize=12, fontweight="bold")
            # Adaugam colorbar micut pentru a vedea scara numerica bruta
            fig.colorbar(im_log, ax=axes[2], fraction=0.046, pad=0.04)
            
            # 3. Refined + Postprocessing
            axes[3].imshow(final_mask_np, cmap="gray", vmin=0, vmax=1)
            axes[3].axis("off")
            axes[3].set_title(f"Post-Process (F1: {p_f1:.4f})", fontsize=12, fontweight="bold")
            
            # 4. Ground Truth
            axes[4].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            axes[4].axis("off")
            axes[4].set_title(col_titles[4], fontsize=12, fontweight="bold")
            
            plt.tight_layout()
            save_name = file_names[0].replace('.png', f'_F1_{p_f1:.4f}.png')
            plt.savefig(os.path.join(plots_dir, save_name), dpi=100, bbox_inches='tight')
            plt.close(fig)

    print(f"[✅ SUCCES] Imaginile comparative au fost salvate in '{plots_dir}'.")

if __name__ == "__main__":
    main()