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

# ==============================================================================
# 🛠️ AICI IMPLEMENTEZI TU POST-PROCESAREA
# ==============================================================================
def apply_my_custom_postprocessing(prob_map: np.ndarray, img_np: np.ndarray) -> np.ndarray:
    """
    Aici poți testa orice filtru, morfologie sau logica vrei tu.
    
    Args:
        prob_map (np.ndarray): O imagine 2D (H, W) cu valori de tip float (0.0 -> 1.0),
                               reprezentând probabilitatea ieșită din modelul în cascadă.
        img_np (np.ndarray): Imaginea originală 2D (Canalul CLAHE) folosită ca reper fizic.
    
    Returns:
        np.ndarray: O imagine binară (H, W) cu valori 0 sau 1 (sau 0 și 255),
                    reprezentând masca finală curățată.
    """
    threshold = 0.5
    binary_mask = (prob_map >= threshold).astype(np.uint8)
    
    # --- 0. Eliminare Zgomot (Insule Mici / Despeckling) ---
    # Acest pas elimină punctele false izolate și aduce cea mai mare creștere F1
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned_mask = np.zeros_like(binary_mask)
    for label_id in range(1, n_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= 30: # Ștergem tot ce e sub 30 pixeli
            cleaned_mask[labels == label_id] = 1
    binary_mask = cleaned_mask
    
    # --- 1. Eliminare Catetere (Morfologie și Analiză de Formă) ---
    labeled = label(binary_mask)
    props = regionprops(labeled)
    for prop in props:
        # [FIX] Protejăm vasele mari! 
        min_row, min_col, max_row, max_col = prop.bbox
        length = max(max_row - min_row, max_col - min_col)
        # Estimăm grosimea (Aria totală împărțită la lungimea cutiei încadratoare)
        avg_width = prop.area / max(1, length)
        
        # Relaxăm puțin pragurile pentru a prinde mai multe "fire" false
        if prop.eccentricity > 0.985 and avg_width < 8.0 and length > 50:
            binary_mask[labeled == prop.label] = 0
            
    # --- 2. Reconectare Vase (Scheletizare + Endpoints + Frangi) ---
    skel = skeletonize(binary_mask > 0).astype(np.uint8)
    # Kernel pentru a număra vecinii. Endpoint-ul are valoarea 11 (10 centrul + 1 vecin)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = ndi.convolve(skel, kernel, mode='constant', cval=0)
    endpoints = np.argwhere(filtered == 11)
    
    # Harta de validare Frangi de pe imaginea reală (X-Ray are vase întunecate -> black_ridges=True)
    f_map = frangi(img_np, sigmas=range(1, 4, 1), black_ridges=True)
    if f_map.max() > 0:
        f_map = f_map / f_map.max()
        
    if len(endpoints) >= 2:
        dists = cdist(endpoints, endpoints, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                if dists[i, j] <= 30: # Distanța maximă de 30 pixeli
                    p1, p2 = endpoints[i], endpoints[j]
                    # Trasăm linia de test invizibilă (groasă de 3 px) și vedem ce zice Frangi sub ea
                    line_mask = np.zeros_like(binary_mask)
                    cv2.line(line_mask, (p1[1], p1[0]), (p2[1], p2[0]), 1, 3)
                    
                    if f_map[line_mask == 1].mean() > 0.05: # Există un vas slab acolo
                        # Dacă Frangi îl validează, desenăm vasul mai gros (grosime 3) pentru a capta corect TP-urile
                        cv2.line(binary_mask, (p1[1], p1[0]), (p2[1], p2[0]), 1, 3)
    
    return binary_mask

# ==============================================================================
# ⚙️ CONFIGURAȚII MODELE
# ==============================================================================
# Pune aici calea către modelele pe care vrei să le folosești
BASE_MODEL_CKPT = "/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/GridSearch_efficientnetv2_s_fpn_Base/best_model.pth"
REF_MODEL_CKPT = "/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/GridSearch_efficientnetv2_s_fpn_Refined/best_model.pth"
ENCODER_NAME = "tu-efficientnetv2_s"
DECODER_ARCH = "fpn"

def main():
    print(f"[{device.type.upper()}] Pornire script de testare post-procesare...")

    # 1. Încărcare Dataset Test
    test_ds = TargetedSyntaxSegmentationDataset(
        target_csv="results/arcade_patient_tables/patient_main_artery_targets.csv",
        syntax_root="data/ARCADE/Unprocessed/arcade/syntax",
        split="test",
        img_size=512,
        mode="test"
    )
    # Batch size 1 pentru a extrage manual 4 pacienți
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 2. Încărcare Modele
    print("[INFO] Încărcare Base Model...")
    base_model = MultiTaskTargetedUNet(arch=DECODER_ARCH, encoder_name=ENCODER_NAME, in_channels=4, img_size=512).to(device)
    if os.path.exists(BASE_MODEL_CKPT):
        base_model.load_state_dict(torch.load(BASE_MODEL_CKPT, map_location=device))
    else:
        print(f"[AVERTISMENT] Nu am găsit modelul de bază la {BASE_MODEL_CKPT}")
    base_model.eval()

    print("[INFO] Încărcare Refinement Model...")
    ref_model = GlobalRefinementUNet(encoder_name=ENCODER_NAME, in_channels=5).to(device)
    if os.path.exists(REF_MODEL_CKPT):
        ref_model.load_state_dict(torch.load(REF_MODEL_CKPT, map_location=device))
    else:
        print(f"[AVERTISMENT] Nu am găsit modelul de rafinare la {REF_MODEL_CKPT}")
    ref_model.eval()

    # 3. Pregătire folder pentru salvarea imaginilor individuale
    plots_dir = "rezultate/test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"[INFO] Imaginile individuale pentru fiecare pacient vor fi salvate in '{plots_dir}'")

    tp_raw, fp_raw, fn_raw = 0.0, 0.0, 0.0
    tp_post, fp_post, fn_post = 0.0, 0.0, 0.0
    col_titles = ["Data (Input)", "Pred (Base Model)", "Refined + Postprocessing", "Ground Truth"]

    print("[INFO] Evaluare post-procesare pe întregul set de test...")
    with torch.no_grad():
        for idx, (inputs, masks, _, _, _, target_ids, file_names) in enumerate(tqdm(test_loader, desc="Testare")):
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
            
            # --- C. Trecerea în format Imagine (NumPy) pentru utilizator ---
            # Extragem imaginea de input (Canalul 0 - CLAHE) pentru vizualizare
            img_np = inputs[0, 0].cpu().numpy() 
            
            # Masca Reală (Ground Truth)
            gt_np = masks[0, 0].cpu().numpy().astype(np.uint8)
            
            # Predicția Base binarizată brut (doar pentru vizualizarea performanței inițiale)
            base_pred_np = (base_prob_t[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            
            # Predicția Refined (probabilități) - Asta merge în funcția ta
            ref_prob_np = ref_prob_t[0, 0].cpu().numpy()
            
            # --- D. Aplicăm Post-Procesarea Ta ---
            final_mask_np = apply_my_custom_postprocessing(ref_prob_np, img_np)
            raw_mask_np = (ref_prob_np >= 0.5).astype(np.uint8)
            
            # --- E. Numărăm pixelii pentru Metricile F1 (La nivel de test set) ---
            tp_raw += np.logical_and(raw_mask_np == 1, gt_np == 1).sum()
            fp_raw += np.logical_and(raw_mask_np == 1, gt_np == 0).sum()
            fn_raw += np.logical_and(raw_mask_np == 0, gt_np == 1).sum()

            tp_post += np.logical_and(final_mask_np == 1, gt_np == 1).sum()
            fp_post += np.logical_and(final_mask_np == 1, gt_np == 0).sum()
            fn_post += np.logical_and(final_mask_np == 0, gt_np == 1).sum()

            # --- F. Calcul F1 per pacient și Salvare Plot Individual ---
            p_tp = np.logical_and(final_mask_np == 1, gt_np == 1).sum()
            p_fp = np.logical_and(final_mask_np == 1, gt_np == 0).sum()
            p_fn = np.logical_and(final_mask_np == 0, gt_np == 1).sum()
            p_f1 = (2.0 * p_tp) / max(1e-8, 2.0 * p_tp + p_fp + p_fn)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            target_name = "RCA" if t_id == 0 else "LCA"
            
            # Coloana 0: Data
            axes[0].imshow(img_np, cmap="gray")
            axes[0].axis("off")
            axes[0].text(10, 30, f"{file_names[0]} ({target_name})", color="white", fontsize=12, backgroundcolor="black")
            axes[0].set_title(col_titles[0], fontsize=14, fontweight="bold")
            
            # Coloana 1: Pred Base
            axes[1].imshow(base_pred_np, cmap="gray", vmin=0, vmax=1)
            axes[1].axis("off")
            axes[1].set_title(col_titles[1], fontsize=14, fontweight="bold")
            
            # Coloana 2: Refined + Postprocessing
            axes[2].imshow(final_mask_np, cmap="gray", vmin=0, vmax=1)
            axes[2].axis("off")
            axes[2].set_title(f"Refined + Post (F1: {p_f1:.4f})", fontsize=14, fontweight="bold")
            
            # Coloana 3: Ground Truth
            axes[3].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            axes[3].axis("off")
            axes[3].set_title(col_titles[3], fontsize=14, fontweight="bold")
            
            plt.tight_layout()
            save_name = file_names[0].replace('.png', f'_F1_{p_f1:.4f}.png')
            plt.savefig(os.path.join(plots_dir, save_name), dpi=100, bbox_inches='tight')
            plt.close(fig) # Curățare memorie pentru următorul loop

    # 4. Afișare Scoruri Totale de F1
    f1_raw = (2.0 * tp_raw) / max(1e-8, 2.0 * tp_raw + fp_raw + fn_raw)
    f1_post = (2.0 * tp_post) / max(1e-8, 2.0 * tp_post + fp_post + fn_post)

    print("\n" + "="*60)
    print("=== REZULTATE PE TOT SETUL DE TEST ===")
    print(f"F1 Fără Post-Procesare (Cascadă brută): {f1_raw:.4f}")
    print(f"F1 CU Post-Procesare Custom:            {f1_post:.4f}")
    diff = f1_post - f1_raw
    print(f"Diferență:                              {'+' if diff >= 0 else ''}{diff:.4f}")
    print("="*60 + "\n")

    print(f"[✅ SUCCES] Toate imaginile individuale au fost salvate in '{plots_dir}'.")

if __name__ == "__main__":
    main()