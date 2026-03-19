import os
import sys
import math
import cv2
import numpy as np
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Adăugăm root-ul proiectului în path pentru a putea importa din modulele existente
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from engine.train_targeted_vessel_segmentation import (
    TargetedSyntaxSegmentationDataset, 
    MultiTaskTargetedUNet, 
    _f1_iou_from_counts
)
from engine.train_lca_refinement import GlobalRefinementUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_path(path_arg: str) -> str:
    p = Path(path_arg).expanduser()
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())

# ==========================================
# 1. GPU-Accelerated Frangi Filter (PyTorch)
# ==========================================
def frangi_2d_torch(image: torch.Tensor, sigmas=[1.0, 2.0, 3.0], alpha=0.5, gamma=5.0):
    """
    Filtru Frangi calculat direct pe GPU pentru viteză extremă.
    image: (B, 1, H, W)
    """
    vesselness = torch.zeros_like(image)
    for sigma in sigmas:
        size = int(2 * round(3 * sigma) + 1)
        x = torch.arange(size, dtype=torch.float32, device=image.device) - size // 2
        y = x.view(-1, 1)
        x = x.view(1, -1)
        
        g = torch.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * math.pi * sigma**2)
        g_xx = (x**2 / sigma**4 - 1 / sigma**2) * g
        g_yy = (y**2 / sigma**4 - 1 / sigma**2) * g
        g_xy = (x * y / sigma**4) * g
        
        k_xx = g_xx.view(1, 1, size, size)
        k_yy = g_yy.view(1, 1, size, size)
        k_xy = g_xy.view(1, 1, size, size)
        
        pad = size // 2
        Dxx = F.conv2d(image, k_xx, padding=pad)
        Dyy = F.conv2d(image, k_yy, padding=pad)
        Dxy = F.conv2d(image, k_xy, padding=pad)
        
        trace = Dxx + Dyy
        det = Dxx * Dyy - Dxy**2
        
        sqrt_term = torch.sqrt((trace**2) / 4 - det + 1e-8)
        L1 = trace / 2 + sqrt_term
        L2 = trace / 2 - sqrt_term
        
        mask_sort = torch.abs(L1) > torch.abs(L2)
        lambda1 = torch.where(mask_sort, L2, L1)
        lambda2 = torch.where(mask_sort, L1, L2)
        
        Rb = torch.abs(lambda1) / (torch.abs(lambda2) + 1e-8)
        S = torch.sqrt(lambda1**2 + lambda2**2)
        
        exp_Rb = torch.exp(-(Rb**2) / (2 * alpha**2))
        exp_S = 1.0 - torch.exp(-(S**2) / (2 * gamma**2))
        
        v_sigma = exp_Rb * exp_S
        
        # Pe raze X, vasele sunt închise la culoare (black ridges)
        v_sigma = torch.where(lambda2 > 0, v_sigma, torch.zeros_like(v_sigma))
        vesselness = torch.max(vesselness, v_sigma)
        
    return vesselness

# ==========================================
# 2. Patch-Line Generation (XCAD Paper Concept)
# ==========================================
def patch_line_generation(binary_mask: np.ndarray, prob_map: np.ndarray, max_dist=30, prob_thresh=0.15) -> np.ndarray:
    """
    1. Scheletizeaza masca binară.
    2. Găsește capetele (endpoints) folosind un kernel de convoluție.
    3. Găsește cele mai apropiate capete folosind distanța Euclidiană.
    4. Trasează o linie de legătură (scanning region de 2 pixeli grosime).
    5. Validează linia DOAR dacă probabilitatea ascunsă a modelului sub linie este > prob_thresh.
    """
    # 1. Extragere schelet (1 pixel grosime)
    skel = skeletonize(binary_mask > 0).astype(np.uint8)
    
    # 2. Detectare Endpoints (Un pixel de capăt are un singur vecin)
    # Kernelul dă valoarea 10 pixelului central și adună vecinii. Un capăt va avea valoarea 11.
    kernel = np.array([[1, 1, 1], 
                       [1, 10, 1], 
                       [1, 1, 1]], dtype=np.uint8)
    filtered = ndi.convolve(skel, kernel, mode='constant', cval=0)
    endpoints = np.argwhere(filtered == 11) # Returneaza array de coordonate [y, x]
    
    out_mask = binary_mask.copy()
    if len(endpoints) < 2:
        return out_mask
        
    # 3. Calcul Distanțe Euclidiene
    dists = cdist(endpoints, endpoints, metric='euclidean')
    np.fill_diagonal(dists, np.inf) # Ignorăm distanța de la un punct la el însuși
    
    # 4. Căutare vecini și validare Patch-Line
    # Căutăm prin TOATE perechile din raza 'max_dist', nu doar cel mai apropiat punct!
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if dists[i, j] <= max_dist:
                pt1 = endpoints[i]
                pt2 = endpoints[j]
                
                # Cream un "Scanning Region" invizibil (mai gros, 3px pentru a capta harta de probabilitate mai bine)
                line_mask = np.zeros_like(binary_mask, dtype=np.uint8)
                cv2.line(line_mask, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 1, 3) 
                
                # Cât de "încrezător" a fost modelul pe ascuns în acea zonă?
                line_prob = prob_map[line_mask == 1].mean()
                
                # Dacă depășește pragul, aprobăm reparația (Valid Patch Line)
                if line_prob >= prob_thresh:
                    cv2.line(out_mask, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 1, 2)
                
    return out_mask

def postprocess_pipeline(prob_map: np.ndarray, cfg: dict, threshold: float = 0.50) -> tuple:
    # 1. Baseline clasic
    base_mask = (prob_map >= threshold).astype(np.uint8)
    
    # 2. Bypass Patch-Line (Modelul SOTA gestioneaza deja topologia nativ)
    # Folosim direct masca de baza pentru curatare
    work_mask = base_mask.copy()

    # 3. Size Refinement ONLY (Stergem doar zgomotul de arie, FARA aspect ratio invaziv)
    min_size = cfg.get("min_size", 40)
    
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work_mask.astype(np.uint8), connectivity=8)
    cleaned_mask = np.zeros_like(work_mask)
    
    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned_mask[labels == label_id] = 1
            
    return base_mask, cleaned_mask


# ==========================================
# 3. Evaluarea pe Test Set
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Test model cu Post-Procesare Topologica (Hysteresis + Frangi)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/syntax_targeted_vessel_segmentation_SOTA_512x512/best_model.pth")
    parser.add_argument("--ref-checkpoint", type=str, default="checkpoints/syntax_global_refinement_512x512/best_model.pth")
    parser.add_argument("--target-csv", type=str, default="results/arcade_patient_tables/patient_main_artery_targets.csv")
    parser.add_argument("--syntax-root", type=str, default="data/ARCADE/Unprocessed/arcade/syntax")
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    args.checkpoint = resolve_path(args.checkpoint)
    args.target_csv = resolve_path(args.target_csv)
    args.syntax_root = resolve_path(args.syntax_root)
    args.ref_checkpoint = resolve_path(args.ref_checkpoint)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"\n[EROARE] Nu am găsit modelul salvat la: {args.checkpoint}\nTe rog să te asiguri că antrenamentul a finalizat măcar o epocă sau folosește flag-ul --checkpoint pentru a indica locația corectă.")

    print(f"[INFO] Initializare Test Dataset...")
    test_ds = TargetedSyntaxSegmentationDataset(
        target_csv=args.target_csv,
        syntax_root=args.syntax_root,
        split="test",
        img_size=args.img_size,
        mode="test",
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"[INFO] Incarcare Model...")
    model = MultiTaskTargetedUNet(
        encoder_name="tu-efficientnetv2_s",
        in_channels=4, classes=1, aux_num_classes=4
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ref_model = None
    if os.path.isfile(args.ref_checkpoint):
        print(f"[INFO] Incarcare Refinement Model...")
        ref_model = GlobalRefinementUNet(encoder_name="tu-efficientnetv2_s", in_channels=5).to(device)
        ref_model.load_state_dict(torch.load(args.ref_checkpoint, map_location=device))
        ref_model.eval()
    else:
        print(f"[AVERTISMENT] Modelul de Refinement nu a fost gasit la {args.ref_checkpoint}. Evaluam doar Base Model.")

    # Parametri algoritm
    postprocess_cfg = {
        0: {"threshold": 0.45, "min_size": 30},
        1: {"threshold": 0.45, "min_size": 20}
    }

    # Salvam TP/FP/FN separat pentru Base si Refined
    tp_base, fp_base, fn_base = 0.0, 0.0, 0.0
    tp_ref, fp_ref, fn_ref = 0.0, 0.0, 0.0
    per_target = {
        0: {"tp_base": 0.0, "fp_base": 0.0, "fn_base": 0.0, "tp_ref": 0.0, "fp_ref": 0.0, "fn_ref": 0.0, "n": 0},
        1: {"tp_base": 0.0, "fp_base": 0.0, "fn_base": 0.0, "tp_ref": 0.0, "fp_ref": 0.0, "fn_ref": 0.0, "n": 0}
    }

    vis_dir = resolve_path("results/postprocess_vis")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"[INFO] Vizualizările vor fi salvate garantat în folderul: {vis_dir}")

    print(f"[INFO] Incepere Evaluare Comparativa...")
    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(test_loader))
        for inputs, masks, aux_masks, centerlines, vectors, target_ids, file_names in pbar:
            inputs = inputs.to(device)
            masks_np = masks.cpu().numpy().astype(np.uint8)[0, 0] 
            t_id = int(target_ids.item())
            
            # 1. Forward Pass Base Model
            seg_logits_soft, cls_logits, seg_both, aux_logits, center_logits, vec_preds = model(inputs)
            
            idx = torch.tensor([t_id], device=device).long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both.shape[-2], seg_both.shape[-1])
            seg_logits_base = torch.gather(seg_both, dim=1, index=idx).squeeze(1)
            base_prob_t = torch.sigmoid(seg_logits_base)
            base_prob = base_prob_t[0, 0].cpu().numpy()
            
            # 2. Forward Pass Refinement Model
            if ref_model is not None:
                ref_logits = ref_model(inputs, base_prob_t)
                ref_prob = torch.sigmoid(ref_logits)[0, 0].cpu().numpy()
            else:
                ref_prob = base_prob
                
            # 3. Generare Măști Curatate (Size Refinement)
            cfg = postprocess_cfg[t_id]
            _, base_clean = postprocess_pipeline(base_prob, cfg, threshold=cfg.get("threshold", 0.45))
            _, ref_clean = postprocess_pipeline(ref_prob, cfg, threshold=0.50) # Refiner a invatat sa separe la 0.5
            
            # --- SALVARE IMAGINI GARANTATĂ ---
            img_vis = (inputs[0, 0].cpu().numpy() * 255).astype(np.uint8)
            gt_vis = (masks_np * 255).astype(np.uint8)
            base_vis = (base_clean * 255).astype(np.uint8)
            ref_vis = (ref_clean * 255).astype(np.uint8)
            
            # Colaj: [Input | GroundTruth | Base Model | Refined Ensemble]
            panel = np.concatenate([img_vis, gt_vis, base_vis, ref_vis], axis=1)
            out_path = os.path.join(vis_dir, f"t{t_id}_{file_names[0]}")
            cv2.imwrite(out_path, panel)

            # 3. Calcul Metrici pentru BASELINE
            tp_b = float(np.logical_and(base_clean == 1, masks_np == 1).sum())
            fp_b = float(np.logical_and(base_clean == 1, masks_np == 0).sum())
            fn_b = float(np.logical_and(base_clean == 0, masks_np == 1).sum())
            tp_base += tp_b; fp_base += fp_b; fn_base += fn_b
            per_target[t_id]["tp_base"] += tp_b
            per_target[t_id]["fp_base"] += fp_b
            per_target[t_id]["fn_base"] += fn_b

            # 4. Calcul Metrici pentru REFINED
            tp_r = float(np.logical_and(ref_clean == 1, masks_np == 1).sum())
            fp_r = float(np.logical_and(ref_clean == 1, masks_np == 0).sum())
            fn_r = float(np.logical_and(ref_clean == 0, masks_np == 1).sum())
            tp_ref += tp_r; fp_ref += fp_r; fn_ref += fn_r
            per_target[t_id]["tp_ref"] += tp_r
            per_target[t_id]["fp_ref"] += fp_r
            per_target[t_id]["fn_ref"] += fn_r
            
            per_target[t_id]["n"] += 1

    # --- Afisare Rezultate ---
    f1_base_global, _ = _f1_iou_from_counts(tp_base, fp_base, fn_base)
    f1_ref_global, _ = _f1_iou_from_counts(tp_ref, fp_ref, fn_ref)
    
    f1_base_rca, _ = _f1_iou_from_counts(per_target[0]["tp_base"], per_target[0]["fp_base"], per_target[0]["fn_base"])
    f1_ref_rca, _ = _f1_iou_from_counts(per_target[0]["tp_ref"], per_target[0]["fp_ref"], per_target[0]["fn_ref"])
    
    f1_base_lca, _ = _f1_iou_from_counts(per_target[1]["tp_base"], per_target[1]["fp_base"], per_target[1]["fn_base"])
    f1_ref_lca, _ = _f1_iou_from_counts(per_target[1]["tp_ref"], per_target[1]["fp_ref"], per_target[1]["fn_ref"])

    print("\n" + "="*50)
    print(f"=== REZULTATE COMPARAȚIE BASE MODEL VS ENSEMBLE (REFINED) ===")
    print(f"Global F1: Base = {f1_base_global:.4f} | Refined = {f1_ref_global:.4f}")
    print(f"RCA F1:    Base = {f1_base_rca:.4f} | Refined = {f1_ref_rca:.4f}")
    print(f"LCA F1:    Base = {f1_base_lca:.4f} | Refined = {f1_ref_lca:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()