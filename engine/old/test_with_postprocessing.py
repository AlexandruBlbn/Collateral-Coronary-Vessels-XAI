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

# --- FORȚARE DETERMINISM HARDWARE ---
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
from utils.helpers import set_seed
set_seed(42) # Pentru reproducibilitate
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
def main(args_override=None):
    parser = argparse.ArgumentParser(description="Test model cu Post-Procesare Topologica (Hysteresis + Frangi)")
    parser.add_argument("--checkpoint", type=str, default="/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/ConvNextV2_Tiny_DCUnet_512x512/best_model.pth")
    parser.add_argument("--ref-checkpoint", type=str, default="/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/convnextv2_tiny_512x512_Refinement/best_model.pth")
    parser.add_argument("--target-csv", type=str, default="results/arcade_patient_tables/patient_main_artery_targets.csv")
    parser.add_argument("--syntax-root", type=str, default="data/ARCADE/Unprocessed/arcade/syntax")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--arch", type=str, default="dcn")
    parser.add_argument("--encoder-name", type=str, default="tu-convnextv2_tiny")
    args = parser.parse_args([] if args_override else None)
    
    if args_override is not None:
        for k, v in args_override.items():
            setattr(args, k, v)

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
        arch=args.arch,
        encoder_name=args.encoder_name,
        in_channels=4, classes=1, aux_num_classes=4
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ref_model = None
    if os.path.isfile(args.ref_checkpoint):
        print(f"[INFO] Incarcare Refinement Model...")
        ref_model = GlobalRefinementUNet(encoder_name=args.encoder_name, in_channels=5).to(device)
        ref_model.load_state_dict(torch.load(args.ref_checkpoint, map_location=device))
        ref_model.eval()
    else:
        print(f"[AVERTISMENT] Modelul de Refinement nu a fost gasit la {args.ref_checkpoint}. Evaluam doar Base Model.")

    # Parametri algoritm (rezultați din Grid Search)
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
    
    results_stats = [] # Pentru a memora scorul F1 per imagine

    rez_dir = resolve_path("rezultate")
    os.makedirs(rez_dir, exist_ok=True)
    print(f"[INFO] Rezultatele vor fi salvate garantat în folderul: {rez_dir}")

    print(f"[INFO] Trecerea 1: Calculare metrice globale și stocare performanță per imagine...")
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
            
            # Stocăm performanța imaginii curente pentru a extrage the best/worst la final
            f1_img_ref, _ = _f1_iou_from_counts(tp_r, fp_r, fn_r)
            results_stats.append({
                "file_name": file_names[0], "t_id": t_id, "f1": f1_img_ref
            })
            
            per_target[t_id]["n"] += 1

    # --- Afisare Rezultate ---
    f1_base_global, _ = _f1_iou_from_counts(tp_base, fp_base, fn_base)
    f1_ref_global, _ = _f1_iou_from_counts(tp_ref, fp_ref, fn_ref)
    
    f1_base_rca, _ = _f1_iou_from_counts(per_target[0]["tp_base"], per_target[0]["fp_base"], per_target[0]["fn_base"])
    f1_ref_rca, _ = _f1_iou_from_counts(per_target[0]["tp_ref"], per_target[0]["fp_ref"], per_target[0]["fn_ref"])
    
    f1_base_lca, _ = _f1_iou_from_counts(per_target[1]["tp_base"], per_target[1]["fp_base"], per_target[1]["fn_base"])
    f1_ref_lca, _ = _f1_iou_from_counts(per_target[1]["tp_ref"], per_target[1]["fp_ref"], per_target[1]["fn_ref"])

    report_text = f"""==================================================
=== REZULTATE COMPARAȚIE BASE MODEL VS ENSEMBLE (REFINED) ===
Global F1: Base = {f1_base_global:.4f} | Refined = {f1_ref_global:.4f}
RCA F1:    Base = {f1_base_rca:.4f} | Refined = {f1_ref_rca:.4f}
LCA F1:    Base = {f1_base_lca:.4f} | Refined = {f1_ref_lca:.4f}
=================================================="""

    print("\n" + report_text + "\n")
    
    # Salvare Tabel metrice
    with open(os.path.join(rez_dir, "tabel_metrice.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
        
    # --- Trecerea 2: Extragere Best/Worst 6 cazuri si Harta de Salienta (GradCAM) ---
    lca_stats = [x for x in results_stats if x['t_id'] == 1]
    rca_stats = [x for x in results_stats if x['t_id'] == 0]
    lca_stats.sort(key=lambda x: x['f1'], reverse=True)
    rca_stats.sort(key=lambda x: x['f1'], reverse=True)
    
    selected_files = {}
    if len(lca_stats) >= 3:
        selected_files[lca_stats[0]['file_name']] = f"Best_LCA_1_F1_{lca_stats[0]['f1']:.2f}"
        selected_files[lca_stats[1]['file_name']] = f"Best_LCA_2_F1_{lca_stats[1]['f1']:.2f}"
        selected_files[lca_stats[-1]['file_name']] = f"Worst_LCA_F1_{lca_stats[-1]['f1']:.2f}"
    if len(rca_stats) >= 3:
        selected_files[rca_stats[0]['file_name']] = f"Best_RCA_1_F1_{rca_stats[0]['f1']:.2f}"
        selected_files[rca_stats[1]['file_name']] = f"Best_RCA_2_F1_{rca_stats[1]['f1']:.2f}"
        selected_files[rca_stats[-1]['file_name']] = f"Worst_RCA_F1_{rca_stats[-1]['f1']:.2f}"

    print(f"[INFO] Generare Hărți Saliență pentru cele mai bune 4 și cele mai slabe 2 cazuri...")
    model.eval()
    
    def draw_text(img, text, pos):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4) # Umbra
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    for inputs, masks, aux_masks, centerlines, vectors, target_ids, file_names in tqdm(test_loader, desc="GradCAM"):
        fn = file_names[0]
        if fn not in selected_files:
            continue
            
        category_name = selected_files[fn]
        t_id = int(target_ids.item())
        inputs = inputs.to(device)
        masks = masks.to(device)
        
        # Refined Prob (doar pt desen)
        with torch.no_grad():
            seg_logits_soft, _, seg_both, _, _, _ = model(inputs)
            idx = torch.tensor([t_id], device=device).long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both.shape[-2], seg_both.shape[-1])
            base_prob_t = torch.sigmoid(torch.gather(seg_both, dim=1, index=idx).squeeze(1))
            ref_logits = ref_model(inputs, base_prob_t) if ref_model else base_prob_t
            ref_prob = torch.sigmoid(ref_logits)[0, 0].cpu().numpy()
            _, ref_clean = postprocess_pipeline(ref_prob, postprocess_cfg[t_id], threshold=0.50)
            
        # Harta de Salienta folosind GradCAM direct pe Input prin Encoder
        with torch.enable_grad():
            inputs_g = inputs.clone().detach().requires_grad_(True)
            feats = model.encoder(inputs_g)
            last_feat = feats[-1]
            last_feat.retain_grad()
            
            orig_size = inputs_g.shape[2:]
            decoder_out = model.decoder(feats)
            if decoder_out.shape[2:] != orig_size:
                decoder_out = F.interpolate(decoder_out, size=orig_size, mode='bilinear', align_corners=False)
                
            seg_both_g = torch.stack([model.seg_head_rca(decoder_out), model.seg_head_lca(decoder_out)], dim=1)
            idx_g = torch.tensor([t_id], device=device).long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both_g.shape[-2], seg_both_g.shape[-1])
            seg_logits_g = torch.gather(seg_both_g, dim=1, index=idx_g).squeeze(1)
            
            # Maximizam activarea pe zona vaselor de sange (Targeting direct)
            score = (seg_logits_g * masks).sum()
            model.zero_grad()
            score.backward()
            
            cam = F.relu((last_feat.grad.mean(dim=(2, 3), keepdim=True) * last_feat).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=orig_size, mode='bilinear', align_corners=False)
            cam = (cam - cam.amin()) / (cam.amax() - cam.amin() + 1e-8)
            cam_np = cam[0, 0].detach().cpu().numpy()

        # Construim Colajul
        img_bgr = cv2.cvtColor((inputs[0, 0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        gt_color = np.zeros_like(img_bgr); gt_color[:,:,1] = (masks[0, 0].cpu().numpy() * 255).astype(np.uint8)
        gt_overlay = cv2.addWeighted(img_bgr, 0.6, gt_color, 0.4, 0)
        
        ref_color = np.zeros_like(img_bgr); ref_color[:,:,0] = (ref_clean * 255).astype(np.uint8) # BGR (Blue pt predictie)
        ref_overlay = cv2.addWeighted(img_bgr, 0.6, ref_color, 0.4, 0)
        
        heatmap = cv2.applyColorMap((cam_np * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        sal_overlay = cv2.addWeighted(img_bgr, 0.4, heatmap, 0.8, 0)
        
        draw_text(img_bgr, "Input X-Ray", (10, 30)); draw_text(gt_overlay, "GT (Verde)", (10, 30))
        draw_text(ref_overlay, "Refined (Albastru)", (10, 30)); draw_text(sal_overlay, "Encoder Saliency", (10, 30))
        
        cv2.imwrite(os.path.join(rez_dir, f"{category_name}_{fn}"), np.concatenate([img_bgr, gt_overlay, ref_overlay, sal_overlay], axis=1))

    return {
        "F1 Base Global": f1_base_global,
        "F1 Refined Global": f1_ref_global,
        "Mean Dice Base (F1)": f1_base_global,
        "Mean Dice Refined (F1)": f1_ref_global,
        "F1 Refined RCA": f1_ref_rca,
        "F1 Refined LCA": f1_ref_lca,
    }

if __name__ == "__main__":
    main()