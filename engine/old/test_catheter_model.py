import os
import sys
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Adăugăm rădăcina proiectului în path pentru importuri
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from engine.train_targeted_vessel_segmentation import TargetedSyntaxSegmentationDataset, MultiTaskTargetedUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# ⚙️ CONFIGURAȚII PENTRU MODELUL ANTRENAT CU CATETERE
# ==============================================================================
CKPT_PATH = "/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/DUUnet_efficientnetv2_testCathere/best_model.pth"
ENCODER_NAME = "tu-efficientnetv2_s"
DECODER_ARCH = "dcn"

def get_f1_score(tp, fp, fn):
    return (2.0 * tp) / max(1e-8, (2.0 * tp + fp + fn))

def main():
    print("="*60)
    print(f"[{device.type.upper()}] Evaluare Model Antrenat cu Fake Catheters")
    print("="*60)

    if not os.path.exists(CKPT_PATH):
        print(f"[EROARE] Nu am găsit checkpoint-ul la: {CKPT_PATH}")
        print("Asigură-te că antrenamentul experimentului 'DUUnet_efficientnetv2_testCathere' s-a terminat cu succes.")
        return

    # 1. Încărcare Dataset Test
    print("[INFO] Încărcare Dataset Test...")
    test_ds = TargetedSyntaxSegmentationDataset(
        target_csv="results/arcade_patient_tables/patient_main_artery_targets.csv",
        syntax_root="data/ARCADE/Unprocessed/arcade/syntax",
        split="test",
        img_size=512,
        mode="test"
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 2. Încărcare Model
    print("[INFO] Încărcare Model (Base Model + DCN)...")
    model = MultiTaskTargetedUNet(
        arch=DECODER_ARCH, 
        encoder_name=ENCODER_NAME, 
        in_channels=4, 
        img_size=512
    ).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # 3. Pregătire Directoare Salvare Imagini
    plots_dir = "rezultate/test_catheters_plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"[INFO] Imaginile de analiză vizuală vor fi salvate in '{plots_dir}'")

    tp_all, fp_all, fn_all = 0.0, 0.0, 0.0
    stats_per_target = {
        0: {"tp": 0.0, "fp": 0.0, "fn": 0.0},
        1: {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    }

    with torch.no_grad():
        for idx, (inputs, masks, _, _, _, target_ids, file_names) in enumerate(tqdm(test_loader, desc="Inferență Test")):
            inputs = inputs.to(device)
            t_id = int(target_ids.item())
            target_name = "RCA" if t_id == 0 else "LCA"
            
            # --- Inferență ---
            _, _, seg_both, _, _, _ = model(inputs)
            target_idx = torch.tensor([t_id], device=device).long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both.shape[-2], seg_both.shape[-1])
            seg_logits = torch.gather(seg_both, dim=1, index=target_idx).squeeze(1)
            
            probs = torch.sigmoid(seg_logits)
            pred_mask = (probs[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)
            gt_mask = masks[0, 0].cpu().numpy().astype(np.uint8)
            img_np = inputs[0, 0].cpu().numpy() # CLAHE img
            
            # --- Metrici ---
            tp = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
            fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
            fn = np.logical_and(pred_mask == 0, gt_mask == 1).sum()
            
            tp_all += tp; fp_all += fp; fn_all += fn
            stats_per_target[t_id]["tp"] += tp
            stats_per_target[t_id]["fp"] += fp
            stats_per_target[t_id]["fn"] += fn
            
            p_f1 = get_f1_score(tp, fp, fn)

            # --- Vizualizare Colorată a Erorilor (Error Map) ---
            # RGB: Red (FN), Green (TP), Blue (FP)
            error_map = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
            
            # Verde - Ce a ghicit corect
            error_map[np.logical_and(pred_mask == 1, gt_mask == 1)] = [0, 255, 0] 
            # Roșu - Ce a ratat (False Negatives)
            error_map[np.logical_and(pred_mask == 0, gt_mask == 1)] = [255, 0, 0]
            # Albastru - Ce a inventat incorect (False Positives - AICI TREBUIE SA TE UITI DUPA CATETERE)
            error_map[np.logical_and(pred_mask == 1, gt_mask == 0)] = [0, 100, 255]
            
            img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            overlay = cv2.addWeighted(img_bgr, 0.5, error_map, 0.7, 0)

            # --- Plotting ---
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("X-Ray Original")
            axes[0].text(10, 30, f"{file_names[0]} ({target_name})", color="white", backgroundcolor="black")
            axes[0].axis("off")
            
            axes[1].imshow(gt_mask, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            
            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title(f"Prediction (F1: {p_f1:.4f})")
            axes[2].axis("off")
            
            axes[3].imshow(overlay)
            axes[3].set_title("Error Map (V:TP, R:FN, B:FP)")
            axes[3].axis("off")
            
            plt.tight_layout()
            save_name = f"{target_name}_F1_{p_f1:.4f}_{file_names[0]}"
            plt.savefig(os.path.join(plots_dir, save_name), dpi=100, bbox_inches='tight')
            plt.close(fig)

    # --- Calculare si Afisare Scoruri Finale ---
    f1_global = get_f1_score(tp_all, fp_all, fn_all)
    f1_rca = get_f1_score(stats_per_target[0]["tp"], stats_per_target[0]["fp"], stats_per_target[0]["fn"])
    f1_lca = get_f1_score(stats_per_target[1]["tp"], stats_per_target[1]["fp"], stats_per_target[1]["fn"])

    print("\n" + "="*60)
    print("=== REZULTATE FINALE (MODEL CATETERE) ===")
    print(f"F1 Global (Întregul set de test): {f1_global:.4f}")
    print(f"F1 RCA:                           {f1_rca:.4f}")
    print(f"F1 LCA:                           {f1_lca:.4f}")
    print("="*60 + "\n")
    print(f"[✅] Uită-te in folderul '{plots_dir}' pentru a vedea cum se descurcă cu cateterele (zonele albastre).")


if __name__ == "__main__":
    main()