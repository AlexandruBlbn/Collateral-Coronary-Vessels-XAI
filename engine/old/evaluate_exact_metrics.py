import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Adăugăm rădăcina proiectului în path pentru importuri
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from engine.train_targeted_vessel_segmentation import TargetedSyntaxSegmentationDataset, MultiTaskTargetedUNet, _apply_target_postprocess
from engine.train_lca_refinement import GlobalRefinementUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_exact_metrics(tp, fp, fn, tn):
    """Calculează metricile matematice exacte (fără aproximări vizuale)"""
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    tn = float(tn)
    
    # 1. Dice / F1 pentru clasa pozitivă (Vase de sânge)
    f1_vessel = (2.0 * tp) / max(1e-8, (2.0 * tp + fp + fn))
    
    # 2. Dice pentru clasa negativă (Background)
    f1_bg = (2.0 * tn) / max(1e-8, (2.0 * tn + fp + fn))
    
    # 3. Mean Dice
    mean_dice = (f1_vessel + f1_bg) / 2.0
    
    # 4. Alte metrici
    specificity = tn / max(1e-8, (tn + fp))
    accuracy = (tp + tn) / max(1e-8, (tp + tn + fp + fn))
    precision = tp / max(1e-8, (tp + fp))
    
    return f1_vessel, mean_dice, specificity, accuracy, precision

def main():
    print("="*60)
    print("=== EVALUARE EXACTĂ METRICI (BASE vs REFINED) ===")
    print("="*60)

    # Configurare Dataloader pentru Test Set
    test_ds = TargetedSyntaxSegmentationDataset(
        target_csv="results/arcade_patient_tables/patient_main_artery_targets.csv",
        syntax_root="data/ARCADE/Unprocessed/arcade/syntax",
        split="test",
        img_size=512,
        mode="test"
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4)

    # Post-Procesare Standard folosită la rafinare
    postprocess_cfg = {"threshold": 0.5, "close_kernel": 3, "min_size": 20, "keep_largest": False}

    backbones = ["tu-resnet50", "tu-efficientnetv2_s", "tu-convnextv2_tiny"]
    decoders = ["unet", "unetplusplus", "manet", "linknet", "fpn", "deeplabv3plus"]
    
    results_list = []

    for b in backbones:
        for d in decoders:
            exp_base_name = f"GridSearch_{b.replace('tu-', '')}_{d}_Base"
            exp_ref_name = f"GridSearch_{b.replace('tu-', '')}_{d}_Refined"
            
            base_ckpt = f"checkpoints/{exp_base_name}/best_model.pth"
            ref_ckpt = f"checkpoints/{exp_ref_name}/best_model.pth"
            
            # Procesăm doar combinațiile care au terminat cu succes antrenamentul
            if not os.path.exists(base_ckpt) or not os.path.exists(ref_ckpt):
                continue
                
            print(f"\n>>> Evaluare Model: {b} + {d}")
            
            # Încărcare Modele
            base_model = MultiTaskTargetedUNet(arch=d, encoder_name=b, in_channels=4, img_size=512).to(device)
            base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
            base_model.eval()
            
            ref_model = GlobalRefinementUNet(encoder_name=b, in_channels=5).to(device)
            ref_model.load_state_dict(torch.load(ref_ckpt, map_location=device))
            ref_model.eval()

            # Contori globali
            base_tp = base_fp = base_fn = base_tn = 0
            ref_tp = ref_fp = ref_fn = ref_tn = 0

            with torch.no_grad():
                for inputs, masks, _, _, _, target_ids, _ in tqdm(test_loader, desc=f"Testing {b}+{d}"):
                    inputs = inputs.to(device)
                    masks = masks.to(device).int()
                    target_ids = target_ids.to(device)
                    
                    # --- Predicție Base Model ---
                    _, _, seg_both, _, _, _ = base_model(inputs)
                    idx = target_ids.long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both.shape[-2], seg_both.shape[-1])
                    seg_logits_base = torch.gather(seg_both, dim=1, index=idx).squeeze(1)
                    base_probs = torch.sigmoid(seg_logits_base)
                    
                    # --- Predicție Refined Model ---
                    ref_logits = ref_model(inputs, base_probs)
                    ref_probs = torch.sigmoid(ref_logits)
                    
                    # Pixel-wise Counting
                    for b_idx in range(inputs.size(0)):
                        gt = masks[b_idx, 0].cpu().numpy()
                        
                        # Numărătoare Base
                        base_p = base_probs[b_idx, 0].cpu().numpy()
                        base_pred = (base_p >= 0.5).astype(np.uint8) # Fara postprocesare avansata la Base
                        
                        base_tp += np.logical_and(base_pred == 1, gt == 1).sum()
                        base_fp += np.logical_and(base_pred == 1, gt == 0).sum()
                        base_fn += np.logical_and(base_pred == 0, gt == 1).sum()
                        base_tn += np.logical_and(base_pred == 0, gt == 0).sum()
                        
                        # Numărătoare Refined (Cu post-procesare cascada)
                        ref_p = ref_probs[b_idx, 0].cpu().numpy()
                        ref_pred = (ref_p >= 0.5).astype(np.uint8)
                        ref_pred = _apply_target_postprocess(ref_pred, postprocess_cfg)
                        
                        ref_tp += np.logical_and(ref_pred == 1, gt == 1).sum()
                        ref_fp += np.logical_and(ref_pred == 1, gt == 0).sum()
                        ref_fn += np.logical_and(ref_pred == 0, gt == 1).sum()
                        ref_tn += np.logical_and(ref_pred == 0, gt == 0).sum()
            
            # Calculăm metricile exacte
            f1_base, _, _, _, _ = get_exact_metrics(base_tp, base_fp, base_fn, base_tn)
            f1_ref, mean_dice_ref, spec_ref, acc_ref, prec_ref = get_exact_metrics(ref_tp, ref_fp, ref_fn, ref_tn)
            
            results_list.append({
                "Backbone (Encoder)": b,
                "Decoder": d,
                "F1 Base (%)": round(f1_base * 100, 4),
                "F1 Refined (%)": round(f1_ref * 100, 4),
                "Mean Dice (%)": round(mean_dice_ref * 100, 4),
                "Specificity (%)": round(spec_ref * 100, 4),
                "Accuracy (%)": round(acc_ref * 100, 4),
                "Precision (%)": round(prec_ref * 100, 4)
            })

    if results_list:
        df = pd.DataFrame(results_list)
        os.makedirs("rezultate", exist_ok=True)
        excel_path = "rezultate/Tabel_Metrici_Exacte_Test.xlsx"
        
        # Colorăm automat în verde modelul cu performanța cea mai bună
        styled_df = df.style.highlight_max(subset=['Mean Dice (%)'], color='lightgreen')
        styled_df.to_excel(excel_path, index=False)
        print(f"\n✅ Evaluarea s-a terminat! Tabelul a fost salvat in: {excel_path}")
    else:
        print("\n[INFO] Nu am gasit checkpoint-uri antrenate pentru a le evalua.")

if __name__ == "__main__":
    main()