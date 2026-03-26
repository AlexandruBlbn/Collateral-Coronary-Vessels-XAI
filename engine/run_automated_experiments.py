import os
import sys
import torch
import pandas as pd
import gc

# Adaugam root-ul proiectului in calea de executie pentru a putea importa modulul 'engine'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from engine.train_targeted_vessel_segmentation import MultiTaskTargetedUNet
from engine.train_targeted_vessel_segmentation import main as train_base
from engine.train_lca_refinement import main as train_refiner
from engine.test_with_postprocessing import main as test_model

TARGET_TRAIN_EPOCHS = 50

def sanity_check(backbones, decoders):
    print("="*50)
    print("=== START SANITY CHECK AUTOMAT ===")
    print("="*50)
    
    valid_combos = []
    for b in backbones:
        for d in decoders:
            print(f"Testare arhitectură: Encoder [{b}] + Decoder [{d}] ... ", end="")
            try:
                # Instanțiem modelul și setăm modul train pentru a activa Dropout/BatchNorm
                model = MultiTaskTargetedUNet(arch=d, encoder_name=b, in_channels=4, img_size=256).cuda()
                model.train()
                
                # Generăm date dummy conform input-ului
                dummy_in = torch.randn(2, 4, 256, 256).cuda()
                dummy_tgt = torch.tensor([0, 1]).cuda()
                
                # Forward + Backward Pass rapid
                out = model(dummy_in, target_ids=dummy_tgt, route_by_target=True)
                loss = out[0].sum()
                loss.backward()
                
                print("✅ SUCCES")
                valid_combos.append((b, d))
                
                # Curățare agresivă a VRAM-ului pentru a nu crasha la următoarele verificări
                del model, dummy_in, dummy_tgt, out, loss
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                # Găsim eroarea principală care indică de ce a picat (ex. stride incompatibil)
                error_msg = str(e).splitlines()[-1] if str(e).splitlines() else str(e)
                print(f"❌ EȘEC ({error_msg})")
                
    print(f"\n[INFO] {len(valid_combos)}/{len(backbones)*len(decoders)} combinații au trecut Sanity Check-ul.\n")
    return valid_combos

def get_completed_experiments(excel_path):
    """Citim Excel-ul pentru a sari peste experimentele deja finalizate (Checkpointing)"""
    if not os.path.exists(excel_path):
        return []
    try:
        df = pd.read_excel(excel_path)
        if "Backbone (Encoder)" in df.columns and "Decoder" in df.columns:
            return list(zip(df["Backbone (Encoder)"], df["Decoder"]))
    except Exception as e:
        print(f"[Avertisment] Eroare la citirea checkpoint-ului Excel: {e}")
    return []

def get_checkpoint_epochs(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"[Avertisment] Nu am putut citi checkpoint-ul {checkpoint_path}: {e}")
        return None

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        try:
            return int(checkpoint["epoch"]) + 1
        except (TypeError, ValueError):
            return None

    return None

def get_experiment_state(experiment_name):
    ckpt_dir = os.path.join("checkpoints", experiment_name)
    last_path = os.path.join(ckpt_dir, "last_model.pth")
    best_path = os.path.join(ckpt_dir, "best_model.pth")

    return {
        "ckpt_dir": ckpt_dir,
        "last_path": last_path,
        "best_path": best_path,
        "trained_epochs": get_checkpoint_epochs(last_path),
    }

def upsert_result(results, new_row):
    key = (new_row.get("Backbone (Encoder)"), new_row.get("Decoder"))
    for idx, row in enumerate(results):
        row_key = (row.get("Backbone (Encoder)"), row.get("Decoder"))
        if row_key == key:
            results[idx] = new_row
            return results

    results.append(new_row)
    return results

def main():
    # 1. Definim configurațiile dorite (inclusiv 'dcn' - Decodorul tău hibrid nativ)
    backbones = ["tu-resnet50", "tu-efficientnetv2_s", "tu-convnextv2_tiny"]
    decoders = ["unet", "manet", "fpn",  "deeplabv3plus"]
    
    # 2. Sanity Check
    valid_combos = sanity_check(backbones, decoders)
    
    excel_path = "rezultate/Grid_Search_Architectures.xlsx"
    completed_combos = get_completed_experiments(excel_path)
    
    if completed_combos:
        print(f"\n[INFO] Am gasit {len(completed_combos)} experimente deja finalizate. Acestea vor fi sarite (Checkpoint).")

    results = []
    # Incarcam rezultatele anterioare pentru a nu le pierde cand suprascriem fisierul
    if os.path.exists(excel_path):
        try:
            results = pd.read_excel(excel_path).to_dict('records')
        except:
            pass

    completed_combos = set(get_completed_experiments(excel_path))
    
    # 3. Bucla secvențială completă pentru combinațiile valide
    for idx, (b, d) in enumerate(valid_combos, 1):
        print("="*60)
        print(f"=== ANTRENAMENT COMBINAȚIA {idx}/{len(valid_combos)}: {b} + {d} ===")
        print("="*60)
        
        exp_base_name = f"GridSearch_{b.replace('tu-', '')}_{d}_Base"
        exp_ref_name = f"GridSearch_{b.replace('tu-', '')}_{d}_Refined"

        base_state = get_experiment_state(exp_base_name)
        ref_state = get_experiment_state(exp_ref_name)
        base_epochs = base_state["trained_epochs"]
        ref_epochs = ref_state["trained_epochs"]
        combo_logged = (b, d) in completed_combos
        base_complete = base_epochs is not None and base_epochs >= TARGET_TRAIN_EPOCHS
        ref_complete = ref_epochs is not None and ref_epochs >= TARGET_TRAIN_EPOCHS

        print(
            f"[INFO] Stare checkpoint Base: {base_epochs if base_epochs is not None else 'N/A'} epoci | "
            f"Refiner: {ref_epochs if ref_epochs is not None else 'N/A'} epoci"
        )

        if combo_logged and base_complete and ref_complete:
            print(f"\n>>> [CHECKPOINT AUTOMAT] Sarim combinatia {b} + {d} (deja evaluata si complet antrenata).")
            continue

        if base_epochs is not None and base_epochs < TARGET_TRAIN_EPOCHS:
            print(f"[RESUME] Base model are {base_epochs}/{TARGET_TRAIN_EPOCHS} epoci. Continuam antrenamentul.")
        elif base_complete:
            print(f"[INFO] Base model are deja cel putin {TARGET_TRAIN_EPOCHS} epoci. Va fi reutilizat.")
        else:
            print(f"[INFO] Nu exista checkpoint pentru base model. Pornim antrenarea pana la {TARGET_TRAIN_EPOCHS} epoci.")

        if ref_epochs is not None and ref_epochs < TARGET_TRAIN_EPOCHS:
            print(f"[RESUME] Refiner-ul are {ref_epochs}/{TARGET_TRAIN_EPOCHS} epoci. Continuam antrenamentul.")
        elif ref_complete:
            print(f"[INFO] Refiner-ul are deja cel putin {TARGET_TRAIN_EPOCHS} epoci. Va fi reutilizat.")
        else:
            print(f"[INFO] Nu exista checkpoint pentru refiner. Pornim antrenarea pana la {TARGET_TRAIN_EPOCHS} epoci.")
        
        print("\n>>> Pasul 1/3: Antrenare Base Model...")
        cfg_base = {
            "experiment_name": exp_base_name, 
            "model": {"arch": d, "encoder_name": b},
            "training": {"patience": 15, "epochs": TARGET_TRAIN_EPOCHS} # <-- EARLY STOPPING
        }
        base_ckpt_path, _ = train_base(config_override=cfg_base)
        
        print("\n>>> Pasul 2/3: Antrenare Global Refinement...")
        cfg_ref = {
            "experiment_name": exp_ref_name, 
            "base_model_ckpt": base_ckpt_path, 
            "model": {"base_arch": d, "base_encoder": b, "refinement_encoder": b},
            "training": {"patience": 15, "epochs": TARGET_TRAIN_EPOCHS} # <-- EARLY STOPPING
        }
        ref_ckpt_path, _ = train_refiner(config_override=cfg_ref)
        
        print("\n>>> Pasul 3/3: Testare și Post-Procesare Topologică...")
        test_args = {"arch": d, "encoder_name": b, "checkpoint": base_ckpt_path, "ref_checkpoint": ref_ckpt_path}
        metrics_dict = test_model(args_override=test_args)
        
        # Salvăm configurația în rândul de rezultate
        metrics_dict["Backbone (Encoder)"] = b
        metrics_dict["Decoder"] = d
        results = upsert_result(results, metrics_dict)
        
        # Exportăm tabelul la FIECARE iterație, astfel încât nu pierzi nimic dacă oprești scriptul la jumătate
        os.makedirs("rezultate", exist_ok=True)
        df = pd.DataFrame(results)
        
        # Ordonăm coloanele
        cols = ["Backbone (Encoder)", "Decoder", "F1 Refined Global", "F1 Base Global", "Mean Dice Refined (F1)", "F1 Refined RCA", "F1 Refined LCA"]
        df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
        
        # Highlighting cel mai bun rezultat F1
        styled_df = df.style.highlight_max(subset=['F1 Refined Global'], color='lightgreen')
        styled_df.to_excel(excel_path, index=False)
        print(f"\n[INFO] Tabelul Excel a fost actualizat la '{excel_path}'")

if __name__ == "__main__":
    main()