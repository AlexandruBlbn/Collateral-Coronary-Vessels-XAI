import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import segmentation_models_pytorch as smp

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from engine.train_targeted_vessel_segmentation import (
    TargetedSyntaxSegmentationDataset,
    MultiTaskTargetedUNet,
    device,
    ensure_dir,
    _f1_iou_from_counts,
    _apply_target_postprocess,
    soft_cldice_loss
)

# ==========================================================
# 1. ARHITECTURĂ CUSTOM PENTRU REFINEMENT (5 Canale)
# ==========================================================
class GlobalRefinementUNet(nn.Module):
    def __init__(self, encoder_name="tu-efficientnetv2_s", in_channels=5):
        super().__init__()
        # in_channels = 4 (canale morfologice imagine) + 1 (probabilitatea modelului de bază)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None, # Corectat: 'None' in loc de 'False' pentru SMP
            in_channels=in_channels,
            classes=1
        )
        
    def forward(self, image, base_mask):
        # Concatenăm imaginea și masca de bază pentru a oferi "Auto-Context"
        x = torch.cat([image, base_mask], dim=1)
        return self.model(x)

# ==========================================================
# 2. PERTURBARE AGRESIVĂ PENTRU A EVITA IDENTITATEA
# ==========================================================
def perturb_base_mask_for_training(mask_prob: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Corrupt base probabilities so refiner learns to recover from realistic base-model errors."""
    out = mask_prob.clone()
    aug_cfg = cfg.get("training", {}).get("input_mask_augmentation", {})
    if not aug_cfg.get("enabled", True):
        return mask_prob

    # 1. Ruperi Morfologice & Pete artificiale
    if torch.rand(1).item() < 0.5:
        if torch.rand(1).item() < 0.5:
            out = -F.max_pool2d(-out, kernel_size=3, stride=1, padding=1) # Eroziune -> Rupe vasele
        else:
            out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)   # Dilatare -> Creează zgomot gros

    # 2. Dropout puternic (simulează porțiuni invizibile complet)
    drop_prob = float(aug_cfg.get("dropout_p", 0.25))
    if drop_prob > 0.0 and torch.rand(1).item() < 0.5:
        out = F.dropout(out, p=drop_prob, training=True)

    # 3. Zgomot Gaussian și Blur
    noise_std = float(aug_cfg.get("gaussian_std", 0.15))
    if noise_std > 0.0 and torch.rand(1).item() < 0.5:
        out = out + torch.randn_like(out) * noise_std

    if torch.rand(1).item() < 0.5:
        out = F.avg_pool2d(out, kernel_size=5, stride=1, padding=2)

    return out.clamp(0.0, 1.0)


def _mask_overlay(img_1ch: torch.Tensor, mask_1ch: torch.Tensor, color_rgb, alpha: float = 0.55) -> torch.Tensor:
    img3 = img_1ch.repeat(1, 3, 1, 1)
    m = (mask_1ch > 0).float()
    color = torch.tensor(color_rgb, dtype=img3.dtype, device=img3.device).view(1, 3, 1, 1)
    return torch.clamp(img3 * (1.0 - alpha * m) + color * (alpha * m), 0.0, 1.0)


def log_refinement_visuals(
    writer: SummaryWriter,
    base_model: nn.Module,
    ref_model: nn.Module,
    loader: DataLoader,
    postprocess_cfg: dict,
    step: int,
    split_name: str,
    max_samples: int = 4,
):
    """Log side-by-side panels to inspect whether refinement really corrects base errors."""
    ref_model.eval()
    base_model.eval()

    with torch.no_grad():
        picked = None
        for batch in loader:
            inputs, masks, aux_masks, centerlines, vectors, target_ids, _ = batch
            inputs = inputs.to(device)
            masks = masks.to(device)
            target_ids = target_ids.to(device)
            
            picked = (inputs, masks, target_ids)
            break

        if picked is None:
            return

        inputs, masks, target_ids = picked

        seg_logits_soft, cls_logits, seg_both, aux_logits, center_logits, vec_preds = base_model(inputs)
        idx = target_ids.long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both.shape[-2], seg_both.shape[-1])
        seg_logits_base = torch.gather(seg_both, dim=1, index=idx).squeeze(1)
        base_prob = torch.sigmoid(seg_logits_base)

        # ACUM Trimitem TOATE cele 4 canale!
        refined_logits = ref_model(inputs, base_prob)
        refined_prob = torch.sigmoid(refined_logits)

        bs = min(max_samples, inputs.shape[0])
        
        img = inputs[:bs, 0:1, :, :].detach().cpu() # Doar pentru desenare Tensorboard
        gt = masks[:bs].detach().cpu()
        base_p = base_prob[:bs].detach().cpu()
        ref_p = refined_prob[:bs].detach().cpu()

        base_bin = torch.zeros_like(base_p, dtype=torch.float32)
        ref_bin = torch.zeros_like(ref_p, dtype=torch.float32)
        thr = float(postprocess_cfg.get("threshold", 0.5))
        for i in range(bs):
            b_pred = (base_p[i, 0].numpy() >= thr).astype(np.uint8)
            b_pred = _apply_target_postprocess(b_pred, postprocess_cfg)
            base_bin[i, 0] = torch.from_numpy(b_pred.astype(np.float32))

            r_pred = (ref_p[i, 0].numpy() >= 0.5).astype(np.uint8)
            r_pred = _apply_target_postprocess(r_pred, postprocess_cfg)
            ref_bin[i, 0] = torch.from_numpy(r_pred.astype(np.float32))

        writer.add_images(f"Viz/{split_name}/Input", img.repeat(1, 3, 1, 1), step)
        writer.add_images(f"Viz/{split_name}/GT", gt.repeat(1, 3, 1, 1), step)
        writer.add_images(f"Viz/{split_name}/BaseProb", base_p.repeat(1, 3, 1, 1), step)
        writer.add_images(f"Viz/{split_name}/RefinedProb", ref_p.repeat(1, 3, 1, 1), step)
        writer.add_images(f"Viz/{split_name}/BasePred", base_bin.repeat(1, 3, 1, 1), step)
        writer.add_images(f"Viz/{split_name}/RefinedPred", ref_bin.repeat(1, 3, 1, 1), step)
        writer.add_images(
            f"Viz/{split_name}/OverlayGT",
            _mask_overlay(img, gt, color_rgb=[0.0, 1.0, 0.0]),
            step,
        )
        writer.add_images(
            f"Viz/{split_name}/OverlayBase",
            _mask_overlay(img, base_bin, color_rgb=[1.0, 0.2, 0.2]),
            step,
        )
        writer.add_images(
            f"Viz/{split_name}/OverlayRefined",
            _mask_overlay(img, ref_bin, color_rgb=[0.2, 0.2, 1.0]),
            step,
        )

def refinement_loss(pred_logits, true_mask, bce_w=0.4, tversky_w=0.4, cldice_w=0.2):
    bce = F.binary_cross_entropy_with_logits(pred_logits, true_mask)
    p = torch.sigmoid(pred_logits).flatten(1)
    t = true_mask.flatten(1)
    tp = (p * t).sum(dim=1)
    fp = (p * (1.0 - t)).sum(dim=1)
    fn = ((1.0 - p) * t).sum(dim=1)
    tversky = (tp + 1e-6) / (tp + 0.3 * fp + 0.7 * fn + 1e-6)
    tversky_loss = (1.0 - tversky).mean()
    cldice = soft_cldice_loss(torch.sigmoid(pred_logits), true_mask)
    return bce_w * bce + tversky_w * tversky_loss + cldice_w * cldice

def train_epoch(base_model, ref_model, loader, optimizer, config):
    ref_model.train()
    total_loss = 0.0
    pbar = tqdm(loader, total=len(loader), desc="Train Refinement")
    
    for inputs, masks, aux_masks, centerlines, vectors, target_ids, _ in pbar:
        inputs = inputs.to(device)
        masks = masks.to(device)
        target_ids = target_ids.to(device)
        
        with torch.no_grad():
            seg_logits_soft, cls_logits, seg_both, aux_logits, center_logits, vec_preds = base_model(inputs)
            
            idx = target_ids.long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both.shape[-2], seg_both.shape[-1])
            seg_logits_base = torch.gather(seg_both, dim=1, index=idx).squeeze(1)
            
            initial_masks_prob = torch.sigmoid(seg_logits_base)
            initial_masks_train = perturb_base_mask_for_training(initial_masks_prob, config)
            
        optimizer.zero_grad()
        
        # ACUM Trimitem TOATE cele 4 canale către refiner
        refined_logits = ref_model(inputs, initial_masks_train)
        
        loss = refinement_loss(
            refined_logits, 
            masks,
            bce_w=config["training"]["bce_weight"],
            tversky_w=config["training"]["tversky_weight"],
            cldice_w=config["training"]["cldice_weight"]
        )
        
        if not torch.isfinite(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ref_model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Ref_Loss": total_loss / max(1, pbar.n + 1)})
        
    return total_loss / max(1, len(loader))

def evaluate(base_model, ref_model, loader, postprocess_cfg):
    ref_model.eval()
    
    stats = {
        0: {"tp_base": 0.0, "fp_base": 0.0, "fn_base": 0.0, "tp_ref": 0.0, "fp_ref": 0.0, "fn_ref": 0.0},
        1: {"tp_base": 0.0, "fp_base": 0.0, "fn_base": 0.0, "tp_ref": 0.0, "fp_ref": 0.0, "fn_ref": 0.0}
    }
    
    pbar = tqdm(loader, total=len(loader), desc="Eval Refinement")
    with torch.no_grad():
        for inputs, masks, aux_masks, centerlines, vectors, target_ids, _ in pbar:
            inputs = inputs.to(device)
            masks = masks.to(device).int()
            target_ids = target_ids.to(device)
            
            seg_logits_soft, cls_logits, seg_both, aux_logits, center_logits, vec_preds = base_model(inputs)
            idx = target_ids.long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both.shape[-2], seg_both.shape[-1])
            seg_logits_base = torch.gather(seg_both, dim=1, index=idx).squeeze(1)
            initial_masks_prob = torch.sigmoid(seg_logits_base)
            
            # ACUM Trimitem TOATE cele 4 canale!
            refined_logits = ref_model(inputs, initial_masks_prob)
            refined_prob = torch.sigmoid(refined_logits)
            
            for b in range(inputs.size(0)):
                gt = masks[b, 0].cpu().numpy()
                t_id = int(target_ids[b].item())
                
                base_p = initial_masks_prob[b, 0].cpu().numpy()
                base_pred = (base_p >= postprocess_cfg.get("threshold", 0.5)).astype(np.uint8)
                base_pred = _apply_target_postprocess(base_pred, postprocess_cfg)
                
                stats[t_id]["tp_base"] += np.logical_and(base_pred == 1, gt == 1).sum()
                stats[t_id]["fp_base"] += np.logical_and(base_pred == 1, gt == 0).sum()
                stats[t_id]["fn_base"] += np.logical_and(base_pred == 0, gt == 1).sum()
                
                ref_p = refined_prob[b, 0].cpu().numpy()
                ref_pred = (ref_p >= 0.5).astype(np.uint8) # Threshold fixed at 0.5 for refinement
                ref_pred = _apply_target_postprocess(ref_pred, postprocess_cfg)
                
                stats[t_id]["tp_ref"] += np.logical_and(ref_pred == 1, gt == 1).sum()
                stats[t_id]["fp_ref"] += np.logical_and(ref_pred == 1, gt == 0).sum()
                stats[t_id]["fn_ref"] += np.logical_and(ref_pred == 0, gt == 1).sum()
                
    global_tp_base = stats[0]["tp_base"] + stats[1]["tp_base"]
    global_fp_base = stats[0]["fp_base"] + stats[1]["fp_base"]
    global_fn_base = stats[0]["fn_base"] + stats[1]["fn_base"]
    
    global_tp_ref = stats[0]["tp_ref"] + stats[1]["tp_ref"]
    global_fp_ref = stats[0]["fp_ref"] + stats[1]["fp_ref"]
    global_fn_ref = stats[0]["fn_ref"] + stats[1]["fn_ref"]
    
    base_f1_global, _ = _f1_iou_from_counts(global_tp_base, global_fp_base, global_fn_base)
    ref_f1_global, _ = _f1_iou_from_counts(global_tp_ref, global_fp_ref, global_fn_ref)
    
    base_f1_rca, _ = _f1_iou_from_counts(stats[0]["tp_base"], stats[0]["fp_base"], stats[0]["fn_base"])
    ref_f1_rca, _ = _f1_iou_from_counts(stats[0]["tp_ref"], stats[0]["fp_ref"], stats[0]["fn_ref"])
    base_f1_lca, _ = _f1_iou_from_counts(stats[1]["tp_base"], stats[1]["fp_base"], stats[1]["fn_base"])
    ref_f1_lca, _ = _f1_iou_from_counts(stats[1]["tp_ref"], stats[1]["fp_ref"], stats[1]["fn_ref"])
    
    return {
        "base_global": base_f1_global, "ref_global": ref_f1_global,
        "base_rca": base_f1_rca, "ref_rca": ref_f1_rca,
        "base_lca": base_f1_lca, "ref_lca": ref_f1_lca,
    }

def main():
    config = {
        "experiment_name": "syntax_global_refinement_512x512",
        "base_model_ckpt": "checkpoints/syntax_targeted_vessel_segmentation_SOTA_512x512/best_model.pth", 
        "data": {
            "target_csv": "results/arcade_patient_tables/patient_main_artery_targets.csv",
            "syntax_root": "data/ARCADE/Unprocessed/arcade/syntax",
            "img_size": 512,
            "batch_size": 4,
            "num_workers": 4,
        },
        "model": {
            "refinement_encoder": "tu-efficientnetv2_s", # Folosim un encoder puternic (același ca la baza)
        },
        "training": {
            "epochs": 100,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "bce_weight": 0.4,
            "tversky_weight": 0.4,
            "cldice_weight": 0.2,
            "input_mask_augmentation": {
                "enabled": True,
                "dropout_p": 0.25,
                "gaussian_std": 0.15,
            },
        },
        "base_postprocess": {
            "threshold": 0.45,
            "close_kernel": 3,
            "min_size": 20,
            "keep_largest": False
        },
        "logging": {
            "log_dir": "runs/{experiment_name}",
            "checkpoint_dir": "checkpoints/{experiment_name}",
            "visualize_every_epochs": 1,
            "visualize_num_samples": 4,
        },
    }

    log_dir = config["logging"]["log_dir"].format(experiment_name=config["experiment_name"])
    ckpt_dir = config["logging"]["checkpoint_dir"].format(experiment_name=config["experiment_name"])
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)

    # 1. Load Base Model
    print("Loading Base Model...")
    base_model = MultiTaskTargetedUNet(
        encoder_name="tu-efficientnetv2_s",
        encoder_weights=None,
        in_channels=4, # Actualizat la 4 canale morfologice
        classes=1,
        aux_num_classes=4,
    ).to(device)
    
    if os.path.exists(config["base_model_ckpt"]):
        base_model.load_state_dict(torch.load(config["base_model_ckpt"], map_location=device))
        print("Base model loaded successfully.")
    else:
        print(f"Warning: Could not find base model weights at {config['base_model_ckpt']}")
        
    base_model.eval() # Always evaluation mode for the base model during refinement training

    # 2. Setup Refinement Model
    print("Initializing Refinement Network...")
    ref_model = GlobalRefinementUNet(
        encoder_name=config["model"]["refinement_encoder"],
        in_channels=5 # 4 Canale Image + 1 Canal Mask Base
    ).to(device)

    # 3. Data Loaders
    train_ds = TargetedSyntaxSegmentationDataset(
        target_csv=config["data"]["target_csv"], syntax_root=config["data"]["syntax_root"], 
        split="train", img_size=config["data"]["img_size"], mode="train"
    )
    val_ds = TargetedSyntaxSegmentationDataset(
        target_csv=config["data"]["target_csv"], syntax_root=config["data"]["syntax_root"], 
        split="val", img_size=config["data"]["img_size"], mode="val"
    )

    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True, 
                              num_workers=config["data"]["num_workers"], persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False, 
                            num_workers=config["data"]["num_workers"], persistent_workers=True)

    # 4. Training objects
    optimizer = optim.AdamW(ref_model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    writer = SummaryWriter(log_dir=log_dir)

    best_val_f1 = -1.0
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    last_path = os.path.join(ckpt_dir, "last_model.pth")

    print("\nStarting Global Refinement Training (Error Correction for RCA & LCA)")
    for epoch in range(config["training"]["epochs"]):
        train_loss = train_epoch(base_model, ref_model, train_loader, optimizer, config)
        metrics = evaluate(base_model, ref_model, val_loader, config["base_postprocess"])
        
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Val/Base_Global_F1", metrics["base_global"], epoch)
        writer.add_scalar("Val/Refined_Global_F1", metrics["ref_global"], epoch)
        writer.add_scalar("Val/Global_Gain", metrics["ref_global"] - metrics["base_global"], epoch)
        
        writer.add_scalar("Val/Refined_RCA_F1", metrics["ref_rca"], epoch)
        writer.add_scalar("Val/Refined_LCA_F1", metrics["ref_lca"], epoch)

        vis_every = max(1, int(config["logging"].get("visualize_every_epochs", 1)))
        if ((epoch + 1) % vis_every) == 0 or epoch == 0:
            log_refinement_visuals(
                writer=writer,
                base_model=base_model,
                ref_model=ref_model,
                loader=val_loader,
                postprocess_cfg=config["base_postprocess"],
                step=epoch,
                split_name="val",
                max_samples=int(config["logging"].get("visualize_num_samples", 4)),
            )

        torch.save(ref_model.state_dict(), last_path)
        if metrics["ref_global"] > best_val_f1:
            best_val_f1 = metrics["ref_global"]
            torch.save(ref_model.state_dict(), best_path)

        print(f"Epoch {epoch + 1}/{config['training']['epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"Base F1: {metrics['base_global']:.4f} | "
              f"Refined F1: {metrics['ref_global']:.4f} "
              f"{'(New Best)' if metrics['ref_global'] == best_val_f1 else ''}")

if __name__ == "__main__":
    main()