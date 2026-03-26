import os
import csv
import json
from pathlib import Path
from collections import defaultdict
import collections.abc

import numpy as np
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.utils as vutils

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import set_seed
from engine.train_targeted_vessel_segmentation import TargetedSyntaxSegmentationDataset # Reutilizam dataset-ul existent
from zoo.hibrid import AngioSegmenter # Noul model din hibrid.py

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ==========================================
# clDice (Topology-Aware Soft Skeleton Loss) - Copiat din train_targeted_vessel_segmentation.py
# ==========================================
def soft_erode(img):
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_=5):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

def soft_cldice_loss(pred, target, iter_=5):
    skel_pred = soft_skel(pred, iter_)
    skel_true = soft_skel(target, iter_)
    tprec = (torch.sum(skel_pred * target, dim=(2, 3)) + 1e-8) / (torch.sum(skel_pred, dim=(2, 3)) + 1e-8)
    tsens = (torch.sum(skel_true * pred, dim=(2, 3)) + 1e-8) / (torch.sum(skel_true, dim=(2, 3)) + 1e-8)
    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-8)
    return 1.0 - cl_dice.mean()

# ==========================================
# Loss Functions - Adaptate pentru AngioSegmenter
# ==========================================
def focal_tversky_loss(pred_logits, true_mask, alpha=0.3, beta=0.7, gamma=2.0, eps=1e-6):
    probs = torch.sigmoid(pred_logits).clamp(min=eps, max=1.0 - eps)
    p = probs.flatten(1)
    t = true_mask.flatten(1)
    
    tp = (p * t).sum(dim=1)
    fp = (p * (1.0 - t)).sum(dim=1)
    fn = ((1.0 - p) * t).sum(dim=1)
    
    tversky_idx = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    focal_tversky = torch.pow((1.0 - tversky_idx), gamma)
    return focal_tversky.mean()

def bce_loss(pred_logits, true_mask):
    return F.binary_cross_entropy_with_logits(pred_logits, true_mask)

class AngioSegmenterLoss(nn.Module):
    def __init__(self, tversky_weight=0.4, bce_weight=0.4, cldice_weight=0.2, 
                 tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=2.0):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.cldice_weight = cldice_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_gamma = tversky_gamma

    def forward(self, seg_logits, target_mask):
        seg_tversky = focal_tversky_loss(seg_logits, target_mask, 
                                         alpha=self.tversky_alpha, 
                                         beta=self.tversky_beta, 
                                         gamma=self.tversky_gamma)
        seg_bce = bce_loss(seg_logits, target_mask)
        seg_cldice = soft_cldice_loss(torch.sigmoid(seg_logits), target_mask)
        
        total_loss = (self.tversky_weight * seg_tversky +
                      self.bce_weight * seg_bce +
                      self.cldice_weight * seg_cldice)
        return total_loss, seg_tversky, seg_bce, seg_cldice


def train_epoch(model, loader, criterion, optimizer, accum_steps=4):
    model.train()
    total_loss = 0.0
    total_seg_tversky_loss = 0.0
    total_seg_bce_loss = 0.0
    total_seg_cldice_loss = 0.0
    pbar = tqdm(loader, total=len(loader), desc="Train")

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (inputs, masks, _, _, _, target_ids, _) in enumerate(pbar):
        inputs = inputs.to(device)
        masks = masks.to(device)

        seg_logits = model(inputs)
        loss, seg_tversky_loss, seg_bce_loss, seg_cldice_loss = criterion(seg_logits, masks)

        if not torch.isfinite(loss):
            continue

        (loss / accum_steps).backward()

        if ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_seg_tversky_loss += seg_tversky_loss.item()
        total_seg_bce_loss += seg_bce_loss.item()
        total_seg_cldice_loss += seg_cldice_loss.item()
        pbar.set_postfix(
            {
                "loss": total_loss / max(1, pbar.n + 1),
                "tversky": total_seg_tversky_loss / max(1, pbar.n + 1),
                "bce": total_seg_bce_loss / max(1, pbar.n + 1),
                "cldice": total_seg_cldice_loss / max(1, pbar.n + 1),
            }
        )

    n = max(1, len(loader))
    return total_loss / n, total_seg_tversky_loss / n, total_seg_bce_loss / n, total_seg_cldice_loss / n


def _f1_iou_from_counts(tp, fp, fn):
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    f1 = (2.0 * tp) / max(1e-8, (2.0 * tp + fp + fn))
    iou = tp / max(1e-8, (tp + fp + fn))
    return f1, iou


def _apply_target_postprocess(binary01: np.ndarray, cfg: dict) -> np.ndarray:
    out = (binary01 > 0).astype(np.uint8)

    k = int(cfg.get("close_kernel", 0))
    if k >= 3:
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    min_size = int(cfg.get("min_size", 0))
    keep_largest = bool(cfg.get("keep_largest", False))
    if min_size > 0 or keep_largest:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(out, dtype=np.uint8)

        if n_labels > 1:
            keep_ids = []
            for label_id in range(1, n_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area >= min_size:
                    keep_ids.append((label_id, area))

            if keep_largest and keep_ids:
                label_id = max(keep_ids, key=lambda x: x[1])[0]
                cleaned[labels == label_id] = 1
            else:
                for label_id, _ in keep_ids:
                    cleaned[labels == label_id] = 1

        out = cleaned

    return out


def _eval_postprocess_config(probs_list, masks_list, threshold: float, cfg: dict):
    tp = fp = fn = 0.0
    for p, m in zip(probs_list, masks_list):
        pred = (p >= threshold).astype(np.uint8)
        pred = _apply_target_postprocess(pred, cfg)
        tp += float(np.logical_and(pred == 1, m == 1).sum())
        fp += float(np.logical_and(pred == 1, m == 0).sum())
        fn += float(np.logical_and(pred == 0, m == 1).sum())
    return _f1_iou_from_counts(tp, fp, fn)


def evaluate(model, loader, criterion, threshold_by_target=None):
    model.eval()
    total_loss = 0.0
    total_seg_tversky_loss = 0.0
    total_seg_bce_loss = 0.0
    total_seg_cldice_loss = 0.0
    total_f1 = 0.0
    total_iou = 0.0
    tp_all = 0.0
    fp_all = 0.0
    fn_all = 0.0
    per_target = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n": 0})

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Eval")
        for idx, (inputs, masks, _, _, _, target_ids, _) in enumerate(pbar):
            inputs = inputs.to(device)
            masks = masks.to(device)

            seg_logits = model(inputs)
            loss, seg_tversky_loss, seg_bce_loss, seg_cldice_loss = criterion(seg_logits, masks)

            probs = torch.sigmoid(seg_logits)
            
            preds_np = np.zeros_like(probs.cpu().numpy(), dtype=np.uint8)
            probs_np = probs.cpu().numpy()
            
            # Gestionăm elegant formatul vechi (float 0.5) cu formatul nou de Grid Search (dicționar complet)
            cfg_rca = threshold_by_target.get(0, {"threshold": 0.5}) if isinstance(threshold_by_target.get(0), dict) else {"threshold": float((threshold_by_target or {}).get(0, 0.5))}
            cfg_lca = threshold_by_target.get(1, {"threshold": 0.5}) if isinstance(threshold_by_target.get(1), dict) else {"threshold": float((threshold_by_target or {}).get(1, 0.5))}

            for b in range(probs_np.shape[0]):
                t_id = int(target_ids[b].item())
                cfg = cfg_rca if t_id == 0 else cfg_lca
                thr = float(cfg.get("threshold", 0.5))
                
                pred_b = (probs_np[b, 0] >= thr).astype(np.uint8)
                pred_b = _apply_target_postprocess(pred_b, cfg)
                preds_np[b, 0] = pred_b

            preds = torch.tensor(preds_np, device=device, dtype=torch.int32)

            masks_i = masks.int()
            tp = ((preds == 1) & (masks_i == 1)).sum().item()
            fp = ((preds == 1) & (masks_i == 0)).sum().item()
            fn = ((preds == 0) & (masks_i == 1)).sum().item()
            tp_all += tp
            fp_all += fp
            fn_all += fn

            for t in (0, 1):
                sel = target_ids == t
                if sel.any():
                    p_t = preds[sel]
                    m_t = masks_i[sel]
                    tp_t = ((p_t == 1) & (m_t == 1)).sum().item()
                    fp_t = ((p_t == 1) & (m_t == 0)).sum().item()
                    fn_t = ((p_t == 0) & (m_t == 1)).sum().item()
                    per_target[t]["tp"] += tp_t
                    per_target[t]["fp"] += fp_t
                    per_target[t]["fn"] += fn_t
                    per_target[t]["n"] += int(sel.sum().item())

            f1, iou = _f1_iou_from_counts(tp, fp, fn)

            total_loss += loss.item()
            total_seg_tversky_loss += seg_tversky_loss.item()
            total_seg_bce_loss += seg_bce_loss.item()
            total_seg_cldice_loss += seg_cldice_loss.item()
            total_f1 += f1
            total_iou += iou

            pbar.set_postfix(
                {
                    "loss": total_loss / (idx + 1),
                    "f1": total_f1 / (idx + 1),
                    "iou": total_iou / (idx + 1),
                }
            )

    n = max(1, len(loader))
    overall_f1_exact, overall_iou_exact = _f1_iou_from_counts(tp_all, fp_all, fn_all)

    by_target = {}
    for t in (0, 1):
        stats = per_target[t]
        if stats["n"] > 0:
            f1_t, iou_t = _f1_iou_from_counts(stats["tp"], stats["fp"], stats["fn"])
        else:
            f1_t, iou_t = 0.0, 0.0
        by_target[t] = {
            "n": stats["n"],
            "f1": f1_t,
            "iou": iou_t,
        }

    return (
        total_loss / n,
        total_seg_tversky_loss / n,
        total_seg_bce_loss / n,
        total_seg_cldice_loss / n,
        total_f1 / n,
        total_iou / n,
        overall_f1_exact,
        overall_iou_exact,
        by_target,
    )


def find_best_thresholds_by_target(model, loader, search_cfg):
    model.eval()
    thresholds = [float(x) for x in search_cfg.get("threshold_grid", np.arange(0.1, 0.95, 0.05).tolist())]
    close_kernels = [int(x) for x in search_cfg.get("close_kernel_grid", [0, 3, 5])]
    min_sizes = [int(x) for x in search_cfg.get("min_size_grid", [0, 32, 64, 128])]
    keep_largest_opts = [bool(x) for x in search_cfg.get("keep_largest_grid", [False, True])]
    max_samples_per_target = int(search_cfg.get("max_samples_per_target", 0))

    cache = {0: {"probs": [], "masks": []}, 1: {"probs": [], "masks": []}}

    with torch.no_grad():
        for inputs, masks, _, _, _, target_ids, _ in tqdm(loader, total=len(loader), desc="Threshold search"):
            inputs = inputs.to(device)
            masks = masks.to(device).int()
            target_ids = target_ids.to(device)

            seg_logits = model(inputs)
            probs = torch.sigmoid(seg_logits)

            for b in range(probs.shape[0]):
                t_id = int(target_ids[b].item())
                if max_samples_per_target > 0 and len(cache[t_id]["probs"]) >= max_samples_per_target:
                    continue
                cache[t_id]["probs"].append(probs[b, 0].detach().cpu().numpy().astype(np.float32))
                cache[t_id]["masks"].append(masks[b, 0].detach().cpu().numpy().astype(np.uint8))

    best = {0: {"f1": -1.0}, 1: {"f1": -1.0}}
    for t_id in (0, 1):
        probs_list = cache[t_id]["probs"]
        masks_list = cache[t_id]["masks"]
        if not probs_list:
            best[t_id] = {
                "threshold": 0.5,
                "close_kernel": 0,
                "min_size": 0,
                "keep_largest": False,
                "f1": 0.0,
            }
            continue

        for t in thresholds:
            for k in close_kernels:
                for msz in min_sizes:
                    for keep_largest in keep_largest_opts:
                        cfg = {
                            "threshold": float(t),
                            "close_kernel": int(k),
                            "min_size": int(msz),
                            "keep_largest": bool(keep_largest),
                        }
                        f1, _ = _eval_postprocess_config(probs_list, masks_list, float(t), cfg)
                        if f1 > best[t_id]["f1"]:
                            best[t_id] = dict(cfg)
                            best[t_id]["f1"] = float(f1)

    return best


def log_tensorboard_visuals(
    writer: SummaryWriter,
    model: nn.Module,
    loader: DataLoader,
    threshold_by_target: dict,
    step: int,
    split_name: str,
    max_samples: int = 4,
):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        batch = next(iter(loader), None)
        if batch is None:
            if was_training:
                model.train()
            return

        inputs, masks, _, _, _, target_ids, file_names = batch
        inputs = inputs.to(device)
        masks = masks.to(device)
        target_ids = target_ids.to(device)

        seg_logits = model(inputs)
        probs = torch.sigmoid(seg_logits)
        
        # Sincronizare vizualuri Tensorboard cu post-procesarea 
        preds_np = np.zeros_like(probs.cpu().numpy(), dtype=np.uint8)
        probs_np = probs.cpu().numpy()
        cfg_rca = threshold_by_target.get(0, {"threshold": 0.5}) if isinstance(threshold_by_target.get(0), dict) else {"threshold": float((threshold_by_target or {}).get(0, 0.5))}
        cfg_lca = threshold_by_target.get(1, {"threshold": 0.5}) if isinstance(threshold_by_target.get(1), dict) else {"threshold": float((threshold_by_target or {}).get(1, 0.5))}

        for b in range(probs_np.shape[0]):
            t_id = int(target_ids[b].item())
            cfg = cfg_rca if t_id == 0 else cfg_lca
            thr = float(cfg.get("threshold", 0.5))
            pred_b = (probs_np[b, 0] >= thr).astype(np.uint8)
            pred_b = _apply_target_postprocess(pred_b, cfg)
            preds_np[b, 0] = pred_b

        preds = torch.tensor(preds_np, device=device, dtype=torch.int32)

        def _log_group(tag_suffix: str, sel: torch.Tensor):
            if sel.sum().item() == 0:
                return
            idx = torch.where(sel)[0][:max_samples]
            img = inputs[idx].detach().cpu()
            gt_bin = masks[idx].detach().cpu()
            pr_bin = preds[idx].float().detach().cpu()

            # Pentru vizualizare folosim doar primul canal (CLAHE) pentru ca Tensorboard cere 1 sau 3 canale
            img_vis = img[:, 0:1, :, :]

            # Lipim imaginea, masca reala si predictia pe orizontala (Data | Label | Pred)
            combined = torch.cat([img_vis.repeat(1, 3, 1, 1), gt_bin.repeat(1, 3, 1, 1), pr_bin.repeat(1, 3, 1, 1)], dim=3)
            writer.add_images(f"Viz/{split_name}/{tag_suffix}/Data_Label_Pred", combined, step)

        sel_all = torch.ones_like(target_ids, dtype=torch.bool)
        _log_group("Mixed", sel_all)
        _log_group("RCA", target_ids == 0)
        _log_group("LCA", target_ids == 1)

    if was_training:
        model.train()


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def main(config_override=None):
    config = {
        "experiment_name": "AngioSegmenter_Hibrid",
        "data": {
            "target_csv": "results/arcade_patient_tables/patient_main_artery_targets.csv",
            "syntax_root": "data/ARCADE/Unprocessed/arcade/syntax",
            "sample_weights_csv": "results/hard_case_mining/sample_weights_train.csv",
            "img_size": 512,
            "batch_size": 4,
            "num_workers": 4,
        },
        "model": {
            "in_chans": 4, # 4 canale de input (CLAHE, TopHat, BlackHat, Unsharp)
            "num_classes": 1,
            "dims": [32, 64, 128, 256], # Dimensiunile canalelor pentru encoder/decoder
            "depths": [2, 2, 2], # Numărul de blocuri ConvNextBlock per stage în encoder
            "drop_path_rate": 0.1,
        },
        "training": {
            "epochs": 200,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "tversky_alpha": 0.3,
            "tversky_beta": 0.7,
            "tversky_gamma": 2.0, # Focal Tversky agresiv
            "tversky_weight": 0.4,
            "bce_weight": 0.4,
            "cldice_weight": 0.2, # clDice pentru continuitate topologică
            "accum_steps": 4,
            "patience": 15, # Early stopping
        },
        "postprocess": {
            "rca_threshold": 0.5,
            "lca_threshold": 0.5,
        },
        "logging": {
            "log_dir": "runs/{experiment_name}",
            "checkpoint_dir": "checkpoints/{experiment_name}",
            "visualize_every_epochs": 2,
            "visualize_num_samples": 4,
        },
    }

    if config_override is not None:
        config = deep_update(config, config_override)

    log_dir = config["logging"]["log_dir"].format(experiment_name=config["experiment_name"])
    ckpt_dir = config["logging"]["checkpoint_dir"].format(experiment_name=config["experiment_name"])
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)

    train_ds = TargetedSyntaxSegmentationDataset(
        target_csv=config["data"]["target_csv"],
        syntax_root=config["data"]["syntax_root"],
        split="train",
        img_size=config["data"]["img_size"],
        mode="train",
    )
    val_ds = TargetedSyntaxSegmentationDataset(
        target_csv=config["data"]["target_csv"],
        syntax_root=config["data"]["syntax_root"],
        split="val",
        img_size=config["data"]["img_size"],
        mode="val",
    )
    test_ds = TargetedSyntaxSegmentationDataset(
        target_csv=config["data"]["target_csv"],
        syntax_root=config["data"]["syntax_root"],
        split="test",
        img_size=config["data"]["img_size"],
        mode="test",
    )

    weights_csv = config["data"].get("sample_weights_csv", "")
    if weights_csv and os.path.isfile(weights_csv):
        print(f"[INFO] Încărcăm ponderile (hard case weights) din {weights_csv}...")
        weight_dict = {}
        with open(weights_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                weight_dict[row['file_name']] = float(row['sample_weight'])
        
        sample_weights = [weight_dict.get(row['file_name'], 1.0) for row in train_ds.rows]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
    )

    model = AngioSegmenter(
        in_chans=config["model"]["in_chans"],
        num_classes=config["model"]["num_classes"],
        dims=config["model"]["dims"],
        depths=config["model"]["depths"],
        drop_path_rate=config["model"]["drop_path_rate"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    criterion = AngioSegmenterLoss(
        tversky_weight=config["training"]["tversky_weight"],
        bce_weight=config["training"]["bce_weight"],
        cldice_weight=config["training"]["cldice_weight"],
        tversky_alpha=config["training"]["tversky_alpha"],
        tversky_beta=config["training"]["tversky_beta"],
        tversky_gamma=config["training"]["tversky_gamma"],
    )

    writer = SummaryWriter(log_dir=log_dir)
    best_val_f1 = -1.0
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    last_path = os.path.join(ckpt_dir, "last_model.pth")

    patience = config["training"].get("patience", 0)
    epochs_no_improve = 0

    threshold_by_target = {
        0: {"threshold": float(config["postprocess"].get("rca_threshold", 0.5))},
        1: {"threshold": float(config["postprocess"].get("lca_threshold", 0.5))},
    }

    start_epoch = 0
    if os.path.exists(last_path):
        print(f"\n[INFO] Reluare antrenament din checkpoint-ul existent: {last_path}")
        ckpt = torch.load(last_path, map_location=device)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_f1 = ckpt.get("best_val_f1", -1.0)
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            print(f"[INFO] S-a reluat de la epoca {start_epoch} (Best F1 anterior: {best_val_f1:.4f})\n")
        else:
            model.load_state_dict(ckpt)
            print("[INFO] S-au încărcat doar ponderile (format vechi).\n")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        train_loss, train_tversky_loss, train_bce_loss, train_cldice_loss = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            accum_steps=config["training"].get("accum_steps", 4)
        )

        (
            val_loss,
            val_tversky_loss,
            val_bce_loss,
            val_cldice_loss,
            val_f1,
            val_iou,
            val_f1_exact,
            val_iou_exact,
            val_by_target,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            threshold_by_target=threshold_by_target,
        )
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Val/RCA_F1", val_by_target[0]["f1"], epoch)
        writer.add_scalar("Val/LCA_F1", val_by_target[1]["f1"], epoch)

        vis_every = max(1, int(config["logging"].get("visualize_every_epochs", 2)))
        if ((epoch + 1) % vis_every) == 0 or epoch == 0:
            log_tensorboard_visuals(
                writer=writer,
                model=model,
                loader=val_loader,
                threshold_by_target=threshold_by_target,
                step=epoch,
                split_name="val",
                max_samples=int(config["logging"].get("visualize_num_samples", 4)),
            )

        if val_f1_exact > best_val_f1: # Folosim F1 exact pentru salvarea best model
            best_val_f1 = val_f1_exact
            torch.save(model.state_dict(), best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "epochs_no_improve": epochs_no_improve
        }
        torch.save(checkpoint, last_path)

        thr_rca_print = threshold_by_target[0].get("threshold", 0.5) if isinstance(threshold_by_target[0], dict) else threshold_by_target[0]
        thr_lca_print = threshold_by_target[1].get("threshold", 0.5) if isinstance(threshold_by_target[1], dict) else threshold_by_target[1]

        patience_msg = f" | Patience: {epochs_no_improve}/{patience}" if patience > 0 else ""

        print(
            f"Epoch {epoch + 1}/{config['training']['epochs']} | "
            f"val_loss={val_loss:.4f} | "
            f"val_f1={val_f1_exact:.4f} | RCA_f1={val_by_target[0]['f1']:.4f} | LCA_f1={val_by_target[1]['f1']:.4f} | "
            f"thr(RCA/LCA)=({thr_rca_print:.2f}/{thr_lca_print:.2f}){patience_msg}"
        )
        
        if patience > 0 and epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING] Nicio imbunatatire timp de {patience} epoci. Se opreste antrenamentul curent.")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    
    print("\n--- Running Topology Grid Search on Validation Set ---")
    search_cfg = {
        "threshold_grid": [0.35, 0.40, 0.45, 0.50, 0.55],
        "close_kernel_grid": [0, 3, 5],
        "min_size_grid": [0, 20, 50, 100],
        "keep_largest_grid": [False]
    }
    best_postprocess_cfg = find_best_thresholds_by_target(model, val_loader, search_cfg)
    print(f"Optimal RCA Config: {best_postprocess_cfg[0]}")
    print(f"Optimal LCA Config: {best_postprocess_cfg[1]}")

    (
        test_loss,
        test_tversky_loss,
        test_bce_loss,
        test_cldice_loss,
        test_f1,
        test_iou,
        test_f1_exact,
        test_iou_exact,
        test_by_target,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        threshold_by_target=best_postprocess_cfg,
    )

    writer.add_scalar("Test/F1", test_f1_exact)
    writer.add_scalar("Test/RCA_F1", test_by_target[0]["f1"])
    writer.add_scalar("Test/LCA_F1", test_by_target[1]["f1"])
    
    log_tensorboard_visuals(
        writer=writer,
        model=model,
        loader=test_loader,
        threshold_by_target=best_postprocess_cfg,
        step=int(config["training"]["epochs"]),
        split_name="test",
        max_samples=int(config["logging"].get("visualize_num_samples", 4)),
    )
    writer.close()

    print("\n=== AngioSegmenter Test Results (best by val F1) ===")
    print(f"Test F1: {test_f1_exact:.4f} (IoU: {test_iou_exact:.4f})")
    print(
        f"RCA: n={test_by_target[0]['n']} | F1={test_by_target[0]['f1']:.4f}\n"
        f"LCA: n={test_by_target[1]['n']} | F1={test_by_target[1]['f1']:.4f}"
    )
    return best_path, test_f1_exact

if __name__ == "__main__":
    main()