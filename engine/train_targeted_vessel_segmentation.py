import os
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

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
import segmentation_models_pytorch as smp
import timm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import set_seed
from engine.train_hybrid_sota import DCN_UNetDecoder, generate_vector_field_and_centerline, vector_direction_loss


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


RCA_LABELS = {"1", "2", "3", "4", "16", "16a", "16b", "16c"}
LCX_LABELS = {"11", "12", "13", "14", "14a", "14b", "15"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_split_index(syntax_root: str, split: str):
    ann_path = Path(syntax_root) / split / "annotations" / f"{split}.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}

    images = {
        int(img["id"]): {
            "file_name": img["file_name"],
            "width": int(img.get("width", 512)),
            "height": int(img.get("height", 512)),
            "rca_polygons": [],
            "lad_polygons": [],
            "lcx_polygons": [],
            "lca_polygons": [],
        }
        for img in coco.get("images", [])
    }

    for ann in coco.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        if image_id not in images:
            continue

        cat_name = cat_id_to_name.get(int(ann.get("category_id", -1)), "")
        seg = ann.get("segmentation", [])
        if not isinstance(seg, list):
            continue

        if cat_name == "stenosis" or cat_name == "":
            continue

        if cat_name in RCA_LABELS:
            target_key = "rca_polygons"
        elif cat_name in LCX_LABELS:
            target_key = "lcx_polygons"
        else:
            target_key = "lad_polygons"

        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                images[image_id][target_key].append(poly)

    for image_id in images:
        images[image_id]["lca_polygons"] = (
            images[image_id]["lad_polygons"] + images[image_id]["lcx_polygons"]
        )

    by_file_name = {v["file_name"]: v for v in images.values()}
    return by_file_name


def rasterize_polygons(width: int, height: int, polygons):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        points = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
        draw.polygon(points, fill=255)
    return mask


class TargetedSyntaxSegmentationDataset(Dataset):
    def __init__(self, target_csv: str, syntax_root: str, split: str, img_size: int = 256, mode: str = "train"):
        self.mode = mode
        self.img_size = img_size
        self.syntax_root = Path(syntax_root)

        # Inițializăm uneltele pentru extragerea canalelor
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.cache = {} # Stocam imaginile pre-procesate in RAM

        split_index = load_split_index(syntax_root, split)

        self.rows = []
        with open(target_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue

                target_id = int(row["target_main_artery_id"])
                if target_id not in (0, 1):
                    continue

                file_name = row["file_name"]
                if file_name not in split_index:
                    continue

                image_meta = split_index[file_name]
                image_path = self.syntax_root / split / "images" / file_name
                if not image_path.is_file():
                    continue

                polygons = image_meta["rca_polygons"] if target_id == 0 else image_meta["lca_polygons"]
                self.rows.append(
                    {
                        "image_path": str(image_path),
                        "file_name": file_name,
                        "patient_number": row.get("patient_number", ""),
                        "target_id": target_id,
                        "width": image_meta["width"],
                        "height": image_meta["height"],
                        "polygons": polygons,
                        "rca_polygons": image_meta["rca_polygons"],
                        "lad_polygons": image_meta["lad_polygons"],
                        "lcx_polygons": image_meta["lcx_polygons"],
                    }
                )

        if not self.rows:
            raise RuntimeError(f"No samples found for split={split}.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        
        # --- CACHING IN MEMORIE PENTRU ELIMINAREA BOTTLENECK-ULUI CPU ---
        if idx in self.cache:
            cached = self.cache[idx]
            stacked = cached["stacked"]
            mask_np = cached["mask"]
            mask_rca_np = cached["mask_rca"]
            mask_lad_np = cached["mask_lad"]
            mask_lcx_np = cached["mask_lcx"]
            centerline_np = cached["centerline"]
            vector_np = cached["vector"]
        else:
            image = Image.open(row["image_path"]).convert("L")
            mask_pil = rasterize_polygons(row["width"], row["height"], row["polygons"])
            mask_rca_pil = rasterize_polygons(row["width"], row["height"], row["rca_polygons"])
            mask_lad_pil = rasterize_polygons(row["width"], row["height"], row["lad_polygons"])
            mask_lcx_pil = rasterize_polygons(row["width"], row["height"], row["lcx_polygons"])

            image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
            mask_np = np.array(mask_pil.resize((self.img_size, self.img_size), resample=Image.NEAREST), dtype=np.uint8)
            mask_rca_np = np.array(mask_rca_pil.resize((self.img_size, self.img_size), resample=Image.NEAREST), dtype=np.uint8)
            mask_lad_np = np.array(mask_lad_pil.resize((self.img_size, self.img_size), resample=Image.NEAREST), dtype=np.uint8)
            mask_lcx_np = np.array(mask_lcx_pil.resize((self.img_size, self.img_size), resample=Image.NEAREST), dtype=np.uint8)

            image_np = np.array(image, dtype=np.uint8)
            c1 = self.clahe.apply(image_np)
            c2 = cv2.morphologyEx(image_np, cv2.MORPH_TOPHAT, self.morph_kernel)
            c3 = cv2.morphologyEx(image_np, cv2.MORPH_BLACKHAT, self.morph_kernel)
            blurred = cv2.GaussianBlur(image_np, (0, 0), sigmaX=10)
            c4 = cv2.addWeighted(image_np, 4, blurred, -4, 128)
            stacked = np.stack([c1, c2, c3, c4], axis=-1)
            
            # --- CALCULAM VECTORII O SINGURA DATA AICI PENTRU A SALVA CPU-UL ---
            mask_bool = (mask_np > 0).astype(np.uint8)
            vector_np, centerline_np = generate_vector_field_and_centerline(mask_bool)
            
            # Salvam in RAM sub forma de numpy arrays pentru a nu incarca procesorul
            self.cache[idx] = {"stacked": stacked, "mask": mask_np, "mask_rca": mask_rca_np, "mask_lad": mask_lad_np, "mask_lcx": mask_lcx_np, "centerline": centerline_np, "vector": vector_np}

        # Extragem ca tensori independenți pentru a permite Augmentarea instantanee pe GPU/CPU la nivel de batch
        img_t = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()
        mask_rca = torch.from_numpy(mask_rca_np).unsqueeze(0).float()
        mask_lad = torch.from_numpy(mask_lad_np).unsqueeze(0).float()
        mask_lcx = torch.from_numpy(mask_lcx_np).unsqueeze(0).float()
        centerline_t = torch.from_numpy(centerline_np)
        vector_t = torch.from_numpy(vector_np)

        if self.mode == "train":
            if torch.rand(1).item() > 0.3:
                angle = float(torch.empty(1).uniform_(-25.0, 25.0).item())
                scale = float(torch.empty(1).uniform_(0.8, 1.2).item())
                tx = int(torch.empty(1).uniform_(-0.1, 0.1).item() * self.img_size)
                ty = int(torch.empty(1).uniform_(-0.1, 0.1).item() * self.img_size)
                
                img_t = TF.affine(img_t, angle=angle, translate=[tx, ty], scale=scale, shear=0.0, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.affine(mask, angle=angle, translate=[tx, ty], scale=scale, shear=0.0, interpolation=TF.InterpolationMode.NEAREST)
                mask_rca = TF.affine(mask_rca, angle=angle, translate=[tx, ty], scale=scale, shear=0.0, interpolation=TF.InterpolationMode.NEAREST)
                mask_lad = TF.affine(mask_lad, angle=angle, translate=[tx, ty], scale=scale, shear=0.0, interpolation=TF.InterpolationMode.NEAREST)
                mask_lcx = TF.affine(mask_lcx, angle=angle, translate=[tx, ty], scale=scale, shear=0.0, interpolation=TF.InterpolationMode.NEAREST)
                
                centerline_t = TF.affine(centerline_t, angle=angle, translate=[tx, ty], scale=scale, shear=0.0, interpolation=TF.InterpolationMode.NEAREST)
                vector_t = TF.affine(vector_t, angle=angle, translate=[tx, ty], scale=scale, shear=0.0, interpolation=TF.InterpolationMode.NEAREST)
                
                # Rotim direcția vectorilor topologici confom rotației imaginii (Matematică pe coordonate de imagine unde Y e în jos)
                rad = math.radians(angle)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                
                dx = vector_t[0, :, :]
                dy = vector_t[1, :, :]
                new_dx = dx * cos_a + dy * sin_a
                new_dy = -dx * sin_a + dy * cos_a
                
                vector_t = torch.stack([new_dx, new_dy], dim=0)
                vector_t = vector_t * mask.squeeze(0)

            if torch.rand(1).item() > 0.5:
                brightness_factor = float(torch.empty(1).uniform_(0.8, 1.2).item())
                img_t = torch.clamp(img_t * brightness_factor, 0.0, 1.0)
            if torch.rand(1).item() > 0.5:
                gamma = float(torch.empty(1).uniform_(0.7, 1.3).item())
                img_t = torch.pow(img_t, gamma)

        msk_t = (mask > 0).float()

        # Reconstruim harta auxiliara rapid, din tensor
        aux_t = torch.zeros((self.img_size, self.img_size), dtype=torch.long)
        aux_t[mask_rca.squeeze(0) > 0] = 1
        aux_t[mask_lad.squeeze(0) > 0] = 2
        aux_t[mask_lcx.squeeze(0) > 0] = 3


        return img_t, msk_t, aux_t, centerline_t, vector_t, row["target_id"], row["file_name"]


def train_epoch(model, loader, criterion, optimizer, accum_steps=4):
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_aux_loss = 0.0
    total_ctr_loss = 0.0
    total_vec_loss = 0.0
    pbar = tqdm(loader, total=len(loader), desc="Train")

    # Curatam gradientii inainte de inceperea buclei pentru acumulare curata
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (inputs, masks, aux_masks, centerlines, vectors, target_ids, _) in enumerate(pbar):
        inputs = inputs.to(device)
        masks = masks.to(device)
        aux_masks = aux_masks.to(device)
        centerlines = centerlines.to(device)
        vectors = vectors.to(device)
        target_ids = target_ids.to(device)

        seg_logits, cls_logits, _, aux_logits, center_logits, vec_preds = model(inputs, target_ids=target_ids, route_by_target=True)
        loss, seg_loss, cls_loss, aux_loss, center_loss, vec_loss = criterion(
            seg_logits, cls_logits, aux_logits, center_logits, vec_preds, 
            masks, aux_masks, centerlines, vectors, target_ids
        )

        if not torch.isfinite(loss):
            continue

        # Impartim loss-ul la numarul de pasi pentru a normaliza gradientii acumulati
        (loss / accum_steps).backward()

        # Facem update-ul optimizatorului doar cand am strans 'accum_steps' batch-uri
        if ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() # Pastram valoarea nenormalizata pt afisaj
        total_seg_loss += seg_loss.item()
        total_cls_loss += cls_loss.item()
        total_aux_loss += aux_loss.item()
        total_ctr_loss += center_loss.item()
        total_vec_loss += vec_loss.item()
        pbar.set_postfix(
            {
                "loss": total_loss / max(1, pbar.n + 1),
                "seg": total_seg_loss / max(1, pbar.n + 1),
                "cls": total_cls_loss / max(1, pbar.n + 1),
                "aux": total_aux_loss / max(1, pbar.n + 1),
                "ctr": total_ctr_loss / max(1, pbar.n + 1),
                "vec": total_vec_loss / max(1, pbar.n + 1),
            }
        )

    n = max(1, len(loader))
    return total_loss / n, total_seg_loss / n, total_cls_loss / n, total_aux_loss / n, total_ctr_loss / n, total_vec_loss / n


def _f1_iou_from_counts(tp, fp, fn):
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    f1 = (2.0 * tp) / max(1e-8, (2.0 * tp + fp + fn))
    iou = tp / max(1e-8, (tp + fp + fn))
    return f1, iou


def _colorize_aux_map(aux_map: torch.Tensor) -> torch.Tensor:
    palette = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=aux_map.device,
    )
    idx = aux_map.long().clamp(min=0, max=3)
    out = palette[idx]  
    return out.permute(0, 3, 1, 2).contiguous()


def _mask_overlay(img_1ch: torch.Tensor, mask: torch.Tensor, color_rgb, alpha=0.55) -> torch.Tensor:
    img3 = img_1ch.repeat(1, 3, 1, 1)
    if mask.ndim == 4:
        m = (mask > 0).float()
    else:
        m = (mask.unsqueeze(1) > 0).float()
    color = torch.tensor(color_rgb, dtype=img3.dtype, device=img3.device).view(1, 3, 1, 1)
    return torch.clamp(img3 * (1.0 - alpha * m) + color * (alpha * m), 0.0, 1.0)


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


def evaluate(model, loader, criterion, threshold_by_target=None, routing_mode="hard_pred"):
    model.eval()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_aux_loss = 0.0
    total_ctr_loss = 0.0
    total_vec_loss = 0.0
    total_f1 = 0.0
    total_iou = 0.0
    tp_all = 0.0
    fp_all = 0.0
    fn_all = 0.0
    per_target = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n": 0})
    cls_correct = 0
    cls_total = 0
    aux_correct = 0
    aux_total = 0

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Eval")
        for idx, (inputs, masks, aux_masks, centerlines, vectors, target_ids, _) in enumerate(pbar):
            inputs = inputs.to(device)
            masks = masks.to(device)
            aux_masks = aux_masks.to(device)
            centerlines = centerlines.to(device)
            vectors = vectors.to(device)
            target_ids = target_ids.to(device)

            seg_logits_soft, cls_logits, seg_both, aux_logits, center_logits, vec_preds = model(inputs)

            idx_tgt = target_ids.long().view(-1, 1, 1, 1, 1).expand(
                -1, 1, 1, seg_both.shape[-2], seg_both.shape[-1]
            )
            seg_logits_oracle = torch.gather(seg_both, dim=1, index=idx_tgt).squeeze(1)
            loss, seg_loss, cls_loss, aux_loss, center_loss, vec_loss = criterion(
                seg_logits_oracle, cls_logits, aux_logits, center_logits, vec_preds,
                masks, aux_masks, centerlines, vectors, target_ids
            )

            if routing_mode == "hard_pred":
                pred_ids = torch.argmax(cls_logits, dim=1)
                idx_pred = pred_ids.long().view(-1, 1, 1, 1, 1).expand(
                    -1, 1, 1, seg_both.shape[-2], seg_both.shape[-1]
                )
                seg_logits_eval = torch.gather(seg_both, dim=1, index=idx_pred).squeeze(1)
            elif routing_mode == "oracle_target":
                seg_logits_eval = seg_logits_oracle
            else:
                seg_logits_eval = seg_logits_soft

            probs = torch.sigmoid(seg_logits_eval)

            # [FIX INGINERESC 2: Activarea Post-Procesării Topologice în fluxul de Evaluare]
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
            total_seg_loss += seg_loss.item()
            total_cls_loss += cls_loss.item()
            total_aux_loss += aux_loss.item()
            total_ctr_loss += center_loss.item()
            total_vec_loss += vec_loss.item()
            total_f1 += f1
            total_iou += iou

            cls_pred = torch.argmax(cls_logits, dim=1)
            cls_correct += (cls_pred == target_ids).sum().item()
            cls_total += target_ids.numel()

            aux_pred = torch.argmax(aux_logits, dim=1)
            aux_correct += (aux_pred == aux_masks).sum().item()
            aux_total += aux_masks.numel()

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

    cls_acc = cls_correct / max(1, cls_total)
    aux_acc = aux_correct / max(1, aux_total)
    return (
        total_loss / n,
        total_seg_loss / n,
        total_cls_loss / n,
        total_aux_loss / n,
        total_ctr_loss / n,
        total_vec_loss / n,
        total_f1 / n,
        total_iou / n,
        overall_f1_exact,
        overall_iou_exact,
        by_target,
        cls_acc,
        aux_acc,
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

            seg_logits, _, _, _, _, _ = model(inputs)
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


class MultiTaskTargetedUNet(nn.Module):
    def __init__(self, encoder_name="tu-efficientnetv2_s", encoder_weights=None, in_channels=4, classes=1, aux_num_classes=4):
        super().__init__()
        # 1. Extractor de Trăsături (Encoder Timm)
        timm_model_name = encoder_name.replace("tu-", "") if encoder_name.startswith("tu-") else encoder_name
        self.encoder = timm.create_model(timm_model_name, pretrained=False, in_chans=in_channels, features_only=True)
        enc_channels = self.encoder.feature_info.channels()
        
        # 2. Decodor bazat pe Deformable Convolutions SOTA
        self.decoder = DCN_UNetDecoder(encoder_channels=enc_channels[::-1])
        dec_ch = self.decoder.out_channels
        
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(enc_channels[-1], 2)
        
        # 3. Capete de Segmentare Multi-Task
        self.seg_head_rca = nn.Conv2d(dec_ch, classes, kernel_size=3, padding=1)
        self.seg_head_lca = nn.Conv2d(dec_ch, classes, kernel_size=3, padding=1)
        self.aux_head_vessel = nn.Conv2d(dec_ch, aux_num_classes, kernel_size=3, padding=1)
        
        # 4. Capete Topologice SOTA (Continuity)
        self.centerline_head = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.vector_head = nn.Sequential(
            nn.Conv2d(dec_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x, target_ids=None, route_by_target=False):
        orig_size = x.shape[2:]
        feats = self.encoder(x)
        decoder_out = self.decoder(feats)
        
        if decoder_out.shape[2:] != orig_size:
            decoder_out = F.interpolate(decoder_out, size=orig_size, mode='bilinear', align_corners=False)

        seg_rca = self.seg_head_rca(decoder_out)
        seg_lca = self.seg_head_lca(decoder_out)
        seg_both = torch.stack([seg_rca, seg_lca], dim=1) 

        cls_logits = self.cls_head(self.cls_pool(feats[-1]).flatten(1))

        if route_by_target and target_ids is not None:
            idx = target_ids.long().view(-1, 1, 1, 1, 1).expand(
                -1, 1, 1, seg_both.shape[-2], seg_both.shape[-1]
            )
            seg_logits = torch.gather(seg_both, dim=1, index=idx).squeeze(1)
        else:
            cls_probs = torch.softmax(cls_logits, dim=1).view(-1, 2, 1, 1, 1)
            seg_logits = (seg_both * cls_probs).sum(dim=1)

        aux_logits = self.aux_head_vessel(decoder_out)
        centerline_logits = self.centerline_head(decoder_out)
        vector_field = self.vector_head(decoder_out)
        
        return seg_logits, cls_logits, seg_both, aux_logits, centerline_logits, vector_field
        
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.cls_pool(feats[-1]).flatten(1)


# ==========================================
# clDice (Topology-Aware Soft Skeleton Loss)
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


def log_tensorboard_visuals(
    writer: SummaryWriter,
    model: nn.Module,
    loader: DataLoader,
    threshold_by_target: dict,
    routing_mode: str,
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

        inputs, masks, aux_masks, centerlines, vectors, target_ids, file_names = batch
        inputs = inputs.to(device)
        masks = masks.to(device)
        aux_masks = aux_masks.to(device)
        target_ids = target_ids.to(device)

        seg_logits_soft, cls_logits, seg_both, aux_logits, _, _ = model(inputs)

        pred_ids = torch.argmax(cls_logits, dim=1)

        if routing_mode == "hard_pred":
            idx_pred = pred_ids.long().view(-1, 1, 1, 1, 1).expand(
                -1, 1, 1, seg_both.shape[-2], seg_both.shape[-1]
            )
            seg_logits_eval = torch.gather(seg_both, dim=1, index=idx_pred).squeeze(1)
        elif routing_mode == "oracle_target":
            idx_tgt = target_ids.long().view(-1, 1, 1, 1, 1).expand(
                -1, 1, 1, seg_both.shape[-2], seg_both.shape[-1]
            )
            seg_logits_eval = torch.gather(seg_both, dim=1, index=idx_tgt).squeeze(1)
        else:
            seg_logits_eval = seg_logits_soft

        probs = torch.sigmoid(seg_logits_eval)
        
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


def main():
    config = {
        "experiment_name": "syntax_targeted_vessel_segmentation_SOTA_512x512",
        "data": {
            "target_csv": "results/arcade_patient_tables/patient_main_artery_targets.csv",
            "syntax_root": "data/ARCADE/Unprocessed/arcade/syntax",
            "sample_weights_csv": "results/hard_case_mining/sample_weights_train.csv",  # Fisierul generat de scriptul tau de mining
            "img_size": 512, # Dublam rezolutia pt a salva capilarele
            "batch_size": 16, # Reducem pt a evita OOM (Out Of Memory)
            "num_workers": 4,
        },
        "model": {
            "arch": "unetplusplus",
            "encoder_name": "tu-efficientnetv2_s",
            "encoder_weights": None,
            "in_channels": 4,
            "classes": 1,
            "aux_num_classes": 4,
            # [FIX INGINERESC 3: Schimbare în Soft Routing]
            # Permite modelului să folosească incertitudinea și să împrumute trăsături.
            "inference_routing": "soft",
        },
        "training": {
            "epochs": 200,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "rca_tversky_alpha": 0.3,
            "rca_tversky_beta": 0.7,
            "lca_tversky_alpha": 0.3,
            "lca_tversky_beta": 0.7,
            "tversky_weight": 0.4,
            "bce_weight": 0.4,
            "cldice_weight": 0.2, # Adaugam clDice in balanta
            "cls_weight": 0.3,
            "aux_weight": 0.35,
            "accum_steps": 1, # Simuleaza un batch size de 4x4 = 16
            "aux_class_weights": [0.25, 1.0, 1.6, 2.2],
            "rca_loss_weight": 1.0,
            "lca_loss_weight": 1.5,
            "center_weight": 0.5,
            "vector_weight": 0.5,
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

    # --- Hard Case Mining: Ponderare / Oversampling ---
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
        pin_memory=True, # Accelerator critic pentru placile video extrem de rapide
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        persistent_workers=(config["data"]["num_workers"] > 0),
        pin_memory=True,
    )

    model = MultiTaskTargetedUNet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"],
        aux_num_classes=config["model"]["aux_num_classes"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    cls_loss_fn = nn.CrossEntropyLoss()
    aux_cls_w = torch.tensor(config["training"]["aux_class_weights"], dtype=torch.float32, device=device)
    aux_ce_loss_fn = nn.CrossEntropyLoss(weight=aux_cls_w)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # [FIX INGINERESC 4: Focal Tversky Loss]
    # Forțăm gradientul să ignore stenozele "ușoare" și să se concentreze pe ramurile fine ale LCA-ului 
    def target_aware_focal_tversky(seg_logits, target, target_ids, eps=1e-6, gamma=1.33):
        probs = torch.sigmoid(seg_logits).clamp(min=eps, max=1.0 - eps)
        b = probs.shape[0]
        p = probs.view(b, -1)
        t = target.view(b, -1)

        tp = (p * t).sum(dim=1)
        fp = (p * (1.0 - t)).sum(dim=1)
        fn = ((1.0 - p) * t).sum(dim=1)

        alpha = torch.where(
            target_ids == 0,
            torch.full_like(target_ids, float(config["training"]["rca_tversky_alpha"]), dtype=torch.float32),
            torch.full_like(target_ids, float(config["training"]["lca_tversky_alpha"]), dtype=torch.float32),
        ).to(device)
        beta = torch.where(
            target_ids == 0,
            torch.full_like(target_ids, float(config["training"]["rca_tversky_beta"]), dtype=torch.float32),
            torch.full_like(target_ids, float(config["training"]["lca_tversky_beta"]), dtype=torch.float32),
        ).to(device)

        tversky_idx = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        # Aplicăm exponențiala gamma pentru a-l transforma în FOCAL Tversky
        focal_tversky = torch.pow((1.0 - tversky_idx), gamma)
        return focal_tversky.mean()

    def criterion(seg_logits, cls_logits, aux_logits, center_logits, vec_preds, target_mask, aux_target_mask, true_center, true_vec, target_ids):
        # Chemăm noua funcție Focal
        seg_tversky = target_aware_focal_tversky(seg_logits, target_mask, target_ids)  
        seg_bce = bce_loss_fn(seg_logits, target_mask).flatten(1).mean(dim=1).mean()  
        
        # Extragem scheletul probabilistic pentru a pedepsi ruperile vaselor de sange
        seg_cldice = soft_cldice_loss(torch.sigmoid(seg_logits), target_mask)
        
        seg_per_sample = (
            config["training"]["tversky_weight"] * seg_tversky
            + config["training"]["bce_weight"] * seg_bce
            + config["training"]["cldice_weight"] * seg_cldice
        )

        cls_loss = cls_loss_fn(cls_logits, target_ids)
        aux_loss = aux_ce_loss_fn(aux_logits, aux_target_mask)
        
        center_loss = bce_loss_fn(center_logits, true_center).flatten(1).mean(dim=1).mean()
        vec_loss = vector_direction_loss(vec_preds, true_vec, target_mask)
        
        total = (
            seg_per_sample
            + config["training"]["cls_weight"] * cls_loss
            + config["training"]["aux_weight"] * aux_loss
            + config["training"]["center_weight"] * center_loss
            + config["training"]["vector_weight"] * vec_loss
        )
        return total, seg_per_sample, cls_loss, aux_loss, center_loss, vec_loss

    writer = SummaryWriter(log_dir=log_dir)
    best_val_f1 = -1.0
    start_epoch = 0
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    last_path = os.path.join(ckpt_dir, "last_model.pth")

    if os.path.exists(last_path):
        print(f"[INFO] Încărcare checkpoint pentru reluare din {last_path}...")
        checkpoint = torch.load(last_path, map_location=device)
        # Verificăm dacă e noul format cu dicționar sau formatul vechi doar cu state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_f1 = checkpoint.get("best_val_f1", -1.0)
            print(f"[INFO] Antrenamentul se va relua de la epoca {start_epoch} (Best F1 anterior: {best_val_f1:.4f})")
        else:
            model.load_state_dict(checkpoint)
            print("[INFO] S-au încărcat doar greutățile modelului (format vechi de checkpoint).")

    threshold_by_target = {
        0: {"threshold": float(config["postprocess"].get("rca_threshold", 0.5))},
        1: {"threshold": float(config["postprocess"].get("lca_threshold", 0.5))},
    }

    for epoch in range(start_epoch, config["training"]["epochs"]):
        train_loss, train_seg_loss, train_cls_loss, train_aux_loss, train_ctr_loss, train_vec_loss = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            accum_steps=config["training"].get("accum_steps", 4)
        )

        (
            val_loss,
            val_seg_loss,
            val_cls_loss,
            val_aux_loss,
            val_ctr_loss,
            val_vec_loss,
            val_f1,
            val_iou,
            val_f1_exact,
            val_iou_exact,
            val_by_target,
            val_cls_acc,
            val_aux_acc,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            threshold_by_target=threshold_by_target,
            routing_mode=config["model"].get("inference_routing", "soft"),
        )
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/train_vec", train_vec_loss, epoch)
        writer.add_scalar("Loss/val_vec", val_vec_loss, epoch)
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
                routing_mode=config["model"].get("inference_routing", "soft"),
                step=epoch,
                split_name="val",
                max_samples=int(config["logging"].get("visualize_num_samples", 4)),
            )

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
        }
        torch.save(checkpoint_dict, last_path)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)

        # Print elegant care extrage safe floatul din dictionar
        thr_rca_print = threshold_by_target[0].get("threshold", 0.5) if isinstance(threshold_by_target[0], dict) else threshold_by_target[0]
        thr_lca_print = threshold_by_target[1].get("threshold", 0.5) if isinstance(threshold_by_target[1], dict) else threshold_by_target[1]

        print(
            f"Epoch {epoch + 1}/{config['training']['epochs']} | "
            f"val_loss={val_loss:.4f} | "
            f"val_f1={val_f1:.4f} | RCA_f1={val_by_target[0]['f1']:.4f} | LCA_f1={val_by_target[1]['f1']:.4f} | "
            f"thr(RCA/LCA)=({thr_rca_print:.2f}/{thr_lca_print:.2f})"
        )

    # [FIX INGINERESC 5: Integrarea Grid Search-ului înainte de Testare]
    model.load_state_dict(torch.load(best_path, map_location=device))
    
    print("\n--- Running Topology Grid Search on Validation Set ---")
    search_cfg = {
        "threshold_grid": [0.35, 0.40, 0.45, 0.50, 0.55],
        "close_kernel_grid": [0, 3, 5],
        "min_size_grid": [0, 20, 50, 100],
        "keep_largest_grid": [False] # False pentru LCA ca să nu șteargă ramurile LCX-ului
    }
    best_postprocess_cfg = find_best_thresholds_by_target(model, val_loader, search_cfg)
    print(f"Optimal RCA Config: {best_postprocess_cfg[0]}")
    print(f"Optimal LCA Config: {best_postprocess_cfg[1]}")

    (
        test_loss,
        test_seg_loss,
        test_cls_loss,
        test_aux_loss,
        test_ctr_loss,
        test_vec_loss,
        test_f1,
        test_iou,
        test_f1_exact,
        test_iou_exact,
        test_by_target,
        test_cls_acc,
        test_aux_acc,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        threshold_by_target=best_postprocess_cfg, # Acum folosim parametrii topologici optimi!
        routing_mode=config["model"].get("inference_routing", "soft"),
    )

    writer.add_scalar("Test/F1", test_f1)
    writer.add_scalar("Test/RCA_F1", test_by_target[0]["f1"])
    writer.add_scalar("Test/LCA_F1", test_by_target[1]["f1"])
    
    log_tensorboard_visuals(
        writer=writer,
        model=model,
        loader=test_loader,
        threshold_by_target=best_postprocess_cfg,
        routing_mode=config["model"].get("inference_routing", "soft"),
        step=int(config["training"]["epochs"]),
        split_name="test",
        max_samples=int(config["logging"].get("visualize_num_samples", 4)),
    )
    writer.close()

    print("\n=== Targeted Vessel Segmentation Test (best by val F1) ===")
    print(f"Test F1: {test_f1:.4f} (IoU: {test_iou:.4f})")
    print(f"Inference routing: {config['model'].get('inference_routing', 'soft')}")
    print(
        f"RCA: n={test_by_target[0]['n']} | F1={test_by_target[0]['f1']:.4f}\n"
        f"LCA: n={test_by_target[1]['n']} | F1={test_by_target[1]['f1']:.4f}"
    )

if __name__ == "__main__":
    main()