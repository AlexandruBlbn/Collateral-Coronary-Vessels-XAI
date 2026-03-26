import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp


RCA_LABELS = {"1", "2", "3", "4", "16", "16a", "16b", "16c"}
LCX_LABELS = {"11", "12", "13", "14", "14a", "14b", "15"}
TARGET_NAMES = {0: "RCA", 1: "LCA"}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class SampleRow:
    file_name: str
    patient_number: str
    target_id: int
    image_path: str
    width: int
    height: int
    polygons: List[List[float]]


class TargetedSplitDataset(Dataset):
    def __init__(
        self,
        target_csv: str,
        syntax_root: str,
        split: str,
        img_size: int = 256,
    ):
        self.img_size = int(img_size)
        self.rows: List[SampleRow] = []

        split_index = load_split_index(syntax_root, split)
        syntax_root = Path(syntax_root)

        with open(target_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "") != split:
                    continue

                try:
                    target_id = int(row.get("target_main_artery_id", -1))
                except ValueError:
                    continue
                if target_id not in (0, 1):
                    continue

                file_name = str(row.get("file_name", "")).strip()
                if not file_name or file_name not in split_index:
                    continue

                meta = split_index[file_name]
                image_path = syntax_root / split / "images" / file_name
                if not image_path.is_file():
                    continue

                polygons = meta["rca_polygons"] if target_id == 0 else meta["lca_polygons"]
                self.rows.append(
                    SampleRow(
                        file_name=file_name,
                        patient_number=str(row.get("patient_number", "")),
                        target_id=target_id,
                        image_path=str(image_path),
                        width=int(meta["width"]),
                        height=int(meta["height"]),
                        polygons=polygons,
                    )
                )

        if not self.rows:
            raise RuntimeError(f"No samples found for split={split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        image = Image.open(row.image_path).convert("L")
        mask = rasterize_polygons(row.width, row.height, row.polygons)

        image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        mask = Image.fromarray((mask * 255).astype(np.uint8)).resize((self.img_size, self.img_size), resample=Image.NEAREST)

        img_t = TF.to_tensor(np.array(image, dtype=np.uint8))
        mask_t = (TF.to_tensor(np.array(mask, dtype=np.uint8)) > 0).float()

        return img_t, mask_t, row.target_id, row.file_name, row.patient_number


def resolve_path(path_arg: str) -> str:
    p = Path(path_arg).expanduser()
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def load_split_index(syntax_root: str, split: str) -> Dict[str, dict]:
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
        images[image_id]["lca_polygons"] = images[image_id]["lad_polygons"] + images[image_id]["lcx_polygons"]

    return {v["file_name"]: v for v in images.values()}


def rasterize_polygons(width: int, height: int, polygons: List[List[float]]) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        points = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
        draw.polygon(points, fill=255)
    return (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)


class MultiTaskTargetedUNet(nn.Module):
    def __init__(self, encoder_name="tu-efficientnetv2_s", encoder_weights=None, in_channels=1, classes=1, aux_num_classes=4):
        super().__init__()
        self.seg_model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        bottleneck_ch = self.seg_model.encoder.out_channels[-1]
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(bottleneck_ch, 2)
        dec_ch = self.seg_model.segmentation_head[0].in_channels
        self.seg_head_rca = nn.Conv2d(dec_ch, classes, kernel_size=3, padding=1)
        self.seg_head_lca = nn.Conv2d(dec_ch, classes, kernel_size=3, padding=1)
        self.aux_head_vessel = nn.Conv2d(dec_ch, aux_num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        feats = self.seg_model.encoder(x)
        decoder_out = self.seg_model.decoder(feats)

        seg_rca = self.seg_head_rca(decoder_out)
        seg_lca = self.seg_head_lca(decoder_out)
        seg_both = torch.stack([seg_rca, seg_lca], dim=1)

        cls_logits = self.cls_head(self.cls_pool(feats[-1]).flatten(1))
        cls_probs = torch.softmax(cls_logits, dim=1).view(-1, 2, 1, 1, 1)
        seg_logits_soft = (seg_both * cls_probs).sum(dim=1)
        aux_logits = self.aux_head_vessel(decoder_out)
        return seg_logits_soft, cls_logits, seg_both, aux_logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.seg_model.encoder(x)
        return self.cls_pool(feats[-1]).flatten(1)


def resolve_default_checkpoint(ckpt_arg: str) -> str:
    if ckpt_arg:
        return resolve_path(ckpt_arg)

    candidates = [
        "checkpoints/syntax_targeted_vessel_segmentation_v2_fficientnetv2/best_model.pth",
        "checkpoints/syntax_targeted_vessel_segmentation/best_model.pth",
    ]
    for rel in candidates:
        p = resolve_path(rel)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Could not auto-detect checkpoint. Please pass --checkpoint.")


def build_model(args, device: torch.device) -> MultiTaskTargetedUNet:
    model = MultiTaskTargetedUNet(
        encoder_name=args.encoder_name,
        encoder_weights=None,
        in_channels=args.in_channels,
        classes=args.classes,
        aux_num_classes=args.aux_num_classes,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    model.eval()
    return model


def f1_iou_per_sample(pred01: torch.Tensor, gt01: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    # pred/gt: [B,1,H,W]
    p = pred01.flatten(1).int()
    g = gt01.flatten(1).int()

    tp = (p & g).sum(dim=1).float()
    fp = (p & (1 - g)).sum(dim=1).float()
    fn = ((1 - p) & g).sum(dim=1).float()

    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return f1.detach().cpu().numpy(), iou.detach().cpu().numpy()


def select_seg_logits(seg_logits_soft, cls_logits, seg_both, target_ids, routing_mode: str):
    if routing_mode == "oracle_target":
        idx = target_ids.long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both.shape[-2], seg_both.shape[-1])
        return torch.gather(seg_both, dim=1, index=idx).squeeze(1)

    if routing_mode == "hard_pred":
        pred_ids = torch.argmax(cls_logits, dim=1)
        idx = pred_ids.long().view(-1, 1, 1, 1, 1).expand(-1, 1, 1, seg_both.shape[-2], seg_both.shape[-1])
        return torch.gather(seg_both, dim=1, index=idx).squeeze(1)

    return seg_logits_soft


def compute_distances(emb: np.ndarray, target_ids: np.ndarray, metric: str) -> np.ndarray:
    out = np.zeros((emb.shape[0],), dtype=np.float32)
    for t_id in (0, 1):
        idx = np.where(target_ids == t_id)[0]
        if idx.size == 0:
            continue
        x = emb[idx]
        center = x.mean(axis=0)

        if metric == "euclidean":
            d = np.linalg.norm(x - center[None, :], axis=1)
        elif metric == "cosine":
            x_n = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            c_n = center / max(np.linalg.norm(center), 1e-12)
            d = 1.0 - np.sum(x_n * c_n[None, :], axis=1)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        out[idx] = d.astype(np.float32)
    return out


def assign_bins_by_target(distances: np.ndarray, target_ids: np.ndarray) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    bins = np.empty((distances.shape[0],), dtype=object)
    q_stats: Dict[int, Dict[str, float]] = {}

    for t_id in (0, 1):
        idx = np.where(target_ids == t_id)[0]
        if idx.size == 0:
            continue

        d = distances[idx]
        q50 = float(np.quantile(d, 0.50))
        q90 = float(np.quantile(d, 0.90))
        q_stats[t_id] = {"q50": q50, "q90": q90}

        bins[idx] = np.where(d <= q50, "easy", np.where(d <= q90, "medium", "hard"))

    return bins, q_stats


def aggregate_bin_stats(rows: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, int, str], List[dict]] = {}
    for r in rows:
        k = (r["split"], int(r["target_id"]), str(r["hardness_bin"]))
        grouped.setdefault(k, []).append(r)

    out: List[dict] = []
    for (split, target_id, hardness_bin), items in grouped.items():
        dice_vals = np.array([float(x["dice"]) for x in items], dtype=np.float32)
        iou_vals = np.array([float(x["iou"]) for x in items], dtype=np.float32)
        out.append(
            {
                "split": split,
                "target_id": target_id,
                "target_name": TARGET_NAMES.get(target_id, "UNK"),
                "hardness_bin": hardness_bin,
                "n": int(len(items)),
                "mean_dice": float(dice_vals.mean()) if len(dice_vals) else 0.0,
                "median_dice": float(np.median(dice_vals)) if len(dice_vals) else 0.0,
                "mean_iou": float(iou_vals.mean()) if len(iou_vals) else 0.0,
                "median_iou": float(np.median(iou_vals)) if len(iou_vals) else 0.0,
            }
        )

    # stable display order
    order_bin = {"easy": 0, "medium": 1, "hard": 2}
    out.sort(key=lambda x: (x["split"], x["target_id"], order_bin.get(x["hardness_bin"], 99)))
    return out


def save_csv(path: str, rows: List[dict], header: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_dice_by_bin(summary_rows: List[dict], output_dir: str):
    for split in sorted({r["split"] for r in summary_rows}):
        fig = plt.figure(figsize=(8.5, 5.0))
        x_order = ["easy", "medium", "hard"]
        x = np.arange(len(x_order))
        width = 0.35

        for j, t_id in enumerate((0, 1)):
            vals = []
            for b in x_order:
                row = next((r for r in summary_rows if r["split"] == split and r["target_id"] == t_id and r["hardness_bin"] == b), None)
                vals.append(0.0 if row is None else float(row["mean_dice"]))
            pos = x + (j - 0.5) * width
            color = "#2563EB" if t_id == 0 else "#16A34A"
            plt.bar(pos, vals, width=width, color=color, alpha=0.85, label=TARGET_NAMES[t_id])

        plt.xticks(x, x_order)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Mean Dice")
        plt.title(f"Mean Dice by hardness bin ({split})")
        plt.legend(loc="best")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"plot_mean_dice_by_bin_{split}.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def analyze_split(model, loader, split: str, device: torch.device, routing_mode: str, distance_metric: str, threshold: float) -> Tuple[List[dict], Dict[int, Dict[str, float]]]:
    rows: List[dict] = []

    all_embeddings = []
    all_target_ids = []

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc=f"Analyze {split}")
        for images, masks, target_ids, file_names, patient_numbers in pbar:
            images = images.to(device)
            masks = masks.to(device)
            target_ids = target_ids.to(device)

            seg_logits_soft, cls_logits, seg_both, _ = model(images)
            seg_logits = select_seg_logits(seg_logits_soft, cls_logits, seg_both, target_ids, routing_mode=routing_mode)

            probs = torch.sigmoid(seg_logits)
            pred01 = (probs >= float(threshold)).int()
            f1_vals, iou_vals = f1_iou_per_sample(pred01, masks.int())

            emb = model.extract_embedding(images).detach().cpu().numpy().astype(np.float32)
            all_embeddings.append(emb)
            all_target_ids.append(target_ids.detach().cpu().numpy().astype(np.int64))

            for i in range(images.shape[0]):
                rows.append(
                    {
                        "split": split,
                        "file_name": str(file_names[i]),
                        "patient_number": str(patient_numbers[i]),
                        "target_id": int(target_ids[i].item()),
                        "target_name": TARGET_NAMES.get(int(target_ids[i].item()), "UNK"),
                        "dice": float(f1_vals[i]),
                        "f1": float(f1_vals[i]),
                        "iou": float(iou_vals[i]),
                    }
                )

    emb_all = np.concatenate(all_embeddings, axis=0)
    target_all = np.concatenate(all_target_ids, axis=0)
    dists = compute_distances(emb_all, target_all, metric=distance_metric)
    bins, q_stats = assign_bins_by_target(dists, target_all)

    for i in range(len(rows)):
        rows[i]["distance"] = float(dists[i])
        rows[i]["hardness_bin"] = str(bins[i])

    return rows, q_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-sample Dice/F1 on train/val and compare performance across hardness bins per target."
    )
    parser.add_argument("--target-csv", type=str, default="results/arcade_patient_tables/patient_main_artery_targets.csv")
    parser.add_argument("--syntax-root", type=str, default="data/ARCADE/Unprocessed/arcade/syntax")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results/hardness_performance")

    parser.add_argument("--encoder-name", type=str, default="tu-efficientnetv2_s")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--classes", type=int, default=1)
    parser.add_argument("--aux-num-classes", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--routing-mode", type=str, default="oracle_target", choices=["oracle_target", "hard_pred", "soft"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--distance-metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.target_csv = resolve_path(args.target_csv)
    args.syntax_root = resolve_path(args.syntax_root)
    args.output_dir = resolve_path(args.output_dir)
    args.checkpoint = resolve_default_checkpoint(args.checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.target_csv):
        raise FileNotFoundError(f"Target CSV not found: {args.target_csv}")
    if not os.path.isdir(args.syntax_root):
        raise FileNotFoundError(f"Syntax root not found: {args.syntax_root}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")

    model = build_model(args, device)

    per_sample_rows: List[dict] = []
    quantile_stats: Dict[str, Dict[int, Dict[str, float]]] = {}

    for split in ("train", "val"):
        ds = TargetedSplitDataset(
            target_csv=args.target_csv,
            syntax_root=args.syntax_root,
            split=split,
            img_size=args.img_size,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
        )

        rows_split, q_stats = analyze_split(
            model=model,
            loader=loader,
            split=split,
            device=device,
            routing_mode=args.routing_mode,
            distance_metric=args.distance_metric,
            threshold=args.threshold,
        )
        per_sample_rows.extend(rows_split)
        quantile_stats[split] = q_stats

    summary_rows = aggregate_bin_stats(per_sample_rows)

    per_sample_csv = os.path.join(args.output_dir, "per_sample_metrics_train_val.csv")
    summary_csv = os.path.join(args.output_dir, "summary_dice_by_bin_target.csv")
    quantile_json = os.path.join(args.output_dir, "distance_quantiles_by_split_target.json")

    save_csv(
        per_sample_csv,
        per_sample_rows,
        header=[
            "split",
            "file_name",
            "patient_number",
            "target_id",
            "target_name",
            "dice",
            "f1",
            "iou",
            "distance",
            "hardness_bin",
        ],
    )
    save_csv(
        summary_csv,
        summary_rows,
        header=[
            "split",
            "target_id",
            "target_name",
            "hardness_bin",
            "n",
            "mean_dice",
            "median_dice",
            "mean_iou",
            "median_iou",
        ],
    )

    with open(quantile_json, "w", encoding="utf-8") as f:
        json.dump(quantile_stats, f, indent=2)

    plot_dice_by_bin(summary_rows, args.output_dir)

    print("[DONE] Analysis artifacts:")
    print(f"  - {per_sample_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {quantile_json}")
    print(f"  - {os.path.join(args.output_dir, 'plot_mean_dice_by_bin_train.png')}")
    print(f"  - {os.path.join(args.output_dir, 'plot_mean_dice_by_bin_val.png')}")


if __name__ == "__main__":
    main()
