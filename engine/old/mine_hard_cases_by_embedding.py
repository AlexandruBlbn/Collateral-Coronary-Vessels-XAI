import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp


TARGET_NAMES = {0: "RCA", 1: "LCA"}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class SampleMeta:
    file_name: str
    patient_number: str
    target_id: int
    image_path: str


class TargetedTrainImagesDataset(Dataset):
    def __init__(
        self,
        target_csv: str,
        syntax_root: str,
        split: str = "train",
        img_size: int = 256,
    ):
        self.img_size = int(img_size)
        self.rows: List[SampleMeta] = []

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
                if not file_name:
                    continue

                image_path = syntax_root / split / "images" / file_name
                if not image_path.is_file():
                    continue

                self.rows.append(
                    SampleMeta(
                        file_name=file_name,
                        patient_number=str(row.get("patient_number", "")),
                        target_id=target_id,
                        image_path=str(image_path),
                    )
                )

        if not self.rows:
            raise RuntimeError("No samples found for the requested split and target ids.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row.image_path).convert("L")
        image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        img_t = TF.to_tensor(np.array(image, dtype=np.uint8))
        return img_t, row.target_id, row.file_name, row.patient_number


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
        seg_logits = (seg_both * cls_probs).sum(dim=1)

        aux_logits = self.aux_head_vessel(decoder_out)
        return seg_logits, cls_logits, seg_both, aux_logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.seg_model.encoder(x)
        return self.cls_pool(feats[-1]).flatten(1)


def _build_model(args, device: torch.device) -> MultiTaskTargetedUNet:
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
        print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected)}")
    model.eval()
    return model


def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def _compute_center(vectors: np.ndarray, normalize_center: bool) -> np.ndarray:
    center = vectors.mean(axis=0)
    if normalize_center:
        denom = max(1e-12, float(np.linalg.norm(center)))
        center = center / denom
    return center.astype(np.float32)


def _compute_distances(vectors: np.ndarray, center: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return np.linalg.norm(vectors - center[None, :], axis=1)
    if metric == "cosine":
        v = _l2_normalize(vectors, axis=1)
        c = center / max(1e-12, float(np.linalg.norm(center)))
        sim = np.sum(v * c[None, :], axis=1)
        return 1.0 - sim
    raise ValueError(f"Unsupported metric: {metric}")


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("PCA expects a 2D array.")
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32)

    x0 = x - x.mean(axis=0, keepdims=True)
    # SVD-based PCA keeps dependencies minimal and works for high-dimensional embeddings.
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    comps = vt[:2].T
    proj = x0 @ comps
    if proj.shape[1] == 1:
        proj = np.concatenate([proj, np.zeros((proj.shape[0], 1), dtype=proj.dtype)], axis=1)
    return proj.astype(np.float32)


def save_plots(
    output_dir: str,
    split: str,
    distance_metric: str,
    hard_ratio: float,
    X: np.ndarray,
    target_ids_arr: np.ndarray,
    distances: np.ndarray,
    hard_flags: np.ndarray,
):
    os.makedirs(output_dir, exist_ok=True)
    hard_pct = int(round(100.0 * hard_ratio))

    for t_id in (0, 1):
        idx = np.where(target_ids_arr == t_id)[0]
        if idx.size == 0:
            continue

        d = distances[idx]
        title_name = TARGET_NAMES.get(t_id, str(t_id))
        cutoff = float(np.quantile(d, max(0.0, 1.0 - hard_ratio)))

        fig = plt.figure(figsize=(8.5, 5.0))
        plt.hist(d, bins=30, color="#3B82F6", alpha=0.8, edgecolor="white")
        plt.axvline(cutoff, color="#EF4444", linestyle="--", linewidth=2.0, label=f"Top {hard_pct}% cutoff")
        plt.xlabel(f"Distance to {title_name} center ({distance_metric})")
        plt.ylabel("Number of samples")
        plt.title(f"{title_name} distance distribution ({split})")
        plt.legend(loc="upper right")
        plt.tight_layout()
        hist_path = os.path.join(output_dir, f"plot_hist_{title_name.lower()}_{split}.png")
        fig.savefig(hist_path, dpi=180)
        plt.close(fig)

        order = np.argsort(-d)
        sorted_d = d[order]
        k = max(1, int(math.ceil(hard_ratio * len(sorted_d))))

        fig = plt.figure(figsize=(8.5, 5.0))
        x_axis = np.arange(1, len(sorted_d) + 1)
        plt.plot(x_axis, sorted_d, color="#111827", linewidth=1.8, label="Sorted distance")
        plt.axvspan(1, k, color="#FCA5A5", alpha=0.35, label=f"Hard zone (top {hard_pct}%)")
        plt.xlabel("Rank (1 = hardest)")
        plt.ylabel(f"Distance ({distance_metric})")
        plt.title(f"{title_name} hardness curve ({split})")
        plt.legend(loc="upper right")
        plt.tight_layout()
        curve_path = os.path.join(output_dir, f"plot_rankcurve_{title_name.lower()}_{split}.png")
        fig.savefig(curve_path, dpi=180)
        plt.close(fig)

    proj = _pca_2d(X)
    fig = plt.figure(figsize=(8.5, 6.5))
    colors = np.where(target_ids_arr == 0, "#2563EB", "#16A34A")
    alphas = np.where(hard_flags == 1, 0.95, 0.25)
    sizes = np.where(hard_flags == 1, 24.0, 11.0)
    for t_id in (0, 1):
        sel = target_ids_arr == t_id
        if not np.any(sel):
            continue
        plt.scatter(
            proj[sel, 0],
            proj[sel, 1],
            c=colors[sel],
            s=sizes[sel],
            alpha=float(np.mean(alphas[sel])),
            label=f"{TARGET_NAMES[t_id]}",
            linewidths=0.0,
        )

    hard_sel = hard_flags == 1
    if np.any(hard_sel):
        plt.scatter(
            proj[hard_sel, 0],
            proj[hard_sel, 1],
            facecolors="none",
            edgecolors="#DC2626",
            s=42,
            linewidths=0.8,
            label=f"Hard ({hard_pct}%)",
        )

    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(f"Embedding map ({split}) - color by target, circle = hard")
    plt.legend(loc="best")
    plt.tight_layout()
    pca_path = os.path.join(output_dir, f"plot_embedding_pca_{split}.png")
    fig.savefig(pca_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Mine hard cases from train embeddings using per-target centers.")
    parser.add_argument("--target-csv", type=str, default="results/arcade_patient_tables/patient_main_artery_targets.csv")
    parser.add_argument("--syntax-root", type=str, default="data/ARCADE/Unprocessed/arcade/syntax")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Model checkpoint path. If omitted, script auto-detects best available targeted segmentation checkpoint.",
    )
    parser.add_argument("--encoder-name", type=str, default="tu-efficientnetv2_s")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--classes", type=int, default=1)
    parser.add_argument("--aux-num-classes", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--distance-metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--l2-normalize-embeddings", action="store_true")
    parser.add_argument("--hard-ratio", type=float, default=0.10, help="Top ratio per target to mark as hard, e.g. 0.10.")
    parser.add_argument("--weight-lambda", type=float, default=0.7, help="Weight scaling: w = 1 + lambda * normalized_distance.")
    parser.add_argument("--max-weight", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="results/hard_case_mining")
    return parser.parse_args()


def resolve_path(path_arg: str) -> str:
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def resolve_default_checkpoint(ckpt_arg: str) -> str:
    if ckpt_arg:
        return resolve_path(ckpt_arg)

    candidates = [
        "checkpoints/syntax_targeted_vessel_segmentation_v2_fficientnetv2/best_model.pth",
        "checkpoints/syntax_targeted_vessel_segmentation/best_model.pth",
    ]
    for path in candidates:
        resolved = resolve_path(path)
        if os.path.isfile(resolved):
            return resolved

    raise FileNotFoundError(
        "Could not auto-detect a targeted segmentation checkpoint. "
        "Please provide --checkpoint explicitly."
    )


def main():
    args = parse_args()
    args.target_csv = resolve_path(args.target_csv)
    args.syntax_root = resolve_path(args.syntax_root)
    args.output_dir = resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    args.checkpoint = resolve_default_checkpoint(args.checkpoint)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.target_csv):
        raise FileNotFoundError(f"Target CSV not found: {args.target_csv}")
    if not os.path.isdir(args.syntax_root):
        raise FileNotFoundError(f"Syntax root folder not found: {args.syntax_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")

    dataset = TargetedTrainImagesDataset(
        target_csv=args.target_csv,
        syntax_root=args.syntax_root,
        split=args.split,
        img_size=args.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    model = _build_model(args, device)

    embeddings: List[np.ndarray] = []
    metas: List[SampleMeta] = []

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Extract embeddings")
        for images, target_ids, file_names, patient_numbers in pbar:
            images = images.to(device)
            emb = model.extract_embedding(images).detach().cpu().numpy().astype(np.float32)
            embeddings.append(emb)

            for t_id, f_name, p_no in zip(target_ids.tolist(), list(file_names), list(patient_numbers)):
                metas.append(
                    SampleMeta(
                        file_name=str(f_name),
                        patient_number=str(p_no),
                        target_id=int(t_id),
                        image_path="",
                    )
                )

    if not embeddings:
        raise RuntimeError("No embeddings were extracted.")

    X = np.concatenate(embeddings, axis=0)
    if args.l2_normalize_embeddings:
        X = _l2_normalize(X, axis=1)

    target_ids_arr = np.array([m.target_id for m in metas], dtype=np.int64)
    if X.shape[0] != target_ids_arr.shape[0]:
        raise RuntimeError("Embedding count does not match metadata count.")

    centers: Dict[int, np.ndarray] = {}
    distances = np.zeros((X.shape[0],), dtype=np.float32)
    hard_flags = np.zeros((X.shape[0],), dtype=np.int32)
    ranks = np.zeros((X.shape[0],), dtype=np.int32)
    n_target = np.zeros((X.shape[0],), dtype=np.int32)
    normalized_dist = np.zeros((X.shape[0],), dtype=np.float32)
    sample_weight = np.ones((X.shape[0],), dtype=np.float32)

    summary = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "distance_metric": args.distance_metric,
        "l2_normalize_embeddings": bool(args.l2_normalize_embeddings),
        "hard_ratio": float(args.hard_ratio),
        "targets": {},
    }

    for t_id in (0, 1):
        idx = np.where(target_ids_arr == t_id)[0]
        if idx.size == 0:
            summary["targets"][str(t_id)] = {"name": TARGET_NAMES[t_id], "n": 0}
            continue

        Xt = X[idx]
        center = _compute_center(Xt, normalize_center=(args.distance_metric == "cosine"))
        centers[t_id] = center

        d = _compute_distances(Xt, center, metric=args.distance_metric).astype(np.float32)
        d_norm = _minmax(d)
        w = 1.0 + float(args.weight_lambda) * d_norm
        w = np.clip(w, 1.0, float(args.max_weight)).astype(np.float32)

        order_desc = np.argsort(-d)
        rank_in_t = np.empty_like(order_desc)
        rank_in_t[order_desc] = np.arange(1, order_desc.size + 1)

        k = max(1, int(math.ceil(float(args.hard_ratio) * idx.size)))
        hard_local = (rank_in_t <= k).astype(np.int32)

        distances[idx] = d
        normalized_dist[idx] = d_norm
        sample_weight[idx] = w
        hard_flags[idx] = hard_local
        ranks[idx] = rank_in_t
        n_target[idx] = int(idx.size)

        top_local = order_desc[:k]
        top_global_indices = idx[top_local]
        top_examples = [
            {
                "file_name": metas[gidx].file_name,
                "patient_number": metas[gidx].patient_number,
                "distance": float(distances[gidx]),
                "rank": int(ranks[gidx]),
            }
            for gidx in top_global_indices[:20]
        ]

        summary["targets"][str(t_id)] = {
            "name": TARGET_NAMES[t_id],
            "n": int(idx.size),
            "center_norm": float(np.linalg.norm(center)),
            "distance_mean": float(d.mean()),
            "distance_std": float(d.std()),
            "distance_p90": float(np.quantile(d, 0.90)),
            "distance_p95": float(np.quantile(d, 0.95)),
            "hard_k": int(k),
            "top_examples_preview": top_examples,
        }

    np.savez_compressed(
        os.path.join(args.output_dir, f"embeddings_{args.split}.npz"),
        embeddings=X.astype(np.float32),
        target_ids=target_ids_arr.astype(np.int64),
        file_names=np.array([m.file_name for m in metas], dtype=object),
        patient_numbers=np.array([m.patient_number for m in metas], dtype=object),
    )

    np.savez_compressed(
        os.path.join(args.output_dir, f"centers_{args.split}.npz"),
        center_rca=centers.get(0, np.array([], dtype=np.float32)),
        center_lca=centers.get(1, np.array([], dtype=np.float32)),
        distance_metric=np.array([args.distance_metric], dtype=object),
    )

    all_csv_path = os.path.join(args.output_dir, f"distance_report_{args.split}.csv")
    hard_csv_path = os.path.join(args.output_dir, f"hard_cases_top_{int(args.hard_ratio * 100)}pct_{args.split}.csv")
    weights_csv_path = os.path.join(args.output_dir, f"sample_weights_{args.split}.csv")

    header = [
        "file_name",
        "patient_number",
        "target_id",
        "target_name",
        "distance",
        "distance_normalized",
        "rank_within_target",
        "n_in_target",
        "is_hard",
        "sample_weight",
    ]

    with open(all_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, m in enumerate(metas):
            writer.writerow(
                [
                    m.file_name,
                    m.patient_number,
                    int(m.target_id),
                    TARGET_NAMES.get(int(m.target_id), "UNK"),
                    float(distances[i]),
                    float(normalized_dist[i]),
                    int(ranks[i]),
                    int(n_target[i]),
                    int(hard_flags[i]),
                    float(sample_weight[i]),
                ]
            )

    with open(hard_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, m in enumerate(metas):
            if int(hard_flags[i]) != 1:
                continue
            writer.writerow(
                [
                    m.file_name,
                    m.patient_number,
                    int(m.target_id),
                    TARGET_NAMES.get(int(m.target_id), "UNK"),
                    float(distances[i]),
                    float(normalized_dist[i]),
                    int(ranks[i]),
                    int(n_target[i]),
                    int(hard_flags[i]),
                    float(sample_weight[i]),
                ]
            )

    with open(weights_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "patient_number", "target_id", "sample_weight"])
        for i, m in enumerate(metas):
            writer.writerow([m.file_name, m.patient_number, int(m.target_id), float(sample_weight[i])])

    summary_path = os.path.join(args.output_dir, f"summary_{args.split}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_plots(
        output_dir=args.output_dir,
        split=args.split,
        distance_metric=args.distance_metric,
        hard_ratio=float(args.hard_ratio),
        X=X,
        target_ids_arr=target_ids_arr,
        distances=distances,
        hard_flags=hard_flags,
    )

    plot_paths = [
        os.path.join(args.output_dir, f"plot_hist_rca_{args.split}.png"),
        os.path.join(args.output_dir, f"plot_hist_lca_{args.split}.png"),
        os.path.join(args.output_dir, f"plot_rankcurve_rca_{args.split}.png"),
        os.path.join(args.output_dir, f"plot_rankcurve_lca_{args.split}.png"),
        os.path.join(args.output_dir, f"plot_embedding_pca_{args.split}.png"),
    ]

    print("[DONE] Hard-case mining artifacts saved:")
    print(f"  - {all_csv_path}")
    print(f"  - {hard_csv_path}")
    print(f"  - {weights_csv_path}")
    print(f"  - {summary_path}")
    print(f"  - {os.path.join(args.output_dir, f'embeddings_{args.split}.npz')}")
    print(f"  - {os.path.join(args.output_dir, f'centers_{args.split}.npz')}")
    print("[DONE] Plot artifacts saved:")
    for p in plot_paths:
        if os.path.isfile(p):
            print(f"  - {p}")


if __name__ == "__main__":
    main()
