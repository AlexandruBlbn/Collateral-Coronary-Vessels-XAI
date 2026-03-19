import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


RCA_LABELS = {"1", "2", "3", "4", "16", "16a", "16b", "16c"}
LCX_LABELS = {"11", "12", "13", "14", "14a", "14b", "15"}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def resolve_path(path_arg: str) -> str:
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


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


def load_hard_rows(hard_csv: str, top_k: int, target_id_filter: int) -> List[dict]:
    rows = []
    with open(hard_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t_id = int(row.get("target_id", -1))
            except ValueError:
                continue
            if t_id not in (0, 1):
                continue
            if target_id_filter in (0, 1) and t_id != target_id_filter:
                continue

            try:
                row["distance"] = float(row.get("distance", "nan"))
            except ValueError:
                row["distance"] = float("nan")

            try:
                row["rank_within_target"] = int(float(row.get("rank_within_target", 0)))
            except ValueError:
                row["rank_within_target"] = 0

            rows.append(row)

    rows.sort(key=lambda r: (np.isnan(r["distance"]), -r["distance"]))
    if top_k > 0:
        rows = rows[:top_k]
    return rows


def make_overlay(image_u8: np.ndarray, mask01: np.ndarray, alpha: float, color_rgb=(255, 0, 0)) -> np.ndarray:
    img_rgb = np.stack([image_u8, image_u8, image_u8], axis=-1).astype(np.float32)
    color = np.array(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    m = mask01.astype(np.float32)[..., None]
    out = img_rgb * (1.0 - alpha * m) + color * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def plot_case_panel(
    image_u8: np.ndarray,
    mask01: np.ndarray,
    out_path: str,
    title_prefix: str,
    alpha: float,
):
    overlay = make_overlay(image_u8, mask01, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    axes[0].imshow(image_u8, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Data")
    axes[0].axis("off")

    axes[1].imshow(mask01, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Label")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlapped")
    axes[2].axis("off")

    fig.suptitle(title_prefix, fontsize=11)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Data | Label | Overlapped panels for hard cases.")
    parser.add_argument("--hard-csv", type=str, default="results/hard_case_mining/hard_cases_top_10pct_train.csv")
    parser.add_argument("--syntax-root", type=str, default="data/ARCADE/Unprocessed/arcade/syntax")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="results/hard_case_mining/hard_case_panels")
    parser.add_argument("--top-k", type=int, default=0, help="If >0, only the top-k hardest rows are plotted.")
    parser.add_argument("--target-id", type=int, default=-1, help="Set to 0 or 1 to filter target, or -1 for both.")
    parser.add_argument("--resize", type=int, default=512, help="Resize output panels to this square size; <=0 keeps native size.")
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    return parser.parse_args()


def main():
    args = parse_args()
    args.hard_csv = resolve_path(args.hard_csv)
    args.syntax_root = resolve_path(args.syntax_root)
    args.output_dir = resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.hard_csv):
        raise FileNotFoundError(f"Hard-cases CSV not found: {args.hard_csv}")
    if not os.path.isdir(args.syntax_root):
        raise FileNotFoundError(f"Syntax root not found: {args.syntax_root}")

    split_index = load_split_index(args.syntax_root, args.split)
    hard_rows = load_hard_rows(args.hard_csv, top_k=int(args.top_k), target_id_filter=int(args.target_id))

    if not hard_rows:
        raise RuntimeError("No hard-case rows found after filtering.")

    index_csv_path = os.path.join(args.output_dir, "panels_index.csv")
    with open(index_csv_path, "w", newline="", encoding="utf-8") as f_idx:
        writer = csv.writer(f_idx)
        writer.writerow([
            "panel_path",
            "file_name",
            "patient_number",
            "target_id",
            "distance",
            "rank_within_target",
        ])

        pbar = tqdm(hard_rows, total=len(hard_rows), desc="Plot hard cases")
        for i, row in enumerate(pbar, start=1):
            file_name = str(row.get("file_name", "")).strip()
            if not file_name:
                continue
            if file_name not in split_index:
                continue

            meta = split_index[file_name]
            width = int(meta["width"])
            height = int(meta["height"])

            target_id = int(row["target_id"])
            polygons = meta["rca_polygons"] if target_id == 0 else meta["lca_polygons"]
            mask01 = rasterize_polygons(width, height, polygons)

            image_path = Path(args.syntax_root) / args.split / "images" / file_name
            if not image_path.is_file():
                continue
            image_u8 = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)

            if args.resize and args.resize > 0:
                s = int(args.resize)
                image_u8 = np.array(Image.fromarray(image_u8).resize((s, s), resample=Image.BILINEAR), dtype=np.uint8)
                mask01 = np.array(Image.fromarray((mask01 * 255).astype(np.uint8)).resize((s, s), resample=Image.NEAREST), dtype=np.uint8)
                mask01 = (mask01 > 0).astype(np.uint8)

            stem = Path(file_name).stem
            panel_name = f"{i:04d}_t{target_id}_{stem}.png"
            panel_path = os.path.join(args.output_dir, panel_name)

            title_prefix = (
                f"{file_name} | patient={row.get('patient_number', '')} | "
                f"target={target_id} | dist={float(row.get('distance', float('nan'))):.4f} | "
                f"rank={int(row.get('rank_within_target', 0))}"
            )
            plot_case_panel(
                image_u8=image_u8,
                mask01=mask01,
                out_path=panel_path,
                title_prefix=title_prefix,
                alpha=float(args.overlay_alpha),
            )

            writer.writerow([
                panel_path,
                file_name,
                row.get("patient_number", ""),
                target_id,
                row.get("distance", ""),
                row.get("rank_within_target", ""),
            ])

    print("[DONE] Hard-case panels generated.")
    print(f"  - Panels folder: {args.output_dir}")
    print(f"  - Panel index CSV: {index_csv_path}")


if __name__ == "__main__":
    main()
