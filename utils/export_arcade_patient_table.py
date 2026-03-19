import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

# Vessel family mapping used by ARCADE SYNTAX challenge papers/rules.
RCA_LABELS = {"1", "2", "3", "4", "16", "16a", "16b", "16c"}
LCX_LABELS = {"11", "12", "13", "14", "14a", "14b", "15"}

MAIN_ARTERY_CLASS_TO_ID = {
    "RCA": 0,
    "LCA": 1,
    "BOTH": 2,
    "UNKNOWN": 3,
}


def safe_stem(file_name: str) -> str:
    try:
        return Path(file_name).stem
    except Exception:
        return str(file_name)


def label_to_tree(label_name: str) -> str:
    if label_name in RCA_LABELS:
        return "RCA"
    if label_name in LCX_LABELS:
        return "LCX"
    if label_name == "stenosis":
        return "stenosis"
    return "LAD_or_left_main"


def main_artery_target(area_rca: float, area_lca: float) -> str:
    has_rca = area_rca > 0.0
    has_lca = area_lca > 0.0
    if has_rca and not has_lca:
        return "RCA"
    if has_lca and not has_rca:
        return "LCA"
    if has_rca and has_lca:
        return "BOTH"
    return "UNKNOWN"


def summarize_split(split_name: str, json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}

    images = {}
    for img in data.get("images", []):
        image_id = int(img["id"])
        images[image_id] = {
            "split": split_name,
            "image_id": image_id,
            "case_id": safe_stem(str(img.get("file_name", ""))),
            "file_name": str(img.get("file_name", "")),
            "width": int(img.get("width", 0)),
            "height": int(img.get("height", 0)),
            "num_annotations": 0,
            "segments_present": set(),
            "vessel_groups": set(),
            "total_area": 0.0,
            "bbox_count": 0,
            "bbox_total_w": 0.0,
            "bbox_total_h": 0.0,
            "tree_annotation_count": defaultdict(int),
            "tree_area": defaultdict(float),
        }

    segment_rows = []
    segment_stats = defaultdict(lambda: {
        "count": 0,
        "sum_area": 0.0,
        "tree": "",
    })

    for ann in data.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        category_id = int(ann.get("category_id", -1))
        label = categories.get(category_id, f"unknown_{category_id}")
        tree = label_to_tree(label)
        area = float(ann.get("area", 0.0))
        bbox = ann.get("bbox", [])

        if image_id not in images:
            continue

        img_info = images[image_id]
        img_info["num_annotations"] += 1
        img_info["segments_present"].add(label)
        img_info["vessel_groups"].add(tree)
        img_info["total_area"] += area
        img_info["tree_annotation_count"][tree] += 1
        img_info["tree_area"][tree] += area

        if isinstance(bbox, list) and len(bbox) == 4:
            img_info["bbox_count"] += 1
            img_info["bbox_total_w"] += float(bbox[2])
            img_info["bbox_total_h"] += float(bbox[3])

        key = (image_id, label)
        segment_stats[key]["count"] += 1
        segment_stats[key]["sum_area"] += area
        segment_stats[key]["tree"] = tree

    for (image_id, label), stats in segment_stats.items():
        image_row = images[image_id]
        segment_rows.append(
            {
                "split": split_name,
                "image_id": image_id,
                "case_id": image_row["case_id"],
                "file_name": image_row["file_name"],
                "segment_label": label,
                "tree_group": stats["tree"],
                "annotation_count": stats["count"],
                "sum_area": round(stats["sum_area"], 3),
            }
        )

    patient_rows = []
    for image_id in sorted(images):
        img = images[image_id]
        segments_sorted = sorted(img["segments_present"], key=lambda x: (len(x), x))
        vessel_groups = sorted(img["vessel_groups"])

        tree_area = dict(img["tree_area"])
        tree_count = dict(img["tree_annotation_count"])
        area_rca = float(tree_area.get("RCA", 0.0))
        area_lca = float(tree_area.get("LCX", 0.0) + tree_area.get("LAD_or_left_main", 0.0))
        target_main_artery = main_artery_target(area_rca, area_lca)
        target_main_artery_id = MAIN_ARTERY_CLASS_TO_ID[target_main_artery]

        if not tree_area:
            target_vessel = "unknown"
        elif len(tree_area) == 1:
            target_vessel = next(iter(tree_area.keys()))
        else:
            dominant_tree = max(tree_area.items(), key=lambda kv: kv[1])[0]
            target_vessel = f"multi({dominant_tree}_dominant)"

        patient_rows.append(
            {
                "split": img["split"],
                "image_id": img["image_id"],
                "patient_number": img["case_id"],
                "case_id": img["case_id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "num_annotations": img["num_annotations"],
                "num_unique_segments": len(segments_sorted),
                "segments_present": ";".join(segments_sorted),
                "vessel_groups_present": ";".join(vessel_groups),
                "has_RCA": int("RCA" in img["vessel_groups"]),
                "has_LCA": int(
                    ("LCX" in img["vessel_groups"]) or ("LAD_or_left_main" in img["vessel_groups"])
                ),
                "has_LCX": int("LCX" in img["vessel_groups"]),
                "has_LAD_or_left_main": int("LAD_or_left_main" in img["vessel_groups"]),
                "has_stenosis": int("stenosis" in img["vessel_groups"]),
                "target_main_artery": target_main_artery,
                "target_main_artery_id": target_main_artery_id,
                "target_vessel_type": target_vessel,
                "area_RCA": round(area_rca, 3),
                "area_LCX": round(tree_area.get("LCX", 0.0), 3),
                "area_LAD_or_left_main": round(tree_area.get("LAD_or_left_main", 0.0), 3),
                "area_LCA_total": round(area_lca, 3),
                "area_stenosis": round(tree_area.get("stenosis", 0.0), 3),
                "ann_count_RCA": int(tree_count.get("RCA", 0)),
                "ann_count_LCX": int(tree_count.get("LCX", 0)),
                "ann_count_LAD_or_left_main": int(tree_count.get("LAD_or_left_main", 0)),
                "ann_count_LCA_total": int(
                    tree_count.get("LCX", 0) + tree_count.get("LAD_or_left_main", 0)
                ),
                "ann_count_stenosis": int(tree_count.get("stenosis", 0)),
                "total_area": round(img["total_area"], 3),
                "mean_bbox_w": round(img["bbox_total_w"] / img["bbox_count"], 3) if img["bbox_count"] else 0.0,
                "mean_bbox_h": round(img["bbox_total_h"] / img["bbox_count"], 3) if img["bbox_count"] else 0.0,
            }
        )

    return patient_rows, segment_rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export ARCADE SYNTAX per-patient and per-segment summary tables from COCO annotations."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/ARCADE/Unprocessed/arcade/syntax"),
        help="Path containing split folders train/val/test.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to parse.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/arcade_patient_tables"),
        help="Directory where CSV tables are written.",
    )
    args = parser.parse_args()

    all_patient_rows = []
    all_segment_rows = []

    for split in args.splits:
        ann_path = args.base_dir / split / "annotations" / f"{split}.json"
        if not ann_path.is_file():
            print(f"[WARN] Missing annotation file for split '{split}': {ann_path}")
            continue

        patient_rows, segment_rows = summarize_split(split_name=split, json_path=ann_path)
        all_patient_rows.extend(patient_rows)
        all_segment_rows.extend(segment_rows)
        print(
            f"[OK] {split}: {len(patient_rows)} patients/images, "
            f"{len(segment_rows)} image-segment rows from {ann_path}"
        )

    if not all_patient_rows:
        raise SystemExit("No rows were generated. Check paths/splits.")

    patient_fields = [
        "split",
        "image_id",
        "patient_number",
        "case_id",
        "file_name",
        "width",
        "height",
        "num_annotations",
        "num_unique_segments",
        "segments_present",
        "vessel_groups_present",
        "has_RCA",
        "has_LCA",
        "has_LCX",
        "has_LAD_or_left_main",
        "has_stenosis",
        "target_main_artery",
        "target_main_artery_id",
        "target_vessel_type",
        "area_RCA",
        "area_LCX",
        "area_LAD_or_left_main",
        "area_LCA_total",
        "area_stenosis",
        "ann_count_RCA",
        "ann_count_LCX",
        "ann_count_LAD_or_left_main",
        "ann_count_LCA_total",
        "ann_count_stenosis",
        "total_area",
        "mean_bbox_w",
        "mean_bbox_h",
    ]
    segment_fields = [
        "split",
        "image_id",
        "case_id",
        "file_name",
        "segment_label",
        "tree_group",
        "annotation_count",
        "sum_area",
    ]

    patient_out = args.out_dir / "patient_summary.csv"
    segment_out = args.out_dir / "patient_segments.csv"
    target_out = args.out_dir / "patient_main_artery_targets.csv"

    target_rows = [
        {
            "split": r["split"],
            "patient_number": r["patient_number"],
            "image_id": r["image_id"],
            "file_name": r["file_name"],
            "target_main_artery": r["target_main_artery"],
            "target_main_artery_id": r["target_main_artery_id"],
            "area_RCA": r["area_RCA"],
            "area_LCA_total": r["area_LCA_total"],
        }
        for r in all_patient_rows
    ]
    target_fields = [
        "split",
        "patient_number",
        "image_id",
        "file_name",
        "target_main_artery",
        "target_main_artery_id",
        "area_RCA",
        "area_LCA_total",
    ]

    write_csv(patient_out, all_patient_rows, patient_fields)
    write_csv(segment_out, all_segment_rows, segment_fields)
    write_csv(target_out, target_rows, target_fields)

    print(f"[DONE] Wrote: {patient_out}")
    print(f"[DONE] Wrote: {segment_out}")
    print(f"[DONE] Wrote: {target_out}")


if __name__ == "__main__":
    main()
