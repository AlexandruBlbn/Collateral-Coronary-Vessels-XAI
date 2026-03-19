import os
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter
import timm


ID_TO_NAME = {
    0: "RCA",
    1: "LCA",
    2: "BOTH",
    3: "UNKNOWN",
}


def build_coco_segmentation_index(syntax_root: str, split: str):
    ann_path = Path(syntax_root) / split / "annotations" / f"{split}.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_image = {
        int(img["id"]): {
            "file_name": img["file_name"],
            "width": int(img.get("width", 512)),
            "height": int(img.get("height", 512)),
        }
        for img in coco.get("images", [])
    }

    by_file = {}
    for image_id, meta in id_to_image.items():
        by_file[meta["file_name"]] = {
            "width": meta["width"],
            "height": meta["height"],
            "segmentations": [],
        }

    for ann in coco.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        seg = ann.get("segmentation", [])
        if image_id not in id_to_image:
            continue
        file_name = id_to_image[image_id]["file_name"]
        if isinstance(seg, list):
            by_file[file_name]["segmentations"].extend(seg)

    return by_file


def render_mask_from_coco_polygons(segmentation_item: dict):
    h = int(segmentation_item["height"])
    w = int(segmentation_item["width"])
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for poly in segmentation_item["segmentations"]:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        points = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
        draw.polygon(points, fill=255)

    return mask


class SyntaxArteryTargetDataset(Dataset):
    def __init__(self, target_csv: str, syntax_root: str, split: str, img_size: int = 256):
        self.rows = []
        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
        base = Path(syntax_root)

        with open(target_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                image_path = base / split / "images" / row["file_name"]
                if image_path.is_file():
                    self.rows.append(
                        {
                            "image_path": str(image_path),
                            "label": int(row["target_main_artery_id"]),
                            "patient_number": row.get("patient_number", ""),
                            "file_name": row["file_name"],
                        }
                    )

        if not self.rows:
            raise RuntimeError("No rows found for selected split/path.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("L")
        image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label, row["patient_number"], row["file_name"]


class BackboneClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=False,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )
        in_features = getattr(self.encoder, "num_features")
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats)


def compute_saliency(model: nn.Module, image_tensor: torch.Tensor, class_index: int):
    x = image_tensor.unsqueeze(0)
    x.requires_grad_(True)

    logits = model(x)
    score = logits[0, class_index]
    model.zero_grad(set_to_none=True)
    score.backward()

    sal = x.grad.detach().abs()[0, 0]
    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8) + 2
    return sal.cpu().numpy(), logits.detach()[0]


def save_panel(
    gray_img: np.ndarray,
    saliency: np.ndarray,
    gt_mask: np.ndarray,
    out_path: Path,
    title: str,
    saliency_alpha: float = 0.85,
    mask_alpha: float = 0.9,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=140)
    fig.suptitle(title, fontsize=10)

    axes[0].imshow(gray_img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Data")
    axes[0].axis("off")

    axes[1].imshow(gray_img, cmap="gray", vmin=0.0, vmax=1.0)
    # Red saliency overlay.
    axes[1].imshow(saliency, cmap="Reds", alpha=np.clip(saliency, 0.0, 1.0) * saliency_alpha)
    # Green segmentation GT overlay.
    green = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 4), dtype=np.float32)
    green[..., 1] = 1.0
    green[..., 3] = np.clip(gt_mask, 0.0, 1.0) * mask_alpha
    axes[1].imshow(green)
    axes[1].set_title("Saliency + GT (green)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Save data|saliency-overlay panels per patient for main-artery classifier.")
    parser.add_argument("--target-csv", default="results/arcade_patient_tables/patient_main_artery_targets.csv")
    parser.add_argument("--syntax-root", default="data/ARCADE/Unprocessed/arcade/syntax")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="checkpoints/syntax_main_artery_classifier/best_model.pth")
    parser.add_argument("--backbone", default="convnextv2_pico")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--use-pred-class", action="store_true", help="Backprop using predicted class instead of GT class.")
    parser.add_argument("--saliency-alpha", type=float, default=1)
    parser.add_argument("--mask-alpha", type=float, default=0.1)
    parser.add_argument("--mask-dilate", type=int, default=1, help="How many 3x3 max-filter dilations for GT visibility.")
    parser.add_argument("--out-dir", default="results/saliency/syntax_main_artery_panels")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SyntaxArteryTargetDataset(
        target_csv=args.target_csv,
        syntax_root=args.syntax_root,
        split=args.split,
        img_size=args.img_size,
    )

    model = BackboneClassifier(backbone_name=args.backbone, num_classes=args.num_classes).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    coco_index = build_coco_segmentation_index(args.syntax_root, args.split)

    n = min(args.max_samples, len(ds))
    print(f"Saving {n} saliency panels to {out_dir}")

    with torch.enable_grad():
        for i in range(n):
            image, label, patient_number, file_name = ds[i]
            image = image.to(device)

            saliency, logits = compute_saliency(model, image, class_index=label.item())
            pred = int(torch.argmax(logits).item())
            target_for_grad = label.item()

            if args.use_pred_class and pred != target_for_grad:
                saliency, logits = compute_saliency(model, image, class_index=pred)
                target_for_grad = pred

            gray = image.detach().cpu().numpy()[0]
            seg_item = coco_index.get(file_name)
            if seg_item is None:
                gt_mask = np.zeros_like(gray, dtype=np.float32)
            else:
                mask_img = render_mask_from_coco_polygons(seg_item)
                # Optional dilation to make thin vessels visible in overlay figures.
                for _ in range(max(0, int(args.mask_dilate))):
                    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=3))
                mask_img = mask_img.resize((args.img_size, args.img_size), resample=Image.NEAREST)
                gt_mask = (np.array(mask_img, dtype=np.float32) > 0).astype(np.float32)

            gt_name = ID_TO_NAME.get(label.item(), str(label.item()))
            pred_name = ID_TO_NAME.get(pred, str(pred))
            grad_name = ID_TO_NAME.get(target_for_grad, str(target_for_grad))

            title = (
                f"patient={patient_number} file={file_name} | "
                f"GT={gt_name} PRED={pred_name} | grad_target={grad_name}"
            )
            out_path = out_dir / f"patient_{patient_number}_{Path(file_name).stem}_panel.png"
            save_panel(
                gray,
                saliency,
                gt_mask,
                out_path,
                title,
                saliency_alpha=float(args.saliency_alpha),
                mask_alpha=float(args.mask_alpha),
            )

    print("Done.")


if __name__ == "__main__":
    main()