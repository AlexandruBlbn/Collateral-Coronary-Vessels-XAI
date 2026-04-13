import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

from trainv3 import VesselNetV2, VesselNetV3EfficientNet


RCA_LABELS = {"1", "2", "3", "4", "16", "16a", "16b", "16c"}
LCX_LABELS = {"11", "12", "13", "14", "14a", "14b", "15"}

METRIC_FIELDS = [
	"gt_area_frac",
	"cam_mean_in",
	"cam_mean_out",
	"cam_in_out_ratio",
	"cam_energy_in",
	"topk_overlap",
	"pointing_hit",
	"pred_dice",
	"pred_iou",
]


def parse_args():
	parser = argparse.ArgumentParser(description="Encoder saliency maps on 3 RCA + 3 LCA test samples.")
	parser.add_argument("--checkpoint", type=str, default="/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/trainv2_dcn_subpixel_multitask/best_model.pth")
	parser.add_argument("--model-name", type=str, default="auto", choices=["auto", "VesselNetV2", "VesselNetV3EfficientNet"])
	parser.add_argument("--encoder-name", type=str, default="efficientnetv2_s")
	parser.add_argument("--encoder-pretrained", action="store_true")
	parser.add_argument("--drop-path-rate", type=float, default=0.2)
	parser.add_argument("--img-size", type=int, default=512)
	parser.add_argument("--in-chans", type=int, default=4)
	parser.add_argument("--num-classes", type=int, default=1)
	parser.add_argument("--num-per-class", type=int, default=3)
	parser.add_argument("--pred-threshold", type=float, default=0.45)
	parser.add_argument("--split", type=str, default="test")
	parser.add_argument("--target-csv", type=str, default="results/arcade_patient_tables/patient_main_artery_targets.csv")
	parser.add_argument("--syntax-root", type=str, default="data/ARCADE/Unprocessed/arcade/syntax")
	parser.add_argument("--output-dir", type=str, default="results/saliency/trainv2_encoder")
	parser.add_argument("--topk-frac", type=float, default=0.10)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def _extract_state_dict(ckpt_obj):
	if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
		return ckpt_obj["model_state_dict"]
	if isinstance(ckpt_obj, dict):
		return ckpt_obj
	raise RuntimeError("Checkpoint format not understood. Expected dict or dict with model_state_dict.")


def _infer_model_name_from_state(state_dict: Dict[str, torch.Tensor]) -> str:
	for key in state_dict.keys():
		if key.startswith("encoder."):
			return "VesselNetV3EfficientNet"
	return "VesselNetV2"


def _build_model(args, model_name: str):
	if model_name == "VesselNetV3EfficientNet":
		return VesselNetV3EfficientNet(
			in_chans=args.in_chans,
			num_classes=args.num_classes,
			encoder_name=args.encoder_name,
			encoder_pretrained=args.encoder_pretrained,
			drop_path_rate=args.drop_path_rate,
		)
	if model_name == "VesselNetV2":
		return VesselNetV2(
			in_chans=args.in_chans,
			num_classes=args.num_classes,
			dims=(48, 96, 192, 384),
			depths=(2, 2, 2, 2),
			drop_path_rate=args.drop_path_rate,
		)
	raise ValueError(f"Unsupported model name: {model_name}")


def _choose_samples(target_csv: Path, syntax_root: Path, split: str, n_per_class: int, seed: int):
	rca_rows: List[Dict] = []
	lca_rows: List[Dict] = []

	with open(target_csv, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if str(row.get("split", "")).strip() != split:
				continue
			file_name = str(row.get("file_name", "")).strip()
			if not file_name:
				continue
			image_path = syntax_root / split / "images" / file_name
			if not image_path.is_file():
				continue

			try:
				target_id = int(row.get("target_main_artery_id", -1))
			except Exception:
				continue

			entry = {
				"file_name": file_name,
				"image_path": image_path,
				"target_id": target_id,
				"target_name": "RCA" if target_id == 0 else "LCA" if target_id == 1 else "UNKNOWN",
				"patient_number": str(row.get("patient_number", "")),
			}
			if target_id == 0:
				rca_rows.append(entry)
			elif target_id == 1:
				lca_rows.append(entry)

	rng = random.Random(seed)
	rng.shuffle(rca_rows)
	rng.shuffle(lca_rows)

	if len(rca_rows) < n_per_class or len(lca_rows) < n_per_class:
		raise RuntimeError(
			f"Not enough samples in split={split}. Found RCA={len(rca_rows)}, LCA={len(lca_rows)}, requested {n_per_class}."
		)

	return rca_rows[:n_per_class] + lca_rows[:n_per_class]


def _load_split_index(syntax_root: Path, split: str):
	ann_path = syntax_root / split / "annotations" / f"{split}.json"
	if not ann_path.is_file():
		raise FileNotFoundError(f"COCO annotation not found: {ann_path}")

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


def _rasterize_polygons(width: int, height: int, polygons) -> np.ndarray:
	mask = Image.new("L", (width, height), 0)
	draw = ImageDraw.Draw(mask)
	for poly in polygons:
		points = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
		draw.polygon(points, fill=255)
	return (np.array(mask, dtype=np.uint8) > 0).astype(np.float32)


def _build_four_channels(image_path: Path, img_size: int, clahe, morph_kernel):
	image = Image.open(image_path).convert("L")
	image = image.resize((img_size, img_size), resample=Image.BILINEAR)
	image_np = np.array(image, dtype=np.uint8)

	c1 = clahe.apply(image_np)
	c2 = cv2.morphologyEx(image_np, cv2.MORPH_TOPHAT, morph_kernel)
	c3 = cv2.morphologyEx(image_np, cv2.MORPH_BLACKHAT, morph_kernel)
	blurred = cv2.GaussianBlur(image_np, (0, 0), sigmaX=10)
	c4 = cv2.addWeighted(image_np, 4.0, blurred, -4.0, 128)

	stacked = np.stack([c1, c2, c3, c4], axis=-1)
	tensor = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0
	return tensor, image_np


class _ActivationTap:
	def __init__(self, model):
		self.activation = None
		if hasattr(model, "encoder"):
			self.handle = model.encoder.register_forward_hook(self._hook_encoder)
		elif hasattr(model, "stage4"):
			self.handle = model.stage4.register_forward_hook(self._hook_tensor)
		else:
			raise RuntimeError("Could not find encoder/stage4 module for saliency tapping.")

	def _hook_encoder(self, module, inputs, output):
		if isinstance(output, (list, tuple)):
			act = output[-1]
		else:
			act = output
		self.activation = act
		if isinstance(self.activation, torch.Tensor):
			self.activation.retain_grad()

	def _hook_tensor(self, module, inputs, output):
		self.activation = output
		if isinstance(self.activation, torch.Tensor):
			self.activation.retain_grad()

	def close(self):
		self.handle.remove()


def _normalize_map(x: np.ndarray) -> np.ndarray:
	x = x.astype(np.float32)
	x = x - float(x.min())
	den = float(x.max())
	if den <= 1e-12:
		return np.zeros_like(x, dtype=np.float32)
	return x / den


def _safe_mean(values: List[float]) -> float:
	if not values:
		return float("nan")
	return float(np.mean(values))


def _safe_std(values: List[float]) -> float:
	if not values:
		return float("nan")
	return float(np.std(values))


def _compute_binary_metrics(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-8) -> Tuple[float, float]:
	pred = pred01 > 0.5
	gt = gt01 > 0.5
	tp = float(np.logical_and(pred, gt).sum())
	fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
	fn = float(np.logical_and(np.logical_not(pred), gt).sum())
	dice = (2.0 * tp) / max(eps, (2.0 * tp + fp + fn))
	iou = tp / max(eps, (tp + fp + fn))
	return float(dice), float(iou)


def _compute_saliency_metrics(cam01: np.ndarray, gt01: np.ndarray, pred01: np.ndarray, topk_frac: float) -> Dict[str, float]:
	eps = 1e-8
	cam = np.clip(cam01.astype(np.float32), 0.0, 1.0)
	gt = (gt01 > 0.5).astype(np.float32)

	inside = cam[gt > 0.5]
	outside = cam[gt <= 0.5]
	cam_mean_in = float(inside.mean()) if inside.size > 0 else 0.0
	cam_mean_out = float(outside.mean()) if outside.size > 0 else 0.0
	cam_in_out_ratio = cam_mean_in / max(eps, cam_mean_out)

	cam_energy_in = float((cam * gt).sum() / max(eps, cam.sum()))

	flat_cam = cam.reshape(-1)
	flat_gt = gt.reshape(-1)
	k = int(max(1, min(flat_cam.size, round(float(topk_frac) * flat_cam.size))))
	top_idx = np.argpartition(-flat_cam, k - 1)[:k]
	topk_overlap = float(flat_gt[top_idx].mean())
	pointing_hit = float(flat_gt[int(np.argmax(flat_cam))] > 0.5)

	pred_dice, pred_iou = _compute_binary_metrics(pred01=pred01, gt01=gt)

	return {
		"gt_area_frac": float(gt.mean()),
		"cam_mean_in": cam_mean_in,
		"cam_mean_out": cam_mean_out,
		"cam_in_out_ratio": float(cam_in_out_ratio),
		"cam_energy_in": cam_energy_in,
		"topk_overlap": topk_overlap,
		"pointing_hit": pointing_hit,
		"pred_dice": pred_dice,
		"pred_iou": pred_iou,
	}


def _make_panel(gray_u8: np.ndarray, cam01: np.ndarray, pred01: np.ndarray, title: str):
	base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
	heat = np.clip(cam01 * 255.0, 0, 255).astype(np.uint8)
	heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
	overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0.0)

	pred_u8 = np.clip(pred01 * 255.0, 0, 255).astype(np.uint8)
	pred_bgr = cv2.cvtColor(pred_u8, cv2.COLOR_GRAY2BGR)

	panel = np.concatenate([base, overlay, pred_bgr], axis=1)
	cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(panel, "Input", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
	cv2.putText(panel, "Encoder Grad-CAM", (base.shape[1] + 8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
	cv2.putText(panel, "Prediction", (2 * base.shape[1] + 8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
	return panel


def main():
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	checkpoint_path = Path(args.checkpoint)
	if not checkpoint_path.is_file():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	ckpt = torch.load(checkpoint_path, map_location="cpu")
	state_dict = _extract_state_dict(ckpt)

	if args.model_name == "auto":
		model_name = _infer_model_name_from_state(state_dict)
	else:
		model_name = args.model_name

	model = _build_model(args, model_name=model_name).to(device)
	model.load_state_dict(state_dict, strict=True)
	model.eval()

	target_csv = Path(args.target_csv)
	syntax_root = Path(args.syntax_root)
	split_index = _load_split_index(syntax_root=syntax_root, split=args.split)
	selected = _choose_samples(
		target_csv=target_csv,
		syntax_root=syntax_root,
		split=args.split,
		n_per_class=args.num_per_class,
		seed=args.seed,
	)

	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

	tap = _ActivationTap(model)
	rows = []
	for i, item in enumerate(selected, start=1):
		file_name = item["file_name"]
		if file_name not in split_index:
			raise RuntimeError(f"No annotation metadata for selected file: {file_name}")

		meta = split_index[file_name]
		polygons = meta["rca_polygons"] if int(item["target_id"]) == 0 else meta["lca_polygons"]
		gt_mask = _rasterize_polygons(meta["width"], meta["height"], polygons)
		gt_mask = cv2.resize(gt_mask, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)

		x, gray = _build_four_channels(item["image_path"], args.img_size, clahe, morph_kernel)
		x = x.unsqueeze(0).to(device)

		model.zero_grad(set_to_none=True)
		outputs = model(x)
		seg_logits = outputs["seg_logits"]
		score = torch.sigmoid(seg_logits).mean()
		score.backward()

		if tap.activation is None or tap.activation.grad is None:
			raise RuntimeError("Failed to capture activation/gradient for saliency.")

		act = tap.activation
		grad = tap.activation.grad
		weights = torch.mean(grad, dim=(2, 3), keepdim=True)
		cam = torch.relu(torch.sum(weights * act, dim=1, keepdim=True))
		cam = F.interpolate(cam, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
		cam_np = cam[0, 0].detach().cpu().numpy()
		cam_np = _normalize_map(cam_np)

		pred = (torch.sigmoid(seg_logits)[0, 0] > args.pred_threshold).float().detach().cpu().numpy()
		metrics = _compute_saliency_metrics(cam01=cam_np, gt01=gt_mask, pred01=pred, topk_frac=args.topk_frac)
		title = f"{item['target_name']} | file={item['file_name']} | patient={item['patient_number']}"
		panel = _make_panel(gray_u8=gray, cam01=cam_np, pred01=pred, title=title)

		out_name = f"{i:02d}_{item['target_name'].lower()}_{Path(item['file_name']).stem}.png"
		out_path = out_dir / out_name
		cv2.imwrite(str(out_path), panel)

		rows.append(
			{
				"index": i,
				"target_name": item["target_name"],
				"patient_number": item["patient_number"],
				"file_name": item["file_name"],
				"output_path": str(out_path),
				**metrics,
			}
		)

	tap.close()

	index_csv = out_dir / "selected_samples.csv"
	with open(index_csv, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["index", "target_name", "patient_number", "file_name", "output_path"],
			extrasaction="ignore",
		)
		writer.writeheader()
		writer.writerows(rows)

	metrics_csv = out_dir / "saliency_metrics.csv"
	with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["index", "target_name", "patient_number", "file_name", "output_path", *METRIC_FIELDS])
		writer.writeheader()
		for row in rows:
			writer.writerow({k: row[k] for k in ["index", "target_name", "patient_number", "file_name", "output_path", *METRIC_FIELDS]})

	summary_rows = []
	groups = {
		"ALL": rows,
		"RCA": [r for r in rows if r["target_name"] == "RCA"],
		"LCA": [r for r in rows if r["target_name"] == "LCA"],
	}
	for group_name, group_rows in groups.items():
		if not group_rows:
			continue
		summary = {"group": group_name, "n": len(group_rows)}
		for m in METRIC_FIELDS:
			vals = [float(r[m]) for r in group_rows]
			summary[f"{m}_mean"] = _safe_mean(vals)
			summary[f"{m}_std"] = _safe_std(vals)
		summary_rows.append(summary)

	summary_csv = out_dir / "saliency_summary.csv"
	if summary_rows:
		summary_fields = ["group", "n"] + [f"{m}_mean" for m in METRIC_FIELDS] + [f"{m}_std" for m in METRIC_FIELDS]
		with open(summary_csv, "w", encoding="utf-8", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=summary_fields)
			writer.writeheader()
			writer.writerows(summary_rows)

	print(f"Saved {len(rows)} saliency panels in: {out_dir}")
	print(f"Selection index saved to: {index_csv}")
	print(f"Per-sample saliency metrics saved to: {metrics_csv}")
	print(f"Aggregate saliency summary saved to: {summary_csv}")


if __name__ == "__main__":
	main()