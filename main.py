import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
from PIL import Image
from skimage.filters import frangi


class LeJepaCropper:
	"""LeJepa-style multi-crop logic used for visualization."""

	def __init__(
		self,
		img_size: int = 256,
		num_global_crops: int = 2,
		num_local_crops: int = 4,
		local_threshold: float = 0.02,
		background_threshold: float = 0.01,
		global_vessel_threshold: float = 0.015,
		max_global_retries: int = 12,
		max_local_retries: int = 20,
	):
		self.img_size = img_size
		self.border_jitter_size = int(img_size * 0.88)
		self.global_scale = (0.8, 1.0)
		self.local_scale = (0.10, 0.35)
		self.crop_ratio = (3.0 / 4.0, 4.0 / 3.0)
		self.num_global_crops = num_global_crops
		self.num_local_crops = num_local_crops
		self.local_vessel_fraction = 0.75
		self.global_vessel_threshold = global_vessel_threshold
		self.max_global_retries = max_global_retries
		self.local_threshold = local_threshold
		self.background_threshold = background_threshold
		self.max_local_retries = max_local_retries

	@staticmethod
	def _fill_fov_border(img: torch.Tensor) -> torch.Tensor:
		img = img.clone()
		fill_val = img.mean()

		border_mask = img < -0.85
		if border_mask.float().mean() >= 0.01:
			img[border_mask] = fill_val

		h = img.shape[-2]
		fringe = max(1, int(h * 0.08))
		top_band = img[..., :fringe, :]
		bottom_band = img[..., -fringe:, :]
		if (top_band > 0.7).float().mean() > 0.35:
			img[..., :fringe, :] = fill_val
		if (bottom_band > 0.7).float().mean() > 0.35:
			img[..., -fringe:, :] = fill_val

		return img

	def _apply_border_jitter(self, img: torch.Tensor, vesselness: torch.Tensor):
		i, j, h, w = transforms.RandomCrop.get_params(
			img,
			output_size=(self.border_jitter_size, self.border_jitter_size),
		)
		img_crop = TF.crop(img, i, j, h, w)
		vessel_crop = TF.crop(vesselness, i, j, h, w)
		img_crop = TF.resize(
			img_crop,
			[self.img_size, self.img_size],
			interpolation=transforms.InterpolationMode.BICUBIC,
			antialias=True,
		)
		vessel_crop = TF.resize(
			vessel_crop,
			[self.img_size, self.img_size],
			interpolation=transforms.InterpolationMode.BILINEAR,
			antialias=True,
		)
		return img_crop, vessel_crop

	def _sample_resized_crop_params(self, img: torch.Tensor, scale):
		return transforms.RandomResizedCrop.get_params(img, scale=scale, ratio=self.crop_ratio)

	def _crop_and_resize(self, img: torch.Tensor, params, interpolation=transforms.InterpolationMode.BICUBIC):
		i, j, h, w = params
		crop = TF.crop(img, i, j, h, w)
		return TF.resize(
			crop,
			[self.img_size, self.img_size],
			interpolation=interpolation,
			antialias=True,
		)

	def _crop_vessel_score(self, vesselness: torch.Tensor, params) -> float:
		i, j, h, w = params
		crop = vesselness[..., i : i + h, j : j + w]
		return float(crop.mean().item())

	def _guided_crop(
		self,
		img: torch.Tensor,
		vesselness: torch.Tensor,
		mode: str,
		scale,
		threshold: float,
		max_retries: int,
	):
		if mode not in ("vessel", "background"):
			raise ValueError(f"Unknown local crop mode: {mode}")

		best_params = None
		best_score = -1.0 if mode == "vessel" else float("inf")
		accepted = False
		tries_used = 0

		for attempt in range(max_retries):
			params = self._sample_resized_crop_params(img, scale)
			score = self._crop_vessel_score(vesselness, params)
			tries_used = attempt + 1

			if mode == "vessel":
				if score > best_score:
					best_score = score
					best_params = params
				if score >= threshold:
					accepted = True
					best_params = params
					best_score = score
					break
			else:
				if score < best_score:
					best_score = score
					best_params = params
				if score <= threshold:
					accepted = True
					best_params = params
					best_score = score
					break

		crop = self._crop_and_resize(img, best_params)
		return crop, best_params, best_score, accepted, tries_used

	def __call__(self, img: torch.Tensor, vesselness: torch.Tensor):
		img = self._fill_fov_border(img)
		vesselness = vesselness.clamp(0.0, 1.0)
		crops = []
		crop_stats = []
		global_pairs = []
		for g_idx in range(self.num_global_crops):
			jitter_img, jitter_vessel = self._apply_border_jitter(img, vesselness)
			global_img, g_params, g_score, g_accepted, g_tries = self._guided_crop(
				jitter_img,
				jitter_vessel,
				mode="vessel",
				scale=self.global_scale,
				threshold=self.global_vessel_threshold,
				max_retries=self.max_global_retries,
			)
			global_vessel = self._crop_and_resize(
				jitter_vessel,
				g_params,
				interpolation=transforms.InterpolationMode.BILINEAR,
			)
			global_vessel = global_vessel.clamp(0.0, 1.0)
			global_pairs.append((global_img, global_vessel))
			crops.append(global_img)
			crop_stats.append(
				{
					"kind": "global",
					"mode": "vessel",
					"score": g_score,
					"accepted": g_accepted,
					"tries": g_tries,
					"parent_global": g_idx,
				}
			)

		n_vessel = int(round(self.num_local_crops * self.local_vessel_fraction))
		n_vessel = min(self.num_local_crops, max(0, n_vessel))
		n_background = self.num_local_crops - n_vessel
		local_modes = ["vessel"] * n_vessel + ["background"] * n_background
		random.shuffle(local_modes)

		# Local crops are explicitly sampled INSIDE one of the existing global crops.
		for local_mode in local_modes:
			parent_idx = random.randrange(len(global_pairs))
			parent_img, parent_vessel = global_pairs[parent_idx]
			threshold = self.local_threshold if local_mode == "vessel" else self.background_threshold
			crop, _, score, accepted, tries = self._guided_crop(
				parent_img,
				parent_vessel,
				mode=local_mode,
				scale=self.local_scale,
				threshold=threshold,
				max_retries=self.max_local_retries,
			)
			crops.append(crop)
			crop_stats.append(
				{
					"kind": "local-guided",
					"mode": local_mode,
					"score": score,
					"accepted": accepted,
					"tries": tries,
					"parent_global": parent_idx,
				}
			)
		return crops, crop_stats


def _resolve_image_path(workspace_root: Path, json_path: Path, rel_path: str) -> Path:
	candidates = [
		workspace_root / rel_path,
		json_path.parent / rel_path,
		json_path.parent.parent / rel_path,
	]
	for p in candidates:
		if p.exists():
			return p
	raise FileNotFoundError(f"Could not resolve image path '{rel_path}' from dataset json.")


def _collect_patients(dataset_json: Path, split: str = "train"):
	with dataset_json.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if split not in data:
		raise ValueError(f"Split '{split}' not found in {dataset_json}.")

	entries = []
	for source_name, source_data in data[split].items():
		if not isinstance(source_data, dict):
			continue
		for patient_id, sample in source_data.items():
			image_rel = sample.get("data")
			if isinstance(image_rel, str) and image_rel:
				entries.append(
					{
						"patient_id": patient_id,
						"source": source_name,
						"image_rel": image_rel,
					}
				)

	if not entries:
		raise RuntimeError("No patients with valid image paths were found.")

	return entries


def _to_tensor_for_cropper(img_pil: Image.Image) -> torch.Tensor:
	arr = np.array(img_pil.convert("L"), dtype=np.float32) / 255.0
	tensor = torch.from_numpy(arr).unsqueeze(0)
	tensor = (tensor - 0.5) / 0.5
	return tensor


def _to_numpy_display(img_tensor: torch.Tensor) -> np.ndarray:
	arr = img_tensor.squeeze(0).detach().cpu().numpy()
	arr = (arr * 0.5) + 0.5
	return np.clip(arr, 0.0, 1.0)


def _largest_component_mask(mask_u8: np.ndarray) -> np.ndarray:
	n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
	if n_labels <= 1:
		return mask_u8
	areas = stats[1:, cv2.CC_STAT_AREA]
	best = 1 + int(np.argmax(areas))
	out = np.zeros_like(mask_u8, dtype=np.uint8)
	out[labels == best] = 255
	return out


def _filter_border_components(
	binary_mask: np.ndarray,
	fov_mask: np.ndarray,
	border_px: int = 2,
	max_touch_ratio: float = 0.35,
	min_area: int = 12,
) -> np.ndarray:
	"""Remove components dominated by border pixels while preserving edge-crossing vessels."""
	bm = (binary_mask > 0).astype(np.uint8)
	n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bm, connectivity=8)
	if n_labels <= 1:
		return bm.astype(np.float32)

	h, w = bm.shape
	img_border = np.zeros_like(bm, dtype=bool)
	img_border[:border_px, :] = True
	img_border[-border_px:, :] = True
	img_border[:, :border_px] = True
	img_border[:, -border_px:] = True

	fov_u8 = (fov_mask > 0.5).astype(np.uint8) * 255
	eroded = cv2.erode(fov_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
	fov_ring = (fov_u8 > 0) & (eroded == 0)

	keep = np.zeros_like(bm, dtype=np.uint8)
	for label_id in range(1, n_labels):
		area = int(stats[label_id, cv2.CC_STAT_AREA])
		if area < min_area:
			continue

		component = labels == label_id
		touch_img = np.count_nonzero(component & img_border)
		touch_ring = np.count_nonzero(component & fov_ring)

		img_ratio = touch_img / (area + 1e-8)
		ring_ratio = touch_ring / (area + 1e-8)

		# Keep elongated vessel components that only partially touch edges;
		# drop components mostly explained by border/frame transitions.
		if img_ratio > max_touch_ratio:
			continue
		if ring_ratio > 0.85 and area < 180:
			continue

		keep[component] = 1

	return keep.astype(np.float32)


def _valid_fov_mask(img_u8: np.ndarray, img_size: int) -> np.ndarray:
	"""Build a robust mask of valid imaging area to suppress border/frame artefacts."""
	resized = cv2.resize(img_u8, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
	blur = cv2.GaussianBlur(resized, (9, 9), 0)

	# Most edge artefacts are near black outside the circular FOV.
	_, mask = cv2.threshold(blur, 8, 255, cv2.THRESH_BINARY)
	mask = _largest_component_mask(mask)

	k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

	# Gentle erosion to reduce border gradients without removing near-edge vessels.
	k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	mask = cv2.erode(mask, k_erode, iterations=1)
	return (mask > 0).astype(np.float32)


def _fov_boundary_weight(fov_mask: np.ndarray, fade_px: float = 14.0) -> np.ndarray:
	"""Create a soft weight: 0 at FOV boundary, 1 deeper inside the valid region."""
	mask_u8 = (fov_mask > 0.5).astype(np.uint8) * 255
	dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3).astype(np.float32)
	weight = np.clip(dist / max(fade_px, 1.0), 0.0, 1.0)
	return weight * fov_mask


def _frame_line_suppression_weight(img_u8: np.ndarray, img_size: int) -> np.ndarray:
	"""Detect long horizontal/vertical frame lines and down-weight Frangi there."""
	resized = cv2.resize(img_u8, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
	blur = cv2.GaussianBlur(resized, (5, 5), 0)
	edges = cv2.Canny(blur, threshold1=40, threshold2=120)

	line_mask = np.zeros_like(edges, dtype=np.uint8)
	lines = cv2.HoughLinesP(
		edges,
		rho=1,
		theta=np.pi / 180.0,
		threshold=40,
		minLineLength=int(0.35 * img_size),
		maxLineGap=6,
	)

	if lines is not None:
		for l in lines[:, 0, :]:
			x1, y1, x2, y2 = [int(v) for v in l]
			dx = x2 - x1
			dy = y2 - y1
			angle = abs(np.degrees(np.arctan2(dy, dx + 1e-8)))
			# Frame edges are usually close to horizontal or vertical.
			if angle < 12.0 or angle > 78.0:
				cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=2)

	# Dilate to cover the full bright/dark border band around detected frame lines.
	line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
	line_mask_f = (line_mask > 0).astype(np.float32)

	# Soft suppression: keep 20% response on detected frame lines, full response elsewhere.
	weight = 1.0 - 0.8 * line_mask_f
	return np.clip(weight, 0.2, 1.0)


def _frangi_response_from_pil(img_pil: Image.Image, img_size: int) -> np.ndarray:
	img_np = np.array(img_pil.convert("L"), dtype=np.uint8)
	fov_mask = _valid_fov_mask(img_np, img_size)
	fov_weight = _fov_boundary_weight(fov_mask, fade_px=14.0)
	frame_weight = _frame_line_suppression_weight(img_np, img_size)
	img_np = cv2.GaussianBlur(img_np, (7, 7), sigmaX=3)
	pad = 16
	if img_np.shape[0] > 2 * pad and img_np.shape[1] > 2 * pad:
		img_np = img_np[pad:-pad, pad:-pad]
	img_np = cv2.resize(img_np, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

	# Fill outside FOV with median inside FOV to avoid edge gradients before Frangi.
	inside_vals = img_np[fov_mask > 0.5]
	fill = float(np.median(inside_vals)) if inside_vals.size else float(np.median(img_np))
	img_np = img_np.astype(np.float32)
	img_np[fov_mask <= 0.5] = fill

	img_f = img_np
	img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8)

	vesselness = frangi(
		img_f,
		sigmas=range(1, 16, 2),
		alpha=0.5,
		beta=1,
		gamma=10,
		mode="reflect",
		black_ridges=True,
	)

	p_low, p_high = np.percentile(vesselness, (1, 99.9))
	vesselness_clip = np.clip(vesselness, p_low, p_high)
	v_norm = (vesselness_clip - p_low) / (p_high - p_low + 1e-8)

	vesselness_enhanced = np.power(v_norm, 0.5)
	vessel_mask = (v_norm > 0.045).astype(np.float32)
	vessel_mask = vessel_mask * fov_mask
	vessel_mask = _filter_border_components(vessel_mask, fov_mask=fov_mask, border_px=2)
	vesselness_final = vesselness_enhanced * vessel_mask * fov_weight * frame_weight
	vesselness_final = cv2.GaussianBlur(vesselness_final, (3, 3), 0)

	return np.clip(vesselness_final, 0.0, 1.0)


def main():
	parser = argparse.ArgumentParser(
		description="Plot LeJepa global/local crops for 4 random patients."
	)
	parser.add_argument(
		"--dataset_json",
		type=str,
		default="data/ARCADE/processed/dataset.json",
		help="Path to dataset json.",
	)
	parser.add_argument(
		"--split",
		type=str,
		default="train",
		help="Dataset split to sample from.",
	)
	parser.add_argument(
		"--num_patients",
		type=int,
		default=4,
		help="How many random patients to plot.",
	)
	parser.add_argument(
		"--img_size",
		type=int,
		default=256,
		help="Output crop size used by LeJepa augmentation.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for patient sampling.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="results/lejepa_crops_random_patients.png",
		help="Where to save the figure.",
	)
	parser.add_argument(
		"--output_frangi",
		type=str,
		default="results/frangi_random_patients.png",
		help="Where to save the Frangi response figure.",
	)
	parser.add_argument(
		"--local_vessel_threshold",
		type=float,
		default=0.02,
		help="Minimum mean Frangi vesselness required to accept a local crop.",
	)
	parser.add_argument(
		"--global_vessel_threshold",
		type=float,
		default=0.015,
		help="Minimum mean Frangi vesselness required to accept a global crop.",
	)
	parser.add_argument(
		"--local_num_crops",
		type=int,
		default=4,
		help="Number of local crops (default 4 gives exact 3/4 vessel + 1/4 background).",
	)
	parser.add_argument(
		"--local_background_threshold",
		type=float,
		default=0.01,
		help="Maximum mean Frangi vesselness required to accept a background local crop.",
	)
	parser.add_argument(
		"--local_max_retries",
		type=int,
		default=20,
		help="Maximum rejection-sampling retries for each local crop.",
	)
	args = parser.parse_args()

	workspace_root = Path(__file__).resolve().parent
	dataset_json = (workspace_root / args.dataset_json).resolve()
	if not dataset_json.exists():
		raise FileNotFoundError(f"Dataset json not found: {dataset_json}")

	patients = _collect_patients(dataset_json=dataset_json, split=args.split)
	if len(patients) < args.num_patients:
		raise ValueError(
			f"Requested {args.num_patients} patients, but split '{args.split}' has only {len(patients)} entries."
		)

	random.seed(args.seed)
	sampled = random.sample(patients, args.num_patients)

	cropper = LeJepaCropper(
		img_size=args.img_size,
		num_local_crops=args.local_num_crops,
		local_threshold=args.local_vessel_threshold,
		background_threshold=args.local_background_threshold,
		global_vessel_threshold=args.global_vessel_threshold,
		max_local_retries=args.local_max_retries,
	)

	sample_records = []
	for sample in sampled:
		image_path = _resolve_image_path(workspace_root, dataset_json, sample["image_rel"])
		img_pil = Image.open(image_path).convert("L")
		img_tensor = _to_tensor_for_cropper(img_pil)
		frangi_resp = _frangi_response_from_pil(img_pil, img_size=args.img_size)
		frangi_tensor = torch.from_numpy(frangi_resp).unsqueeze(0).float()
		crops, crop_stats = cropper(img_tensor, frangi_tensor)
		sample_records.append(
			{
				"sample": sample,
				"img_tensor": img_tensor,
				"frangi_resp": frangi_resp,
				"crops": crops,
				"crop_stats": crop_stats,
			}
		)

	global_cols = [f"Global {i+1}" for i in range(cropper.num_global_crops)]
	local_cols = [f"Guided Local {i+1}" for i in range(cropper.num_local_crops)]
	cols = ["Original", "Frangi"] + global_cols + local_cols
	fig, axes = plt.subplots(args.num_patients, len(cols), figsize=(2.7 * len(cols), 3.8 * args.num_patients))
	if args.num_patients == 1:
		axes = np.expand_dims(axes, axis=0)

	for row, record in enumerate(sample_records):
		sample = record["sample"]
		images_to_plot = [record["img_tensor"], torch.from_numpy(record["frangi_resp"]).unsqueeze(0).float()] + record["crops"]
		for col, image_tensor in enumerate(images_to_plot):
			ax = axes[row, col]
			if col == 1:
				ax.imshow(image_tensor.squeeze(0).numpy(), cmap="gray", vmin=0.0, vmax=1.0)
			else:
				ax.imshow(_to_numpy_display(image_tensor), cmap="gray", vmin=0.0, vmax=1.0)
			ax.axis("off")
			if row == 0:
				ax.set_title(cols[col], fontsize=11)
			if col >= 2:
				stat = record["crop_stats"][col - 2]
				label = f"v={stat['score']:.3f}"
				if stat["kind"] == "local-guided":
					status = "ok" if stat["accepted"] else "best"
					parent = stat.get("parent_global", 0) + 1
					mode = stat.get("mode", "vessel")
					mode_short = "ves" if mode == "vessel" else "bg"
					label = f"{label} | {mode_short} | {status} | t={stat['tries']} | g={parent}"
				ax.set_xlabel(label, fontsize=8)

		axes[row, 0].set_ylabel(
			f"{sample['source']}\\nID {sample['patient_id']}",
			fontsize=10,
			rotation=0,
			labelpad=35,
			va="center",
		)

	fig.suptitle(
		f"Frangi-Guided LeJepa Cropping: {cropper.num_global_crops} Global (vessel) + "
		f"{cropper.num_local_crops} Local (3/4 vessel, 1/4 background)",
		fontsize=14,
	)
	plt.tight_layout(rect=[0, 0, 1, 0.98])

	output_path = (workspace_root / args.output).resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=200)
	print(f"Saved crop visualization to: {output_path}")

	fig_f, axes_f = plt.subplots(args.num_patients, 2, figsize=(7, 3.8 * args.num_patients))
	if args.num_patients == 1:
		axes_f = np.expand_dims(axes_f, axis=0)

	for row, record in enumerate(sample_records):
		sample = record["sample"]
		img_disp = _to_numpy_display(record["img_tensor"])
		frangi_resp = record["frangi_resp"]

		axes_f[row, 0].imshow(img_disp, cmap="gray", vmin=0.0, vmax=1.0)
		axes_f[row, 0].axis("off")
		axes_f[row, 1].imshow(frangi_resp, cmap="gray", vmin=0.0, vmax=1.0)
		axes_f[row, 1].axis("off")

		if row == 0:
			axes_f[row, 0].set_title("Original", fontsize=11)
			axes_f[row, 1].set_title("Frangi Response", fontsize=11)

		axes_f[row, 0].set_ylabel(
			f"{sample['source']}\\nID {sample['patient_id']}",
			fontsize=10,
			rotation=0,
			labelpad=35,
			va="center",
		)

	fig_f.suptitle("Frangi Vesselness on Same Random Patients", fontsize=14)
	plt.tight_layout(rect=[0, 0, 1, 0.98])

	output_frangi_path = (workspace_root / args.output_frangi).resolve()
	output_frangi_path.parent.mkdir(parents=True, exist_ok=True)
	fig_f.savefig(output_frangi_path, dpi=200)
	print(f"Saved Frangi visualization to: {output_frangi_path}")


if __name__ == "__main__":
	main()
