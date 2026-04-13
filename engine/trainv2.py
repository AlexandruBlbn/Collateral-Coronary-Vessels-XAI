import os
import sys
import csv
import yaml
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms.functional as TF

try:
	import timm
except Exception:
	timm = None

try:
	import torchvision.ops as ops
	HAS_DEFORM_CONV = hasattr(ops, "DeformConv2d")
except Exception:
	ops = None
	HAS_DEFORM_CONV = False


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import set_seed


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configCreate(path, config):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		yaml.dump(config, f, sort_keys=False)


def _resolve_split_key(data: Dict, split: str) -> str:
	candidates = [split]
	if split == "val":
		candidates.append("validation")
	if split == "validation":
		candidates.append("val")
	for key in candidates:
		if key in data:
			return key
	raise ValueError(f"Split '{split}' not found in dataset keys: {list(data.keys())}")


def _signed_distance_map(mask01: np.ndarray) -> np.ndarray:
	mask_u8 = (mask01 > 0).astype(np.uint8)
	if mask_u8.max() == 0:
		return np.full(mask_u8.shape, -1.0, dtype=np.float32)

	pos_dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
	neg_dist = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 5)
	sdm = pos_dist - neg_dist
	max_abs = float(np.max(np.abs(sdm)))
	if max_abs > 0:
		sdm = sdm / max_abs
	return sdm.astype(np.float32)


class VesselSegmentationDatasetV2(Dataset):
	def __init__(
		self,
		json_path: str,
		split: str = "train",
		source: str = "syntax",
		img_size: int = 512,
		mode: str = "train",
		root_dir: str = ".",
	):
		self.json_path = json_path
		self.split = split
		self.source = source
		self.img_size = int(img_size)
		self.mode = mode
		self.root_dir = Path(root_dir)

		self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

		with open(self.json_path, "r", encoding="utf-8") as f:
			data = yaml.safe_load(f)

		split_key = _resolve_split_key(data, split)
		split_data = data[split_key]
		if self.source not in split_data:
			raise ValueError(
				f"Source '{self.source}' not found in split '{split_key}'. Available: {list(split_data.keys())}"
			)

		self.samples: List[Dict[str, str]] = []
		source_data = split_data[self.source]
		for sample_id, sample_info in source_data.items():
			image_path = sample_info.get("data")
			label_path = sample_info.get("label")
			if not image_path:
				continue
			self.samples.append(
				{
					"id": str(sample_id),
					"image_path": str(image_path),
					"label_path": str(label_path) if label_path else "",
					"file_name": Path(str(image_path)).name,
				}
			)

		if not self.samples:
			raise RuntimeError(f"No samples loaded for split={split}, source={source}.")

	def __len__(self):
		return len(self.samples)

	def _resolve_path(self, path_str: str) -> Path:
		p = Path(path_str)
		if p.is_absolute():
			return p
		return self.root_dir / p

	def _apply_geometric_aug(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
		if torch.rand(1).item() > 0.5:
			image = TF.hflip(image)
			mask = TF.hflip(mask)
		if torch.rand(1).item() > 0.5:
			image = TF.vflip(image)
			mask = TF.vflip(mask)

		if torch.rand(1).item() > 0.25:
			angle = float(torch.empty(1).uniform_(-25.0, 25.0).item())
			scale = float(torch.empty(1).uniform_(0.85, 1.15).item())
			tx = int(torch.empty(1).uniform_(-0.1, 0.1).item() * self.img_size)
			ty = int(torch.empty(1).uniform_(-0.1, 0.1).item() * self.img_size)

			image = TF.affine(
				image,
				angle=angle,
				translate=[tx, ty],
				scale=scale,
				shear=[0.0, 0.0],
				interpolation=TF.InterpolationMode.BILINEAR,
			)
			mask = TF.affine(
				mask,
				angle=angle,
				translate=[tx, ty],
				scale=scale,
				shear=[0.0, 0.0],
				interpolation=TF.InterpolationMode.NEAREST,
			)

		return image, mask

	def _inject_hard_negative_artifacts(self, image_np: np.ndarray):
		if torch.rand(1).item() < 0.30:
			num_lines = int(torch.randint(1, 4, (1,)).item())
			for _ in range(num_lines):
				x1 = int(torch.randint(0, self.img_size, (1,)).item())
				y1 = int(torch.randint(0, self.img_size, (1,)).item())
				x2 = x1 + int(torch.randint(-150, 150, (1,)).item())
				y2 = y1 + int(torch.randint(-150, 150, (1,)).item())
				thickness = int(torch.randint(2, 6, (1,)).item())
				color = int(torch.randint(10, 80, (1,)).item())
				cv2.line(image_np, (x1, y1), (x2, y2), color, thickness)

		if torch.rand(1).item() < 0.20:
			cx = int(torch.randint(0, self.img_size, (1,)).item())
			cy = int(torch.randint(0, self.img_size, (1,)).item())
			radius = int(torch.randint(15, 60, (1,)).item())
			color = int(torch.randint(30, 100, (1,)).item())
			cv2.circle(image_np, (cx, cy), radius, color, -1)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		image_path = self._resolve_path(sample["image_path"])
		label_path = self._resolve_path(sample["label_path"]) if sample["label_path"] else None

		image = Image.open(image_path).convert("L")
		if label_path is not None and label_path.is_file():
			mask = Image.open(label_path).convert("L")
		else:
			mask = Image.new("L", image.size, 0)

		image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
		mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

		if self.mode == "train":
			image, mask = self._apply_geometric_aug(image, mask)

		image_np = np.array(image, dtype=np.uint8)

		if self.mode == "train":
			self._inject_hard_negative_artifacts(image_np)

			if torch.rand(1).item() > 0.65:
				sigma = float(torch.empty(1).uniform_(0.5, 1.6).item())
				image_np = cv2.GaussianBlur(image_np, (5, 5), sigmaX=sigma)

			if torch.rand(1).item() > 0.5:
				alpha = float(torch.empty(1).uniform_(0.85, 1.15).item())
				beta = float(torch.empty(1).uniform_(-12.0, 12.0).item())
				image_np = np.clip(alpha * image_np + beta, 0, 255).astype(np.uint8)

		# Channel 1: CLAHE
		c1 = self.clahe.apply(image_np)
		# Channel 2: White-hat (top-hat)
		c2 = cv2.morphologyEx(image_np, cv2.MORPH_TOPHAT, self.morph_kernel)
		# Channel 3: Black-hat
		c3 = cv2.morphologyEx(image_np, cv2.MORPH_BLACKHAT, self.morph_kernel)
		# Channel 4: High-pass sharpness (unsharp-like)
		blurred = cv2.GaussianBlur(image_np, (0, 0), sigmaX=10)
		c4 = cv2.addWeighted(image_np, 4.0, blurred, -4.0, 128)

		stacked = np.stack([c1, c2, c3, c4], axis=-1)
		img_t = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0

		if self.mode == "train":
			if torch.rand(1).item() > 0.5:
				brightness = float(torch.empty(1).uniform_(0.85, 1.15).item())
				img_t = torch.clamp(img_t * brightness, 0.0, 1.0)
			if torch.rand(1).item() > 0.5:
				gamma = float(torch.empty(1).uniform_(0.75, 1.30).item())
				img_t = torch.pow(img_t, gamma)
			if torch.rand(1).item() > 0.5:
				noise = torch.randn_like(img_t) * 0.03
				img_t = torch.clamp(img_t + noise, 0.0, 1.0)

		mask_np = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
		mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()

		sdm_np = _signed_distance_map(mask_np)
		sdm_t = torch.from_numpy(sdm_np).unsqueeze(0).float()

		return img_t, mask_t, sdm_t, sample["file_name"]


def loader(img_size, batch_size, split="train", config=None, sampler=None):
	if config is None:
		raise ValueError("config must be provided to loader()")

	def seed_worker(worker_id):
		worker_seed = torch.initial_seed() % 2**32
		np.random.seed(worker_seed)
		random.seed(worker_seed)

	ds = VesselSegmentationDatasetV2(
		json_path=config["data"]["json_path"],
		split=split,
		source=config["data"].get("source", "syntax"),
		img_size=img_size,
		mode="train" if split == "train" else "eval",
		root_dir=config["data"].get("root_dir", "."),
	)

	g = torch.Generator()
	g.manual_seed(42)

	num_workers = int(config["data"].get("num_workers", 4))
	pin_memory = bool(config["data"].get("pin_memory", torch.cuda.is_available()))
	prefetch_factor = int(config["data"].get("prefetch_factor", 4))

	loader_kwargs = {
		"batch_size": batch_size,
		"shuffle": (split == "train" and sampler is None),
		"sampler": sampler,
		"num_workers": num_workers,
		"persistent_workers": (num_workers > 0),
		"pin_memory": pin_memory,
		"worker_init_fn": seed_worker,
		"generator": g,
	}
	if num_workers > 0:
		loader_kwargs["prefetch_factor"] = max(2, prefetch_factor)

	return DataLoader(ds, **loader_kwargs)


def build_weighted_sampler(dataset: VesselSegmentationDatasetV2, sample_weights_csv: str) -> Optional[WeightedRandomSampler]:
	if not sample_weights_csv or not os.path.isfile(sample_weights_csv):
		return None

	weight_dict: Dict[str, float] = {}
	with open(sample_weights_csv, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			file_name = row.get("file_name", "")
			value = float(row.get("sample_weight", 1.0))
			weight_dict[file_name] = max(1e-3, value)

	sample_weights = [weight_dict.get(sample["file_name"], 1.0) for sample in dataset.samples]
	weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
	return WeightedRandomSampler(weights=weights_tensor, num_samples=len(sample_weights), replacement=True)


class DropPath(nn.Module):
	def __init__(self, drop_prob: float = 0.0):
		super().__init__()
		self.drop_prob = float(drop_prob)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.drop_prob == 0.0 or not self.training:
			return x
		keep_prob = 1.0 - self.drop_prob
		shape = (x.shape[0],) + (1,) * (x.ndim - 1)
		random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
		random_tensor.floor_()
		return x.div(keep_prob) * random_tensor


class SpatialChannelAttention(nn.Module):
	def __init__(self, channels: int, reduction: int = 16):
		super().__init__()
		reduced = max(8, channels // reduction)
		self.cse = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, reduced, kernel_size=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(reduced, channels, kernel_size=1),
			nn.Sigmoid(),
		)
		self.sse = nn.Sequential(
			nn.Conv2d(channels, 1, kernel_size=1),
			nn.Sigmoid(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x * self.cse(x) + x * self.sse(x)


class DCNv3LikeBlock(nn.Module):
	"""
	DCNv3-style block built with DeformConv2d when available.
	Falls back to regular Conv2d if deformable ops are unavailable.
	"""

	def __init__(self, in_channels: int, out_channels: int, drop_path: float = 0.0):
		super().__init__()
		self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

		if HAS_DEFORM_CONV:
			self.offset = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1)
			nn.init.constant_(self.offset.weight, 0)
			nn.init.constant_(self.offset.bias, 0)
			self.conv = ops.DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
		else:
			self.offset = None
			self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

		self.norm = nn.BatchNorm2d(out_channels)
		self.act = nn.GELU()
		self.pw = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
		self.drop_path = DropPath(drop_path)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = self.proj(x)
		if self.offset is not None:
			out = self.conv(residual, self.offset(residual))
		else:
			out = self.conv(residual)
		out = self.norm(out)
		out = self.act(out)
		out = self.pw(out)
		return residual + self.drop_path(out)


class EncoderStage(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		depth: int,
		downsample: bool,
		drop_path_rates: List[float],
	):
		super().__init__()
		if downsample:
			self.down = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
				nn.BatchNorm2d(out_channels),
				nn.GELU(),
			)
		elif in_channels != out_channels:
			self.down = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
				nn.BatchNorm2d(out_channels),
				nn.GELU(),
			)
		else:
			self.down = nn.Identity()

		blocks = []
		for i in range(depth):
			blocks.append(DCNv3LikeBlock(out_channels, out_channels, drop_path=drop_path_rates[i]))
		self.blocks = nn.Sequential(*blocks)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.down(x)
		x = self.blocks(x)
		return x


class SubPixelUpBlock(nn.Module):
	def __init__(self, in_channels_up: int, in_channels_skip: int, out_channels: int, drop_path: float = 0.0):
		super().__init__()
		self.up = nn.Sequential(
			nn.Conv2d(in_channels_up, out_channels * 4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2),
			nn.BatchNorm2d(out_channels),
			nn.GELU(),
		)
		self.fuse = nn.Sequential(
			DCNv3LikeBlock(out_channels + in_channels_skip, out_channels, drop_path=drop_path),
			SpatialChannelAttention(out_channels),
			DCNv3LikeBlock(out_channels, out_channels, drop_path=drop_path),
		)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		x = self.up(x)
		if x.shape[2:] != skip.shape[2:]:
			x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
		x = torch.cat([x, skip], dim=1)
		return self.fuse(x)


class VesselNetV2(nn.Module):
	def __init__(
		self,
		in_chans: int = 4,
		num_classes: int = 1,
		dims: Tuple[int, int, int, int] = (64, 128, 256, 512),
		depths: Tuple[int, int, int, int] = (2, 2, 3, 2),
		drop_path_rate: float = 0.1,
	):
		super().__init__()
		assert len(dims) == 4, "dims must contain 4 stages"
		assert len(depths) == 4, "depths must contain 4 stages"

		total_blocks = int(sum(depths))
		dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
		dp_idx = 0

		self.stem = nn.Sequential(
			nn.Conv2d(in_chans, dims[0], kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(dims[0]),
			nn.GELU(),
		)

		self.stage1 = EncoderStage(
			dims[0], dims[0], depth=depths[0], downsample=False, drop_path_rates=dp_rates[dp_idx : dp_idx + depths[0]]
		)
		dp_idx += depths[0]
		self.stage2 = EncoderStage(
			dims[0], dims[1], depth=depths[1], downsample=True, drop_path_rates=dp_rates[dp_idx : dp_idx + depths[1]]
		)
		dp_idx += depths[1]
		self.stage3 = EncoderStage(
			dims[1], dims[2], depth=depths[2], downsample=True, drop_path_rates=dp_rates[dp_idx : dp_idx + depths[2]]
		)
		dp_idx += depths[2]
		self.stage4 = EncoderStage(
			dims[2], dims[3], depth=depths[3], downsample=True, drop_path_rates=dp_rates[dp_idx : dp_idx + depths[3]]
		)

		self.dec3 = SubPixelUpBlock(dims[3], dims[2], dims[2], drop_path=drop_path_rate)
		self.dec2 = SubPixelUpBlock(dims[2], dims[1], dims[1], drop_path=drop_path_rate)
		self.dec1 = SubPixelUpBlock(dims[1], dims[0], dims[0], drop_path=drop_path_rate)

		self.seg_head = nn.Conv2d(dims[0], num_classes, kernel_size=1)
		self.sdm_head = nn.Conv2d(dims[0], 1, kernel_size=1)

		self.deep_head3 = nn.Conv2d(dims[2], num_classes, kernel_size=1)
		self.deep_head2 = nn.Conv2d(dims[1], num_classes, kernel_size=1)

	def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
		h, w = x.shape[2:]

		x = self.stem(x)
		f1 = self.stage1(x)
		f2 = self.stage2(f1)
		f3 = self.stage3(f2)
		f4 = self.stage4(f3)

		d3 = self.dec3(f4, f3)
		d2 = self.dec2(d3, f2)
		d1 = self.dec1(d2, f1)

		seg_logits = self.seg_head(d1)
		sdm_pred = self.sdm_head(d1)

		deep3 = F.interpolate(self.deep_head3(d3), size=(h, w), mode="bilinear", align_corners=False)
		deep2 = F.interpolate(self.deep_head2(d2), size=(h, w), mode="bilinear", align_corners=False)

		return {
			"seg_logits": seg_logits,
			"sdm_pred": sdm_pred,
			"deep_logits": [deep3, deep2],
		}


class VesselNetV2EfficientEncoder(nn.Module):
	def __init__(
		self,
		in_chans: int = 4,
		num_classes: int = 1,
		encoder_name: str = "efficientnetv2_s",
		encoder_pretrained: bool = False,
		encoder_img_size: Optional[int] = None,
		drop_path_rate: float = 0.1,
	):
		super().__init__()

		if timm is None:
			raise ImportError(
				"timm is required for VesselNetV2EfficientEncoder. Install timm or switch model.name to 'VesselNetV2'."
			)

		encoder_kwargs = {
			"pretrained": encoder_pretrained,
			"in_chans": in_chans,
			"features_only": True,
		}
		if encoder_img_size is not None:
			img_size = int(encoder_img_size)
			encoder_kwargs["img_size"] = (img_size, img_size)

		try:
			self.encoder = timm.create_model(encoder_name, **encoder_kwargs)
		except TypeError as e:
			if "img_size" in encoder_kwargs and "unexpected keyword argument" in str(e):
				encoder_kwargs.pop("img_size", None)
				self.encoder = timm.create_model(encoder_name, **encoder_kwargs)
			else:
				raise

		enc_channels = list(self.encoder.feature_info.channels())
		if len(enc_channels) < 4:
			raise ValueError(f"Encoder '{encoder_name}' returned {len(enc_channels)} stages; expected at least 4.")
		self.enc_channels = enc_channels[-4:]

		self.dec3 = SubPixelUpBlock(self.enc_channels[3], self.enc_channels[2], self.enc_channels[2], drop_path=drop_path_rate)
		self.dec2 = SubPixelUpBlock(self.enc_channels[2], self.enc_channels[1], self.enc_channels[1], drop_path=drop_path_rate)
		self.dec1 = SubPixelUpBlock(self.enc_channels[1], self.enc_channels[0], self.enc_channels[0], drop_path=drop_path_rate)

		self.seg_head = nn.Conv2d(self.enc_channels[0], num_classes, kernel_size=1)
		self.sdm_head = nn.Conv2d(self.enc_channels[0], 1, kernel_size=1)

		self.deep_head3 = nn.Conv2d(self.enc_channels[2], num_classes, kernel_size=1)
		self.deep_head2 = nn.Conv2d(self.enc_channels[1], num_classes, kernel_size=1)

	def _ensure_nchw(self, feat: torch.Tensor, expected_channels: int, stage_name: str) -> torch.Tensor:
		if feat.ndim != 4:
			raise RuntimeError(f"Encoder feature '{stage_name}' must be 4D, got shape={tuple(feat.shape)}")

		if feat.shape[1] == expected_channels:
			return feat
		if feat.shape[-1] == expected_channels:
			return feat.permute(0, 3, 1, 2).contiguous()

		raise RuntimeError(
			f"Unexpected feature layout at '{stage_name}': shape={tuple(feat.shape)}, "
			f"expected channel dimension={expected_channels}."
		)

	def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
		h, w = x.shape[2:]
		feats = list(self.encoder(x))
		if len(feats) < 4:
			raise RuntimeError(f"Encoder produced {len(feats)} stages; expected at least 4.")

		f1 = self._ensure_nchw(feats[-4], self.enc_channels[0], "f1")
		f2 = self._ensure_nchw(feats[-3], self.enc_channels[1], "f2")
		f3 = self._ensure_nchw(feats[-2], self.enc_channels[2], "f3")
		f4 = self._ensure_nchw(feats[-1], self.enc_channels[3], "f4")

		d3 = self.dec3(f4, f3)
		d2 = self.dec2(d3, f2)
		d1 = self.dec1(d2, f1)

		seg_logits = self.seg_head(d1)
		sdm_pred = self.sdm_head(d1)
		if seg_logits.shape[2:] != (h, w):
			seg_logits = F.interpolate(seg_logits, size=(h, w), mode="bilinear", align_corners=False)
		if sdm_pred.shape[2:] != (h, w):
			sdm_pred = F.interpolate(sdm_pred, size=(h, w), mode="bilinear", align_corners=False)

		deep3 = F.interpolate(self.deep_head3(d3), size=(h, w), mode="bilinear", align_corners=False)
		deep2 = F.interpolate(self.deep_head2(d2), size=(h, w), mode="bilinear", align_corners=False)

		return {
			"seg_logits": seg_logits,
			"sdm_pred": sdm_pred,
			"deep_logits": [deep3, deep2],
		}


def soft_erode(img):
	return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)


def soft_dilate(img):
	return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img):
	return soft_dilate(soft_erode(img))


def soft_skel(img, iter_=5):
	img1 = soft_open(img)
	skel = F.relu(img - img1)
	for _ in range(iter_):
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


def soft_boundary_map(mask: torch.Tensor) -> torch.Tensor:
	dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
	eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
	return torch.clamp(dilated - eroded, 0.0, 1.0)


def ohem_topk_pixels(
	loss_map: torch.Tensor,
	target_mask: torch.Tensor,
	keep_ratio: float = 0.25,
	min_kept: int = 1024,
	pos_keep_ratio: float = 0.5,
	min_pos_kept: int = 256,
	max_pos_to_neg_ratio: float = 1.0,
):
	b = loss_map.shape[0]
	flat_loss = loss_map.view(b, -1)
	flat_target = (target_mask.view(b, -1) > 0.5)

	sample_losses: List[torch.Tensor] = []
	for i in range(b):
		cur_loss = flat_loss[i]
		pos = flat_target[i]

		pos_loss = cur_loss[pos]
		neg_loss = cur_loss[~pos]
		hard_neg = cur_loss.new_empty((0,))

		if neg_loss.numel() > 0:
			k_neg = max(min_kept, int(keep_ratio * neg_loss.numel()))
			k_neg = min(k_neg, neg_loss.numel())
			hard_neg = torch.topk(neg_loss, k=k_neg, largest=True).values

		selected_parts: List[torch.Tensor] = []
		if hard_neg.numel() > 0:
			selected_parts.append(hard_neg)

		if pos_loss.numel() > 0:
			k_pos = max(min_pos_kept, int(pos_keep_ratio * pos_loss.numel()))
			k_pos = min(k_pos, pos_loss.numel())

			if hard_neg.numel() > 0 and max_pos_to_neg_ratio > 0.0:
				max_pos = max(min_pos_kept, int(max_pos_to_neg_ratio * hard_neg.numel()))
				k_pos = min(k_pos, max_pos)

			if k_pos < pos_loss.numel():
				hard_pos = torch.topk(pos_loss, k=k_pos, largest=True).values
			else:
				hard_pos = pos_loss

			selected_parts.append(hard_pos)

		if selected_parts:
			selected = torch.cat(selected_parts, dim=0)
		else:
			selected = cur_loss

		sample_losses.append(selected.mean())

	return torch.stack(sample_losses).mean()


def annealed_bce_boundary_weights(
	epoch: int,
	total_epochs: int,
	bce_start: float,
	bce_end: float,
	boundary_start: float,
	boundary_end: float,
):
	t = float(epoch) / float(max(1, total_epochs - 1))
	bce_w = bce_start + (bce_end - bce_start) * t
	boundary_w = boundary_start + (boundary_end - boundary_start) * t
	return float(bce_w), float(boundary_w)


class VesselHybridLoss(nn.Module):
	def __init__(self, config: Dict):
		super().__init__()
		loss_cfg = config["loss"]

		self.ohem_ratio = float(loss_cfg.get("ohem_ratio", 0.25))
		self.ohem_min_kept = int(loss_cfg.get("ohem_min_kept", 2048))
		self.ohem_pos_keep_ratio = float(loss_cfg.get("ohem_pos_keep_ratio", 0.5))
		self.ohem_min_pos_kept = int(loss_cfg.get("ohem_min_pos_kept", 256))
		self.ohem_max_pos_to_neg_ratio = float(loss_cfg.get("ohem_max_pos_to_neg_ratio", 1.0))
		self.cldice_weight = float(loss_cfg.get("cldice_weight", 0.2))
		self.deep_sup_weight = float(loss_cfg.get("deep_supervision_weight", 0.2))
		self.sdm_weight = float(loss_cfg.get("sdm_weight", 0.08))
		self.cldice_iters = int(loss_cfg.get("cldice_iters", 5))

		self.bce_start_weight = float(loss_cfg.get("bce_start_weight", 1.0))
		self.bce_end_weight = float(loss_cfg.get("bce_end_weight", 0.3))
		self.boundary_start_weight = float(loss_cfg.get("boundary_start_weight", 0.0))
		self.boundary_end_weight = float(loss_cfg.get("boundary_end_weight", 1.0))

		self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
		self.sdm_loss = nn.SmoothL1Loss(reduction="mean")

	def forward(self, outputs, target_mask, target_sdm, epoch: int, total_epochs: int):
		seg_logits = outputs["seg_logits"]
		deep_logits = outputs.get("deep_logits", [])
		sdm_pred = outputs["sdm_pred"]

		bce_map = self.bce_loss(seg_logits, target_mask)
		ohem_bce = ohem_topk_pixels(
			bce_map,
			target_mask,
			keep_ratio=self.ohem_ratio,
			min_kept=self.ohem_min_kept,
			pos_keep_ratio=self.ohem_pos_keep_ratio,
			min_pos_kept=self.ohem_min_pos_kept,
			max_pos_to_neg_ratio=self.ohem_max_pos_to_neg_ratio,
		)

		probs = torch.sigmoid(seg_logits)
		boundary_true = soft_boundary_map(target_mask)
		boundary_pred = soft_boundary_map(probs)
		boundary_logits = torch.logit(boundary_pred.clamp(min=1e-6, max=1.0 - 1e-6))
		boundary_map = F.binary_cross_entropy_with_logits(
			boundary_logits,
			boundary_true,
			reduction="none",
		)
		ohem_boundary = ohem_topk_pixels(
			boundary_map,
			boundary_true,
			keep_ratio=self.ohem_ratio,
			min_kept=max(256, self.ohem_min_kept // 2),
			pos_keep_ratio=self.ohem_pos_keep_ratio,
			min_pos_kept=max(64, self.ohem_min_pos_kept // 2),
			max_pos_to_neg_ratio=self.ohem_max_pos_to_neg_ratio,
		)

		cldice = soft_cldice_loss(probs, target_mask, iter_=self.cldice_iters)

		if deep_logits:
			deep_losses = [F.binary_cross_entropy_with_logits(logit, target_mask) for logit in deep_logits]
			deep_sup_loss = torch.stack(deep_losses).mean()
		else:
			deep_sup_loss = seg_logits.new_tensor(0.0)

		sdm_pred_norm = torch.tanh(sdm_pred)
		sdm_loss = self.sdm_loss(sdm_pred_norm, target_sdm)

		bce_w, boundary_w = annealed_bce_boundary_weights(
			epoch=epoch,
			total_epochs=total_epochs,
			bce_start=self.bce_start_weight,
			bce_end=self.bce_end_weight,
			boundary_start=self.boundary_start_weight,
			boundary_end=self.boundary_end_weight,
		)

		seg_loss = (
			bce_w * ohem_bce
			+ boundary_w * ohem_boundary
			+ self.cldice_weight * cldice
			+ self.deep_sup_weight * deep_sup_loss
		)
		total = seg_loss + self.sdm_weight * sdm_loss

		parts = {
			"total": float(total.detach().item()),
			"seg": float(seg_loss.detach().item()),
			"bce_ohem": float(ohem_bce.detach().item()),
			"boundary_ohem": float(ohem_boundary.detach().item()),
			"cldice_loss": float(cldice.detach().item()),
			"deep_sup": float(deep_sup_loss.detach().item()),
			"sdm": float(sdm_loss.detach().item()),
			"w_bce": float(bce_w),
			"w_boundary": float(boundary_w),
		}
		return total, parts


def _amp_enabled(config: Dict) -> bool:
	return bool(config["training"].get("use_amp", True)) and torch.cuda.is_available()


def _amp_dtype(config: Dict):
	precision = str(config["training"].get("precision", "bfloat16")).lower()
	if precision == "float16":
		return torch.float16
	return torch.bfloat16


def _pick_logging_batch_index(num_batches: int, epoch: int, split_name: str) -> int:
	if num_batches <= 0:
		return 0
	split_offset = sum(ord(c) for c in split_name)
	return int((epoch * 131 + split_offset) % num_batches)


def _count_model_parameters(model: nn.Module) -> Tuple[int, int]:
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return int(total_params), int(trainable_params)


def _log_prediction_grid(tb_writer: SummaryWriter, tag: str, step: int, images, masks, probs, threshold: float):
	num_samples = min(4, images.size(0))
	grid_images = []
	preds = (probs > threshold).float()

	for i in range(num_samples):
		img_vis = images[i, 0:1].detach().cpu().repeat(3, 1, 1)
		pred_vis = preds[i].detach().cpu().repeat(3, 1, 1)
		mask_vis = masks[i].detach().cpu().repeat(3, 1, 1)
		grid_images.extend([img_vis, pred_vis, mask_vis])

	grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
	tb_writer.add_image(tag, grid, step)


def _f1_iou_dice_from_counts(tp: float, fp: float, fn: float):
	f1 = (2.0 * tp) / max(1e-8, (2.0 * tp + fp + fn))
	iou = tp / max(1e-8, (tp + fp + fn))
	dice = (2.0 * tp) / max(1e-8, (2.0 * tp + fp + fn))
	return float(f1), float(iou), float(dice)


def _tp_fp_fn(preds: torch.Tensor, masks: torch.Tensor):
	p = preds.int()
	m = masks.int()
	tp = torch.logical_and(p == 1, m == 1).sum().item()
	fp = torch.logical_and(p == 1, m == 0).sum().item()
	fn = torch.logical_and(p == 0, m == 1).sum().item()
	return float(tp), float(fp), float(fn)


def train_epoch(model, dataloader, criterion, optimiser, epoch, config, scaler=None):
	model.train()
	accum_steps = int(config["training"].get("accum_steps", 1))
	clip_grad = float(config["training"].get("clip_grad_norm", 1.0))
	num_epochs = int(config["training"]["epochs"])

	running = {
		"loss": 0.0,
		"seg": 0.0,
		"bce_ohem": 0.0,
		"boundary_ohem": 0.0,
		"cldice_loss": 0.0,
		"deep_sup": 0.0,
		"sdm": 0.0,
		"w_bce": 0.0,
		"w_boundary": 0.0,
	}

	optimiser.zero_grad(set_to_none=True)
	pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1} [Train]")

	for batch_idx, (images, masks, sdm, _) in pbar:
		images = images.to(device, non_blocking=True)
		masks = masks.to(device, non_blocking=True)
		sdm = sdm.to(device, non_blocking=True)

		with torch.amp.autocast("cuda", enabled=_amp_enabled(config), dtype=_amp_dtype(config)):
			outputs = model(images)
			loss, parts = criterion(outputs, masks, sdm, epoch=epoch, total_epochs=num_epochs)

		if not torch.isfinite(loss):
			continue

		loss_scaled = loss / accum_steps

		if scaler is not None and scaler.is_enabled():
			scaler.scale(loss_scaled).backward()
		else:
			loss_scaled.backward()

		if ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(dataloader)):
			if scaler is not None and scaler.is_enabled():
				scaler.unscale_(optimiser)

			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

			if scaler is not None and scaler.is_enabled():
				scaler.step(optimiser)
				scaler.update()
			else:
				optimiser.step()

			optimiser.zero_grad(set_to_none=True)

		running["loss"] += float(loss.detach().item())
		for k in ["seg", "bce_ohem", "boundary_ohem", "cldice_loss", "deep_sup", "sdm", "w_bce", "w_boundary"]:
			running[k] += float(parts[k])

		denom = batch_idx + 1
		pbar.set_postfix(
			{
				"loss": running["loss"] / denom,
				"bce": running["bce_ohem"] / denom,
				"bnd": running["boundary_ohem"] / denom,
				"cld": running["cldice_loss"] / denom,
			}
		)

	n = max(1, len(dataloader))
	return {k: v / n for k, v in running.items()}


def validate_epoch(model, dataloader, criterion, epoch, config, threshold=0.5, tb_writer=None, split_name="Val"):
	model.eval()
	num_epochs = int(config["training"]["epochs"])
	log_batch_idx = _pick_logging_batch_index(len(dataloader), epoch, split_name) if tb_writer is not None else -1

	running = {
		"loss": 0.0,
		"seg": 0.0,
		"bce_ohem": 0.0,
		"boundary_ohem": 0.0,
		"cldice_loss": 0.0,
		"deep_sup": 0.0,
		"sdm": 0.0,
		"w_bce": 0.0,
		"w_boundary": 0.0,
	}

	total_tp = 0.0
	total_fp = 0.0
	total_fn = 0.0
	total_cldice_metric = 0.0

	with torch.no_grad():
		pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1} [{split_name}]")
		for batch_idx, (images, masks, sdm, _) in pbar:
			images = images.to(device, non_blocking=True)
			masks = masks.to(device, non_blocking=True)
			sdm = sdm.to(device, non_blocking=True)

			with torch.amp.autocast("cuda", enabled=_amp_enabled(config), dtype=_amp_dtype(config)):
				outputs = model(images)
				loss, parts = criterion(outputs, masks, sdm, epoch=epoch, total_epochs=num_epochs)

			probs = torch.sigmoid(outputs["seg_logits"])
			preds = (probs > threshold).float()

			tp, fp, fn = _tp_fp_fn(preds, masks)
			total_tp += tp
			total_fp += fp
			total_fn += fn

			batch_cldice = 1.0 - float(soft_cldice_loss(probs, masks, iter_=int(config["loss"].get("cldice_iters", 5))).item())
			total_cldice_metric += batch_cldice

			running["loss"] += float(loss.detach().item())
			for k in ["seg", "bce_ohem", "boundary_ohem", "cldice_loss", "deep_sup", "sdm", "w_bce", "w_boundary"]:
				running[k] += float(parts[k])

			if tb_writer is not None and batch_idx == log_batch_idx:
				_log_prediction_grid(
					tb_writer=tb_writer,
					tag=f"{split_name}/Predictions_thr_{threshold:.2f}",
					step=epoch,
					images=images,
					masks=masks,
					probs=probs,
					threshold=threshold,
				)

			denom = batch_idx + 1
			f1_so_far, iou_so_far, _ = _f1_iou_dice_from_counts(total_tp, total_fp, total_fn)
			pbar.set_postfix(
				{
					"loss": running["loss"] / denom,
					"f1": f1_so_far,
					"iou": iou_so_far,
				}
			)

	n = max(1, len(dataloader))
	f1, iou, dice = _f1_iou_dice_from_counts(total_tp, total_fp, total_fn)
	out = {k: v / n for k, v in running.items()}
	out.update(
		{
			"f1": f1,
			"iou": iou,
			"dice": dice,
			"cldice_metric": total_cldice_metric / n,
		}
	)
	return out


def _collect_probs_and_masks(model, dataloader, config):
	model.eval()
	all_probs = []
	all_masks = []
	with torch.no_grad():
		for images, masks, _, _ in dataloader:
			images = images.to(device, non_blocking=True)
			with torch.amp.autocast("cuda", enabled=_amp_enabled(config), dtype=_amp_dtype(config)):
				outputs = model(images)
			probs = torch.sigmoid(outputs["seg_logits"]).detach().cpu()
			all_probs.append(probs)
			all_masks.append(masks.int().cpu())

	return torch.cat(all_probs, dim=0), torch.cat(all_masks, dim=0)


def find_best_f1_threshold(model, dataloader, config):
	threshold_grid = config["evaluation"].get("threshold_grid", np.arange(0.1, 0.95, 0.05).tolist())
	thresholds = [float(t) for t in threshold_grid]
	all_probs, all_masks = _collect_probs_and_masks(model, dataloader, config)

	best_threshold = 0.5
	best_f1 = -1.0

	for t in thresholds:
		preds = (all_probs > t).int()
		tp, fp, fn = _tp_fp_fn(preds, all_masks)
		f1, _, _ = _f1_iou_dice_from_counts(tp, fp, fn)
		if f1 > best_f1:
			best_f1 = f1
			best_threshold = float(t)

	return best_threshold, best_f1


def evaluate_all_thresholds(model, dataloader, config):
	threshold_grid = config["evaluation"].get("threshold_grid", np.arange(0.1, 0.95, 0.05).tolist())
	thresholds = [float(t) for t in threshold_grid]

	csv_path = os.path.join(
		config["logging"]["log_dir"].format(experiment_name=config["experiment_name"]),
		"thresholds_results.csv",
	)

	all_probs, all_masks = _collect_probs_and_masks(model, dataloader, config)

	with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["Threshold", "F1_Score", "IoU_Score", "Dice_Score", "clDice_Score"])

		for t in thresholds:
			preds = (all_probs > t).float()
			tp, fp, fn = _tp_fp_fn(preds, all_masks)
			f1, iou, dice = _f1_iou_dice_from_counts(tp, fp, fn)
			cldice_score = 1.0 - float(
				soft_cldice_loss(preds, all_masks.float(), iter_=int(config["loss"].get("cldice_iters", 5))).item()
			)

			writer.writerow([f"{t:.2f}", f"{f1:.6f}", f"{iou:.6f}", f"{dice:.6f}", f"{cldice_score:.6f}"])

	return csv_path


def test_model(model, dataloader, criterion, tb_writer, config, threshold=0.5):
	stats = validate_epoch(
		model=model,
		dataloader=dataloader,
		criterion=criterion,
		epoch=int(config["training"]["epochs"]) - 1,
		config=config,
		threshold=threshold,
		tb_writer=tb_writer,
		split_name="Test",
	)

	print(
		f"Test @thr={threshold:.2f} | F1: {stats['f1']:.4f} | IoU: {stats['iou']:.4f} | "
		f"Dice: {stats['dice']:.4f} | clDice: {stats['cldice_metric']:.4f}"
	)
	tb_writer.add_scalar("Test/F1", stats["f1"])
	tb_writer.add_scalar("Test/IoU", stats["iou"])
	tb_writer.add_scalar("Test/Dice", stats["dice"])
	tb_writer.add_scalar("Test/clDice", stats["cldice_metric"])
	return stats


def test_model_youden(model, dataloader, criterion, tb_writer, config, optimal_threshold):
	stats = validate_epoch(
		model=model,
		dataloader=dataloader,
		criterion=criterion,
		epoch=int(config["training"]["epochs"]) - 1,
		config=config,
		threshold=optimal_threshold,
		tb_writer=tb_writer,
		split_name="Test_Youden",
	)

	print(
		f"Test (BestThr={optimal_threshold:.2f}) | F1: {stats['f1']:.4f} | IoU: {stats['iou']:.4f} | "
		f"Dice: {stats['dice']:.4f} | clDice: {stats['cldice_metric']:.4f}"
	)
	tb_writer.add_scalar("Test_Youden/F1", stats["f1"])
	tb_writer.add_scalar("Test_Youden/IoU", stats["iou"])
	tb_writer.add_scalar("Test_Youden/Dice", stats["dice"])
	tb_writer.add_scalar("Test_Youden/clDice", stats["cldice_metric"])
	return stats


def trainScript(
	model,
	train_loader,
	val_loader,
	test_loader,
	criterion,
	optimiser,
	scheduler,
	num_epochs,
	config,
	tb_writer,
):
	checkpoint_dir = config["logging"]["checkpoint_dir"].format(experiment_name=config["experiment_name"])
	os.makedirs(checkpoint_dir, exist_ok=True)

	best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
	last_model_path = os.path.join(checkpoint_dir, "last_model.pth")

	best_val_f1 = -1.0
	patience = int(config["training"].get("patience", 0))
	epochs_no_improve = 0
	start_epoch = 0

	scaler = torch.amp.GradScaler("cuda", enabled=(_amp_enabled(config) and _amp_dtype(config) == torch.float16))

	if os.path.isfile(last_model_path):
		print(f"[INFO] Resuming from {last_model_path}")
		ckpt = torch.load(last_model_path, map_location=device)
		if isinstance(ckpt, dict) and "epoch" in ckpt:
			model.load_state_dict(ckpt["model_state_dict"])
			try:
				optimiser.load_state_dict(ckpt["optimizer_state_dict"])
				scheduler.load_state_dict(ckpt["scheduler_state_dict"])
				best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
				epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
				start_epoch = int(ckpt["epoch"]) + 1
				print(f"[INFO] Resume complete at epoch {start_epoch} (best val F1: {best_val_f1:.4f})")
			except Exception as e:
				# Common when trainable parameter sets change (e.g., different freeze policy).
				best_val_f1 = -1.0
				epochs_no_improve = 0
				start_epoch = 0
				print(
					"[WARN] Checkpoint optimizer/scheduler state is incompatible with current run; "
					"falling back to model-only resume with fresh optimiser/scheduler. "
					f"Details: {e}"
				)

	default_threshold = float(config["evaluation"].get("default_threshold", 0.5))
	vis_every = max(1, int(config["logging"].get("visualize_every_epochs", 2)))

	for epoch in range(start_epoch, num_epochs):
		train_stats = train_epoch(
			model=model,
			dataloader=train_loader,
			criterion=criterion,
			optimiser=optimiser,
			epoch=epoch,
			config=config,
			scaler=scaler,
		)

		val_stats = validate_epoch(
			model=model,
			dataloader=val_loader,
			criterion=criterion,
			epoch=epoch,
			config=config,
			threshold=default_threshold,
			tb_writer=tb_writer if ((epoch + 1) % vis_every == 0 or epoch == 0) else None,
			split_name="Val",
		)

		scheduler.step()

		tb_writer.add_scalar("Loss/train", train_stats["loss"], epoch)
		tb_writer.add_scalar("Loss/val", val_stats["loss"], epoch)

		tb_writer.add_scalar("Loss/train_bce_ohem", train_stats["bce_ohem"], epoch)
		tb_writer.add_scalar("Loss/train_boundary_ohem", train_stats["boundary_ohem"], epoch)
		tb_writer.add_scalar("Loss/train_cldice", train_stats["cldice_loss"], epoch)
		tb_writer.add_scalar("Loss/train_deep_supervision", train_stats["deep_sup"], epoch)
		tb_writer.add_scalar("Loss/train_sdm", train_stats["sdm"], epoch)

		tb_writer.add_scalar("Loss/val_bce_ohem", val_stats["bce_ohem"], epoch)
		tb_writer.add_scalar("Loss/val_boundary_ohem", val_stats["boundary_ohem"], epoch)
		tb_writer.add_scalar("Loss/val_cldice", val_stats["cldice_loss"], epoch)
		tb_writer.add_scalar("Loss/val_deep_supervision", val_stats["deep_sup"], epoch)
		tb_writer.add_scalar("Loss/val_sdm", val_stats["sdm"], epoch)

		tb_writer.add_scalar("Val/F1", val_stats["f1"], epoch)
		tb_writer.add_scalar("Val/IoU", val_stats["iou"], epoch)
		tb_writer.add_scalar("Val/Dice", val_stats["dice"], epoch)
		tb_writer.add_scalar("Val/clDice", val_stats["cldice_metric"], epoch)
		tb_writer.add_scalar("Train/Anneal_BCE_Weight", train_stats["w_bce"], epoch)
		tb_writer.add_scalar("Train/Anneal_Boundary_Weight", train_stats["w_boundary"], epoch)

		if val_stats["f1"] > best_val_f1:
			best_val_f1 = val_stats["f1"]
			epochs_no_improve = 0
			torch.save(model.state_dict(), best_model_path)
			print(f"[INFO] New best model saved with Val F1={best_val_f1:.4f}")
		else:
			epochs_no_improve += 1

		checkpoint = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimiser.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"best_val_f1": best_val_f1,
			"epochs_no_improve": epochs_no_improve,
		}
		torch.save(checkpoint, last_model_path)

		patience_msg = f" | Patience: {epochs_no_improve}/{patience}" if patience > 0 else ""
		print(
			f"Epoch {epoch + 1}/{num_epochs} | "
			f"train_loss={train_stats['loss']:.4f} | val_loss={val_stats['loss']:.4f} | "
			f"val_f1={val_stats['f1']:.4f} | val_iou={val_stats['iou']:.4f} | "
			f"w_bce={train_stats['w_bce']:.3f} | w_boundary={train_stats['w_boundary']:.3f}{patience_msg}"
		)

		if patience > 0 and epochs_no_improve >= patience:
			print(f"[EARLY STOPPING] No improvement for {patience} epochs.")
			break

	print("\n" + "=" * 60)
	print("Testing")
	print("=" * 60 + "\n")

	model.load_state_dict(torch.load(best_model_path, map_location=device))

	test_model(
		model=model,
		dataloader=test_loader,
		criterion=criterion,
		tb_writer=tb_writer,
		config=config,
		threshold=default_threshold,
	)

	best_threshold, best_f1_val = find_best_f1_threshold(model, val_loader, config)
	print(f"Best validation threshold by F1: {best_threshold:.4f} (F1={best_f1_val:.4f})")

	youden_stats = test_model_youden(
		model=model,
		dataloader=test_loader,
		criterion=criterion,
		tb_writer=tb_writer,
		config=config,
		optimal_threshold=best_threshold,
	)

	threshold_csv = evaluate_all_thresholds(model, test_loader, config)
	print(f"Threshold sweep saved to: {threshold_csv}")

	return best_model_path, youden_stats["f1"]


if __name__ == "__main__":
	config = {
		"experiment_name": "trainv2_dcn_subpixel_multitask_biggedDECODER",
		"logging": {
			"log_dir": "runs/{experiment_name}",
			"checkpoint_dir": "checkpoints/{experiment_name}",
			"visualize_every_epochs": 2,
		},
		"data": {
			"json_path": "data/ARCADE/processed/dataset.json",
			"root_dir": ".",
			"source": "syntax",
			"img_size": 512,
			"batch_size": 4,
			"num_workers": 12,
			"pin_memory": True,
			"prefetch_factor": 4,
			"sample_weights_csv": "results/hard_case_mining/sample_weights_train.csv",
		},
		"training": {
			"epochs": 120,
			"learning_rate": 2e-4,
			"weight_decay": 1e-4,
			"scheduler": "Warmup + CosineAnnealingLR",
			"precision": "bfloat16",
			"use_amp": True,
			"accum_steps": 2,
			"clip_grad_norm": 1.0,
			"patience": 50,
			"warmup_epochs": 5,
		},
		"model": {
			"name": "VesselNetV2EfficientEncoder",
			"in_chans": 4,
			"num_classes": 1,
			"encoder_name": "efficientnetv2_s",
			"encoder_pretrained": False,
			"encoder_img_size": 512,
			"dims": [128, 256, 256, 512],
			"depths": [3, 3, 3, 3],
			"drop_path_rate": 0.2,
		},
		"loss": {
			"recipe": "OHEM BCE + Boundary Annealing + clDice + Deep Supervision + SDM",
			"ohem_ratio": 0.25,
			"ohem_min_kept": 2048,
			"ohem_pos_keep_ratio": 0.5,
			"ohem_min_pos_kept": 256,
			"ohem_max_pos_to_neg_ratio": 1.0,
			"cldice_weight": 0.2,
			"deep_supervision_weight": 0.2,
			"sdm_weight": 0.08,
			"bce_start_weight": 1.0,
			"bce_end_weight": 0.3,
			"boundary_start_weight": 0.0,
			"boundary_end_weight": 1.0,
			"cldice_iters": 5,
		},
		"evaluation": {
			"default_threshold": 0.5,
			"threshold_grid": [round(float(x), 2) for x in np.arange(0.1, 0.95, 0.05)],
		},
	}

	writer = SummaryWriter(log_dir=config["logging"]["log_dir"].format(experiment_name=config["experiment_name"]))

	# Build train dataset first to optionally attach hard-case weighted sampling.
	train_dataset = VesselSegmentationDatasetV2(
		json_path=config["data"]["json_path"],
		split="train",
		source=config["data"].get("source", "syntax"),
		img_size=config["data"]["img_size"],
		mode="train",
		root_dir=config["data"].get("root_dir", "."),
	)
	sampler = build_weighted_sampler(train_dataset, config["data"].get("sample_weights_csv", ""))

	# We use the same loader() API style as train.py for consistency.
	train_loader = loader(
		img_size=config["data"]["img_size"],
		batch_size=config["data"]["batch_size"],
		split="train",
		config=config,
		sampler=sampler,
	)
	val_loader = loader(
		img_size=config["data"]["img_size"],
		batch_size=config["data"]["batch_size"],
		split="validation",
		config=config,
		sampler=None,
	)
	test_loader = loader(
		img_size=config["data"]["img_size"],
		batch_size=config["data"]["batch_size"],
		split="test",
		config=config,
		sampler=None,
	)

	model_name = str(config["model"].get("name", "VesselNetV2"))
	if model_name == "VesselNetV2":
		model = VesselNetV2(
			in_chans=int(config["model"]["in_chans"]),
			num_classes=int(config["model"]["num_classes"]),
			dims=tuple(config["model"].get("dims", [64, 128, 256, 512])),
			depths=tuple(config["model"].get("depths", [2, 2, 3, 2])),
			drop_path_rate=float(config["model"].get("drop_path_rate", 0.1)),
		).to(device)
	elif model_name in {"VesselNetV2EfficientEncoder", "VesselNetV3EfficientNet"}:
		model = VesselNetV2EfficientEncoder(
			in_chans=int(config["model"]["in_chans"]),
			num_classes=int(config["model"]["num_classes"]),
			encoder_name=str(config["model"].get("encoder_name", "efficientnetv2_s")),
			encoder_pretrained=bool(config["model"].get("encoder_pretrained", False)),
			encoder_img_size=config["model"].get("encoder_img_size", config["data"].get("img_size", 512)),
			drop_path_rate=float(config["model"].get("drop_path_rate", 0.1)),
		).to(device)
	else:
		raise ValueError(f"Unknown model name: {model_name}")
	total_params, trainable_params = _count_model_parameters(model)
	print(
		f"[INFO] Model params | total: {total_params:,} ({total_params / 1e6:.2f}M) | "
		f"trainable: {trainable_params:,} ({trainable_params / 1e6:.2f}M)"
	)

	optimiser = optim.AdamW(
		model.parameters(),
		lr=float(config["training"]["learning_rate"]),
		weight_decay=float(config["training"].get("weight_decay", 1e-4)),
	)

	warmup_epochs = int(config["training"].get("warmup_epochs", 5))
	total_epochs = int(config["training"]["epochs"])
	warmup_epochs = min(warmup_epochs, max(1, total_epochs - 1))

	warmup = LinearLR(optimiser, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
	cosine = CosineAnnealingLR(optimiser, T_max=max(1, total_epochs - warmup_epochs))
	scheduler = SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[warmup_epochs])

	criterion = VesselHybridLoss(config=config)

	configCreate(
		os.path.join(
			config["logging"]["log_dir"].format(experiment_name=config["experiment_name"]),
			"config.yaml",
		),
		config,
	)

	trainScript(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		test_loader=test_loader,
		criterion=criterion,
		optimiser=optimiser,
		scheduler=scheduler,
		num_epochs=total_epochs,
		config=config,
		tb_writer=writer,
	)

	writer.close()
