from __future__ import annotations

import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader import Transforms, XCADTestDataset
from engine.unext import get_model
from utils.helpers import set_seed

try:
	from segmentation_models_pytorch.losses import FocalLoss
	from segmentation_models_pytorch.losses import DiceLoss
except Exception:  # pragma: no cover
	FocalLoss = None
	DiceLoss = None


DEFAULT_XCAD_ROOT = Path("data/XCAD/XCAD/test")
DEFAULT_LOG_DIR = Path("runs/unext_xcad")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/unext_xcad")


def _autocast_context(device: torch.device):
	if device.type == "cuda":
		return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
	if device.type == "cpu":
		return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
	return nullcontext()


def _ensure_binary_mask(mask: torch.Tensor) -> torch.Tensor:
	if mask.dtype != torch.float32:
		mask = mask.float()
	return (mask > 0.5).float()


def _extract_logits(outputs):
	if torch.is_tensor(outputs):
		return outputs

	if isinstance(outputs, (tuple, list)):
		if len(outputs) == 0:
			raise KeyError("Model returned an empty sequence, cannot extract logits.")
		first = outputs[0]
		if torch.is_tensor(first):
			return first
		if isinstance(first, dict):
			outputs = first
		else:
			raise KeyError(f"Unsupported model output sequence item type: {type(first)!r}")

	if isinstance(outputs, dict):
		for key in ("seg_logits", "logits", "out", "mask", "pred"):
			if key in outputs:
				return outputs[key]
		if len(outputs) == 1:
			return next(iter(outputs.values()))
	raise KeyError("Unable to extract segmentation logits from model outputs.")


def _counts_from_predictions(preds: torch.Tensor, masks: torch.Tensor) -> Tuple[float, float, float]:
	preds = _ensure_binary_mask(preds)
	masks = _ensure_binary_mask(masks)
	tp = float(torch.sum(preds * masks).item())
	fp = float(torch.sum(preds * (1.0 - masks)).item())
	fn = float(torch.sum((1.0 - preds) * masks).item())
	return tp, fp, fn


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
	denom = (2.0 * tp) + fp + fn
	if denom == 0.0:
		return 1.0
	return (2.0 * tp) / denom


def _make_triplet_grid(images: torch.Tensor, masks: torch.Tensor, probs: torch.Tensor, max_samples: int = 4) -> torch.Tensor:
	n = min(int(images.shape[0]), max_samples)
	triplets = []
	for idx in range(n):
		image = images[idx : idx + 1].detach().float().cpu()
		mask = masks[idx : idx + 1].detach().float().cpu()
		pred = (probs[idx : idx + 1].detach().float().cpu() > 0.5).float()

		if image.min().item() < 0.0:
			image = (image + 1.0) / 2.0
		image = image.clamp(0.0, 1.0)
		mask = mask.clamp(0.0, 1.0)
		pred = pred.clamp(0.0, 1.0)

		triplets.extend([image, mask, pred])

	grid = make_grid(torch.cat(triplets, dim=0), nrow=3, padding=2)
	return grid


class BinaryFocalLoss(nn.Module):
	def __init__(self, gamma: float = 2.0):
		super().__init__()
		self.gamma = float(gamma)
		self.bce = nn.BCEWithLogitsLoss(reduction="none")

	def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		target = target.float()
		bce = self.bce(logits, target)
		prob = torch.sigmoid(logits)
		pt = prob * target + (1.0 - prob) * (1.0 - target)
		modulating = torch.pow(1.0 - pt, self.gamma)
		return (modulating * bce).mean()


def _build_focal_loss(gamma: float = 2.0):
	if FocalLoss is not None:
		try:
			return FocalLoss(mode="binary", gamma=gamma)
		except Exception:
			pass
	return BinaryFocalLoss(gamma=gamma)


def _build_dice_loss():
	if DiceLoss is not None:
		try:
			return DiceLoss(mode="binary")
		except Exception:
			pass
	return None


class SplitTransformDataset(Dataset):
	def __init__(self, dataset: Dataset, indices: Sequence[int], transform):
		self.dataset = dataset
		self.indices = list(indices)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.indices)

	def __getitem__(self, idx: int):
		image, label = self.dataset[self.indices[idx]]
		if self.transform is not None:
			return self.transform(image, label)
		return image, label


def build_xcad_dataloaders(
	data_root: Path = DEFAULT_XCAD_ROOT,
	image_size: int = 256,
	batch_size: int = 8,
	val_split: float = 0.10,
	seed: int = 42,
	num_workers: int = 4,
	pin_memory: Optional[bool] = None,
	train_augment: Optional[Transforms] = None,
	val_transform: Optional[Transforms] = None,
):
	base_dataset = XCADTestDataset(base_dir=data_root, transform=None)
	val_size = max(1, int(round(len(base_dataset) * float(val_split))))
	train_size = len(base_dataset) - val_size
	if train_size < 1:
		raise ValueError("XCAD split produced an empty training subset.")

	generator = torch.Generator().manual_seed(int(seed))
	train_indices, val_indices = random_split(range(len(base_dataset)), [train_size, val_size], generator=generator)

	if train_augment is None:
		train_augment = Transforms(
			image_size=image_size,
			training=True,
			hflip_prob=0.5,
			vflip_prob=0.2,
			rotation_degrees=15.0,
			brightness=0.12,
			contrast=0.12,
			blur_prob=0.15,
			blur_kernel_size=5,
			normalize=True,
		)

	if val_transform is None:
		val_transform = Transforms(
			image_size=image_size,
			training=False,
			normalize=True,
		)

	train_dataset = SplitTransformDataset(base_dataset, train_indices.indices, train_augment)
	val_dataset = SplitTransformDataset(base_dataset, val_indices.indices, val_transform)

	def seed_worker(worker_id: int):
		worker_seed = torch.initial_seed() % 2**32
		random.seed(worker_seed)
		np.random.seed(worker_seed)

	if pin_memory is None:
		pin_memory = torch.cuda.is_available()

	loader_kwargs = {
		"batch_size": int(batch_size),
		"num_workers": int(num_workers),
		"pin_memory": bool(pin_memory),
		"persistent_workers": bool(num_workers > 0),
		"worker_init_fn": seed_worker,
		"generator": torch.Generator().manual_seed(int(seed)),
	}
	if num_workers > 0:
		loader_kwargs["prefetch_factor"] = 2

	train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
	val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
	return train_loader, val_loader, train_dataset, val_dataset


def train_one_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	focal_loss,
	dice_loss,
	device: torch.device,
	optimizer: optim.Optimizer,
) -> Dict[str, float]:
	model.train()
	running_loss = 0.0
	running_focal = 0.0
	running_dice = 0.0
	tp = fp = fn = 0.0

	progress = tqdm(dataloader, desc="Train", leave=False)
	for images, masks in progress:
		images = images.to(device, non_blocking=True)
		masks = masks.to(device, non_blocking=True)
		masks = _ensure_binary_mask(masks)

		optimizer.zero_grad(set_to_none=True)
		with _autocast_context(device):
			outputs = model(images)
			logits = _extract_logits(outputs)
			focal = focal_loss(logits, masks)
			dice = dice_loss(logits, masks) if dice_loss is not None else (1.0 - (2.0 * torch.sum(torch.sigmoid(logits.float()) * masks) + 1e-8) / (torch.sum(torch.sigmoid(logits.float())) + torch.sum(masks) + 1e-8))
			probs = torch.sigmoid(logits.float())
			loss = focal + dice

		loss.backward()
		optimizer.step()

		preds = (probs > 0.5).float()
		batch_tp, batch_fp, batch_fn = _counts_from_predictions(preds, masks)
		tp += batch_tp
		fp += batch_fp
		fn += batch_fn
		running_loss += float(loss.detach().item())
		running_focal += float(focal.detach().item())
		running_dice += float(dice.detach().item())
		progress.set_postfix(loss=running_loss / max(1, len(progress)))

	n = max(1, len(dataloader))
	return {
		"loss": running_loss / n,
		"focal": running_focal / n,
		"dice": running_dice / n,
		"f1": _f1_from_counts(tp, fp, fn),
	}


@torch.no_grad()
def validate_one_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	focal_loss,
	dice_loss,
	device: torch.device,
	writer: Optional[SummaryWriter] = None,
	epoch: int = 0,
) -> Dict[str, float]:
	model.eval()
	running_loss = 0.0
	running_focal = 0.0
	running_dice = 0.0
	tp = fp = fn = 0.0
	logged_images = False

	progress = tqdm(dataloader, desc="Val", leave=False)
	for batch_idx, (images, masks) in enumerate(progress):
		images = images.to(device, non_blocking=True)
		masks = masks.to(device, non_blocking=True)
		masks = _ensure_binary_mask(masks)

		with _autocast_context(device):
			outputs = model(images)
			logits = _extract_logits(outputs)
			focal = focal_loss(logits, masks)
			dice = dice_loss(logits, masks) if dice_loss is not None else (1.0 - (2.0 * torch.sum(torch.sigmoid(logits.float()) * masks) + 1e-8) / (torch.sum(torch.sigmoid(logits.float())) + torch.sum(masks) + 1e-8))
			probs = torch.sigmoid(logits.float())
			loss = focal + dice

		preds = (probs > 0.5).float()
		batch_tp, batch_fp, batch_fn = _counts_from_predictions(preds, masks)
		tp += batch_tp
		fp += batch_fp
		fn += batch_fn
		running_loss += float(loss.detach().item())
		running_focal += float(focal.detach().item())
		running_dice += float(dice.detach().item())

		if writer is not None and not logged_images:
			writer.add_image("Val/data|label|prediction", _make_triplet_grid(images, masks, probs), epoch)
			logged_images = True

		progress.set_postfix(loss=running_loss / max(1, batch_idx + 1))

	n = max(1, len(dataloader))
	return {
		"loss": running_loss / n,
		"focal": running_focal / n,
		"dice": running_dice / n,
		"f1": _f1_from_counts(tp, fp, fn),
	}


def train_unext(
	data_root: Path = DEFAULT_XCAD_ROOT,
	image_size: int = 256,
	batch_size: int = 8,
	epochs: int = 200,
	lr: float = 1e-4,
	weight_decay: float = 1e-5,
	seed: int = 42,
	val_split: float = 0.10,
	num_workers: int = 4,
	log_dir: Path = DEFAULT_LOG_DIR,
	checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
	focal_gamma: float = 2.0,
	device: Optional[torch.device] = None,
	resume_path: Optional[Path] = None,
) -> Dict[str, float]:
	set_seed(int(seed))
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_loader, val_loader, _, _ = build_xcad_dataloaders(
		data_root=data_root,
		image_size=image_size,
		batch_size=batch_size,
		val_split=val_split,
		seed=seed,
		num_workers=num_workers,
	)

	model = get_model(in_channels=1, num_classes=1, device=device)
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
	focal_loss = _build_focal_loss(gamma=focal_gamma)
	dice_loss = _build_dice_loss()

	log_dir = Path(log_dir)
	checkpoint_dir = Path(checkpoint_dir)
	log_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	writer = SummaryWriter(log_dir=str(log_dir))

	start_epoch = 0
	best_val_f1 = -1.0

	if resume_path is not None and Path(resume_path).is_file():
		checkpoint = torch.load(resume_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		if "scheduler_state_dict" in checkpoint:
			scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
		start_epoch = int(checkpoint.get("epoch", 0)) + 1
		best_val_f1 = float(checkpoint.get("best_val_f1", -1.0))

	best_model_path = checkpoint_dir / "best_model.pth"
	last_model_path = checkpoint_dir / "last_model.pth"

	history: Dict[str, float] = {}
	for epoch in range(start_epoch, epochs):
		train_stats = train_one_epoch(
			model=model,
			dataloader=train_loader,
			focal_loss=focal_loss,
			dice_loss=dice_loss,
			device=device,
			optimizer=optimizer,
		)
		val_stats = validate_one_epoch(
			model=model,
			dataloader=val_loader,
			focal_loss=focal_loss,
			dice_loss=dice_loss,
			device=device,
			writer=writer,
			epoch=epoch,
		)

		scheduler.step()

		writer.add_scalar("Loss/train", train_stats["loss"], epoch)
		writer.add_scalar("Loss/val", val_stats["loss"], epoch)
		writer.add_scalar("Loss/train_focal", train_stats["focal"], epoch)
		writer.add_scalar("Loss/val_focal", val_stats["focal"], epoch)
		writer.add_scalar("Loss/train_dice", train_stats["dice"], epoch)
		writer.add_scalar("Loss/val_dice", val_stats["dice"], epoch)
		writer.add_scalar("Train/F1", train_stats["f1"], epoch)
		writer.add_scalar("Val/F1", val_stats["f1"], epoch)

		checkpoint = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"best_val_f1": best_val_f1,
			"config": {
				"data_root": str(data_root),
				"image_size": image_size,
				"batch_size": batch_size,
				"epochs": epochs,
				"lr": lr,
				"weight_decay": weight_decay,
				"seed": seed,
				"val_split": val_split,
				"focal_gamma": focal_gamma,
			},
		}
		torch.save(checkpoint, last_model_path)

		if val_stats["f1"] >= best_val_f1:
			best_val_f1 = val_stats["f1"]
			torch.save(model.state_dict(), best_model_path)

		history = {
			"train_loss": train_stats["loss"],
			"val_loss": val_stats["loss"],
			"train_f1": train_stats["f1"],
			"val_f1": val_stats["f1"],
			"best_val_f1": best_val_f1,
		}
		print(
			f"Epoch {epoch + 1}/{epochs} | "
			f"train_loss={train_stats['loss']:.4f} | val_loss={val_stats['loss']:.4f} | "
			f"train_f1={train_stats['f1']:.4f} | val_f1={val_stats['f1']:.4f}"
		)

	writer.close()
	return history


if __name__ == "__main__":
	train_unext()

