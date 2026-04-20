import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.helpers import set_seed
from zoo.jepa_models import DenseLeJepaModel


@dataclass
class SyntaxRecord:
    sample_id: str
    image_rel: str
    label_rel: str


def _resolve_path(project_root: Path, json_path: Path, rel_path: str) -> Path:
    candidates = [
        project_root / rel_path,
        json_path.parent / rel_path,
        json_path.parent.parent / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Unable to resolve path from json entry: {rel_path}")


def _load_syntax_train_records(dataset_json: Path, project_root: Path) -> List[SyntaxRecord]:
    with dataset_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "train" not in data:
        raise ValueError(f"Missing 'train' split in {dataset_json}")
    if "syntax" not in data["train"]:
        raise ValueError(f"Missing 'syntax' category in train split of {dataset_json}")

    records: List[SyntaxRecord] = []
    for sample_id, sample in data["train"]["syntax"].items():
        image_rel = sample.get("data")
        label_rel = sample.get("label")
        if not isinstance(image_rel, str) or not image_rel:
            continue
        if not isinstance(label_rel, str) or not label_rel:
            continue

        # Validate early so runtime errors happen before training starts.
        _resolve_path(project_root, dataset_json, image_rel)
        _resolve_path(project_root, dataset_json, label_rel)

        records.append(SyntaxRecord(sample_id=str(sample_id), image_rel=image_rel, label_rel=label_rel))

    if len(records) == 0:
        raise RuntimeError("No valid syntax train samples found in dataset json")

    return records


def _select_subset(records: Sequence[SyntaxRecord], subset_size: int, seed: int) -> List[SyntaxRecord]:
    subset_size = min(subset_size, len(records))
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[i] for i in indices[:subset_size]]


def _load_crops_meta(crops_json: Path) -> Dict[str, dict]:
    if not crops_json.exists():
        print(f"[WARN] crops json missing, using random fallback crops: {crops_json}")
        return {}

    try:
        with crops_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            print("[WARN] crops json is not a dict, using random fallback crops")
            return {}
        print(f"Loaded crops metadata for {len(meta)} images")
        return meta
    except Exception as exc:
        print(f"[WARN] failed to parse crops json ({exc}), using random fallback crops")
        return {}


def _safe_box(v: dict, h_img: int, w_img: int) -> Optional[Tuple[int, int, int, int]]:
    try:
        y = int(round(float(v["y"])))
        x = int(round(float(v["x"])))
        h = int(round(float(v["h"])))
        w = int(round(float(v["w"])))
    except Exception:
        return None

    if h <= 1 or w <= 1:
        return None
    if y < 0 or x < 0:
        return None
    if y + h > h_img or x + w > w_img:
        return None
    return y, x, h, w


def _random_resized_crop_box(
    h_img: int,
    w_img: int,
    scale: Tuple[float, float],
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    max_tries: int = 10,
) -> Tuple[int, int, int, int]:
    area = float(h_img * w_img)
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    for _ in range(max_tries):
        target_area = area * random.uniform(scale[0], scale[1])
        aspect = math.exp(random.uniform(log_ratio[0], log_ratio[1]))

        h = int(round(math.sqrt(target_area / aspect)))
        w = int(round(math.sqrt(target_area * aspect)))
        if h <= h_img and w <= w_img and h > 1 and w > 1:
            y = random.randint(0, h_img - h)
            x = random.randint(0, w_img - w)
            return y, x, h, w

    side = min(h_img, w_img)
    y = (h_img - side) // 2
    x = (w_img - side) // 2
    return y, x, side, side


def _crop_with_resize(img_t: torch.Tensor, box: Tuple[int, int, int, int], out_size: int) -> torch.Tensor:
    y, x, h, w = box
    crop = TF.crop(img_t, y, x, h, w)
    return TF.resize(
        crop,
        [out_size, out_size],
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )


class SyntaxDenseLowDataDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SyntaxRecord],
        dataset_json: Path,
        project_root: Path,
        crops_meta: Dict[str, dict],
        num_global: int = 2,
        num_local: int = 4,
        global_size: int = 224,
        local_size: int = 128,
        max_jitter: int = 4,
        global_scale_fallback: Tuple[float, float] = (0.50, 1.0),
        local_scale_fallback: Tuple[float, float] = (0.10, 0.35),
        noise_prob: float = 0.8,
        noise_sigma_min: float = 0.01,
        noise_sigma_max: float = 0.05,
        context_whole_image: bool = False,
    ):
        self.records = list(records)
        self.dataset_json = dataset_json
        self.project_root = project_root
        self.crops_meta = crops_meta
        self.num_global = num_global
        self.num_local = num_local
        self.global_size = global_size
        self.local_size = local_size
        self.max_jitter = max_jitter
        self.global_scale_fallback = global_scale_fallback
        self.local_scale_fallback = local_scale_fallback
        self.noise_prob = noise_prob
        self.noise_sigma_min = noise_sigma_min
        self.noise_sigma_max = noise_sigma_max
        self.context_whole_image = context_whole_image
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.records)

    def _apply_jitter(self, box: Tuple[int, int, int, int], h_img: int, w_img: int) -> Tuple[int, int, int, int]:
        y, x, h, w = box
        jy = random.randint(-self.max_jitter, self.max_jitter)
        jx = random.randint(-self.max_jitter, self.max_jitter)
        y = max(0, min(y + jy, h_img - h))
        x = max(0, min(x + jx, w_img - w))
        return y, x, h, w

    @staticmethod
    def _center_inside(box: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
        y, x, h, w = box
        oy, ox, oh, ow = outer
        cy = y + 0.5 * h
        cx = x + 0.5 * w
        return (oy <= cy <= oy + oh) and (ox <= cx <= ox + ow)

    def _extract_candidates(
        self,
        image_rel: str,
        h_img: int,
        w_img: int,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        meta = self.crops_meta.get(image_rel, None)
        if not isinstance(meta, dict):
            return [], []

        g = []
        l = []
        for c in meta.get("global_crops", []):
            box = _safe_box(c, h_img, w_img)
            if box is not None:
                g.append(box)
        for c in meta.get("local_crops", []):
            box = _safe_box(c, h_img, w_img)
            if box is not None:
                l.append(box)
        return g, l

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        r = self.records[index]
        img_path = _resolve_path(self.project_root, self.dataset_json, r.image_rel)

        img_np = np.array(Image.open(img_path).convert("L"))
        h_img, w_img = img_np.shape
        img_np = self.clahe.apply(img_np)

        img_t = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0
        img_t = img_t * 2.0 - 1.0

        g_candidates, l_candidates = self._extract_candidates(r.image_rel, h_img, w_img)

        # Whole-image spatial augmentations to preserve context/target geometric consistency.
        if random.random() < 0.5:
            img_t = TF.hflip(img_t)
            g_candidates = [(y, w_img - (x + w), h, w) for (y, x, h, w) in g_candidates]
            l_candidates = [(y, w_img - (x + w), h, w) for (y, x, h, w) in l_candidates]
        if random.random() < 0.5:
            img_t = TF.vflip(img_t)
            g_candidates = [(h_img - (y + h), x, h, w) for (y, x, h, w) in g_candidates]
            l_candidates = [(h_img - (y + h), x, h, w) for (y, x, h, w) in l_candidates]

        if random.random() < self.noise_prob:
            noise_sigma = random.uniform(self.noise_sigma_min, self.noise_sigma_max)
            img_t = (img_t + torch.randn_like(img_t) * noise_sigma).clamp(-1.0, 1.0)

        if len(g_candidates) >= self.num_global:
            g_selected_base = random.sample(g_candidates, k=self.num_global)
        elif len(g_candidates) > 0:
            g_selected_base = random.choices(g_candidates, k=self.num_global)
        else:
            g_selected_base = [
                _random_resized_crop_box(h_img, w_img, self.global_scale_fallback)
                for _ in range(self.num_global)
            ]

        if self.context_whole_image:
            full_box = (0, 0, h_img, w_img)
            if self.num_global <= 1:
                g_selected = [full_box]
            else:
                g_selected = [full_box] + g_selected_base[: self.num_global - 1]
        else:
            g_selected = g_selected_base

        global_crops = []
        global_boxes = []
        global_abs = []
        for b in g_selected:
            b = self._apply_jitter(b, h_img, w_img)
            global_crops.append(_crop_with_resize(img_t, b, self.global_size))
            y, x, h, w = b
            global_boxes.append([y / h_img, x / w_img, h / h_img, w / w_img])
            global_abs.append(b)

        candidate_local_inside = [c for c in l_candidates if any(self._center_inside(c, g) for g in global_abs)]
        local_pool = candidate_local_inside if len(candidate_local_inside) > 0 else l_candidates
        if len(local_pool) >= self.num_local:
            l_selected = random.sample(local_pool, k=self.num_local)
        elif len(local_pool) > 0:
            l_selected = random.choices(local_pool, k=self.num_local)
        else:
            # Fallback requested by user: fast random policy.
            l_selected = [
                _random_resized_crop_box(h_img, w_img, self.local_scale_fallback)
                for _ in range(self.num_local)
            ]

        local_crops = []
        local_boxes = []
        for b in l_selected:
            b = self._apply_jitter(b, h_img, w_img)
            local_crops.append(_crop_with_resize(img_t, b, self.local_size))
            y, x, h, w = b
            local_boxes.append([y / h_img, x / w_img, h / h_img, w / w_img])

        return {
            "global_crops": torch.stack(global_crops),
            "global_boxes": torch.tensor(global_boxes, dtype=torch.float32),
            "local_crops": torch.stack(local_crops),
            "local_boxes": torch.tensor(local_boxes, dtype=torch.float32),
            "full_image": _crop_with_resize(img_t, (0, 0, h_img, w_img), self.global_size),
            "full_box": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "sample_id": r.sample_id,
        }


class SyntaxDiagnosticsDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SyntaxRecord],
        dataset_json: Path,
        project_root: Path,
        image_size: int = 256,
    ):
        self.records = list(records)
        self.dataset_json = dataset_json
        self.project_root = project_root
        self.image_size = image_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        r = self.records[index]
        image_path = _resolve_path(self.project_root, self.dataset_json, r.image_rel)
        label_path = _resolve_path(self.project_root, self.dataset_json, r.label_rel)

        img_np = np.array(Image.open(image_path).convert("L"))
        msk_np = np.array(Image.open(label_path).convert("L"))

        img_np = self.clahe.apply(img_np)
        img_t = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0
        img_t = img_t * 2.0 - 1.0
        img_t = TF.resize(
            img_t,
            [self.image_size, self.image_size],
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

        mask_t = torch.from_numpy((msk_np > 0).astype(np.float32)).unsqueeze(0)
        mask_t = TF.resize(
            mask_t,
            [self.image_size, self.image_size],
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )

        return {
            "image": img_t,
            "mask": mask_t,
            "sample_id": r.sample_id,
        }


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # proj: (N, D)
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-1)
        return statistic.mean()


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _make_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = 0.05
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _apply_random_block_mask(
    x: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    mask_value: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Apply random square block masking directly in pixel space.
    This approximates context token masking while keeping target as full image.
    """
    b, c, h, w = x.shape
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")

    gh = h // patch_size
    gw = w // patch_size
    if gh <= 0 or gw <= 0:
        raise ValueError(f"patch_size={patch_size} too large for input {h}x{w}")

    num_patches = gh * gw
    k = int(round(float(mask_ratio) * num_patches))
    k = max(0, min(num_patches, k))

    pixel_mask = torch.zeros((b, 1, h, w), dtype=torch.bool, device=x.device)

    if k == 0:
        return x, pixel_mask, 0.0

    for i in range(b):
        idx = torch.randperm(num_patches, device=x.device)[:k]
        for p in idx.tolist():
            py = (p // gw) * patch_size
            px = (p % gw) * patch_size
            pixel_mask[i, :, py : py + patch_size, px : px + patch_size] = True

    x_masked = x.clone()
    x_masked[pixel_mask.expand(-1, c, -1, -1)] = mask_value
    effective_ratio = float(pixel_mask.float().mean().item())
    return x_masked, pixel_mask, effective_ratio


def _pixel_mask_to_token_mask(pixel_mask: torch.Tensor, spatial_dims: Tuple[int, int]) -> torch.Tensor:
    h_t, w_t = spatial_dims
    token_mask = F.interpolate(pixel_mask.float(), size=(h_t, w_t), mode="nearest")
    return token_mask.flatten(1).to(torch.bool)


def _dense_masked_visible_loss(
    pred_dense: torch.Tensor,
    target_dense: torch.Tensor,
    token_mask: torch.Tensor,
    masked_weight: float,
    visible_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    # SimSiam Negative Cosine Similarity (1 - cos_sim)
    per_token_loss = 1.0 - F.cosine_similarity(pred_dense, target_dense, dim=-1)  # [B, L]

    masked_count = int(token_mask.sum().item())
    total_count = int(token_mask.numel())
    visible_mask = ~token_mask
    visible_count = total_count - masked_count

    if masked_count > 0:
        loss_masked = per_token_loss[token_mask].mean()
    else:
        loss_masked = per_token_loss.mean()

    if visible_count > 0:
        loss_visible = per_token_loss[visible_mask].mean()
    else:
        loss_visible = per_token_loss.mean()

    w_m = max(0.0, float(masked_weight))
    w_v = max(0.0, float(visible_weight))
    w_sum = max(1e-8, w_m + w_v)
    w_m /= w_sum
    w_v /= w_sum

    loss_dense = w_m * loss_masked + w_v * loss_visible
    token_mask_ratio = float(masked_count / max(1, total_count))
    return loss_dense, loss_masked, loss_visible, token_mask_ratio


def _box_centers_inside(outer_boxes: torch.Tensor, inner_boxes: torch.Tensor) -> torch.Tensor:
    """
    Check whether the center of each inner box lies inside the corresponding outer box.
    Boxes are normalized [y, x, h, w].
    """
    oy, ox, oh, ow = outer_boxes.unbind(dim=-1)
    iy, ix, ih, iw = inner_boxes.unbind(dim=-1)
    cy = iy + 0.5 * ih
    cx = ix + 0.5 * iw
    return (cy >= oy) & (cy <= oy + oh) & (cx >= ox) & (cx <= ox + ow)


def _select_context_target_views(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build JEPA pairs from crop metadata.
    Context is a sampled global crop; target is a local crop whose center is inside context
    when available (fallback: any local crop, then alternative global crop).
    """
    global_crops = batch["global_crops"].to(device, non_blocking=True)
    global_boxes = batch["global_boxes"].to(device, non_blocking=True)
    local_crops = batch["local_crops"].to(device, non_blocking=True)
    local_boxes = batch["local_boxes"].to(device, non_blocking=True)

    if global_crops.ndim != 5 or global_boxes.ndim != 3:
        raise RuntimeError("Expected global crop tensors shaped [B,G,C,H,W] and [B,G,4]")
    if local_crops.ndim != 5 or local_boxes.ndim != 3:
        raise RuntimeError("Expected local crop tensors shaped [B,L,C,H,W] and [B,L,4]")

    bsz, num_global = global_crops.shape[:2]
    if num_global <= 0:
        raise RuntimeError("Need at least one global crop per sample")

    batch_idx = torch.arange(bsz, device=device)
    context_idx = torch.randint(0, num_global, (bsz,), device=device)
    context_view = global_crops[batch_idx, context_idx]
    context_boxes = global_boxes[batch_idx, context_idx]

    num_local = int(local_crops.shape[1])
    if num_local > 0:
        contains = _box_centers_inside(context_boxes[:, None, :], local_boxes)
        local_idx = torch.empty((bsz,), dtype=torch.long, device=device)
        for i in range(bsz):
            valid = torch.nonzero(contains[i], as_tuple=False).flatten()
            if valid.numel() == 0:
                local_idx[i] = torch.randint(0, num_local, (1,), device=device)[0]
            else:
                pick = torch.randint(0, valid.numel(), (1,), device=device)
                local_idx[i] = valid[pick][0]

        target_view = local_crops[batch_idx, local_idx]
        target_boxes = local_boxes[batch_idx, local_idx]
        return context_view, context_boxes, target_view, target_boxes

    if num_global == 1:
        return context_view, context_boxes, context_view, context_boxes

    offset = torch.randint(1, num_global, (bsz,), device=device)
    target_idx = (context_idx + offset) % num_global
    target_view = global_crops[batch_idx, target_idx]
    target_boxes = global_boxes[batch_idx, target_idx]
    return context_view, context_boxes, target_view, target_boxes


def _resolve_stage_weights(num_stages: int, stage_weights: Sequence[float]) -> List[float]:
    if num_stages <= 0:
        raise ValueError("num_stages must be > 0")

    if len(stage_weights) == 0:
        return [1.0 / num_stages for _ in range(num_stages)]
    if len(stage_weights) != num_stages:
        raise ValueError(
            f"Expected {num_stages} deep stage weights, got {len(stage_weights)}"
        )

    clean = [max(0.0, float(w)) for w in stage_weights]
    total = sum(clean)
    if total <= 0.0:
        return [1.0 / num_stages for _ in range(num_stages)]
    return [w / total for w in clean]


def _variance_floor_loss(x: torch.Tensor, min_std: float, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return F.relu(float(min_std) - std).mean()


def _batch_embedding_health_metrics(x: torch.Tensor) -> Dict[str, float]:
    std_mean = float(x.std(dim=0, unbiased=False).mean().item())

    if x.shape[0] < 2:
        return {
            "std_mean": std_mean,
            "offdiag_cos_mean": 0.0,
        }

    z = F.normalize(x, dim=1, eps=1e-8)
    sim = z @ z.t()
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    offdiag_cos_mean = float(sim[mask].mean().item())
    return {
        "std_mean": std_mean,
        "offdiag_cos_mean": offdiag_cos_mean,
    }


def _grad_conflict_metrics(
    loss_dense: torch.Tensor,
    loss_sig: torch.Tensor,
    wrt_tensor: torch.Tensor,
) -> Optional[Dict[str, float]]:
    """
    Compute dense-vs-sigreg gradient alignment on a shared representation tensor.
    cos < 0 means the two objectives are in direct conflict on that representation.
    """
    g_dense = torch.autograd.grad(
        loss_dense,
        wrt_tensor,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    g_sig = torch.autograd.grad(
        loss_sig,
        wrt_tensor,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]

    if g_dense is None or g_sig is None:
        return None

    gd = g_dense.detach().reshape(-1)
    gs = g_sig.detach().reshape(-1)
    gd_norm = float(gd.norm(p=2).item())
    gs_norm = float(gs.norm(p=2).item())
    if gd_norm <= 0.0 or gs_norm <= 0.0:
        return None

    cos = float(torch.dot(gd, gs).item() / (gd_norm * gs_norm + 1e-12))
    return {
        "cos": cos,
        "dense_norm": gd_norm,
        "sig_norm": gs_norm,
        "conflict": 1.0 if cos < 0.0 else 0.0,
    }


def _log_epoch_crop_grid(
    writer: SummaryWriter,
    batch: Dict[str, torch.Tensor],
    epoch: int,
    max_rows: int = 4,
) -> None:
    if "context_masked" in batch and ("target_view" in batch or "target_full" in batch):
        context_batch = batch["context_masked"]
        target_batch = batch["target_view"] if "target_view" in batch else batch["target_full"]
        b = min(max_rows, context_batch.shape[0])

        target_h = max(int(context_batch.shape[-2]), int(target_batch.shape[-2]))
        target_w = max(int(context_batch.shape[-1]), int(target_batch.shape[-1]))

        tiles: List[torch.Tensor] = []
        for i in range(b):
            context = context_batch[i]
            target = target_batch[i]
            if context.shape[-2:] != (target_h, target_w):
                context = TF.resize(
                    context,
                    [target_h, target_w],
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            if target.shape[-2:] != (target_h, target_w):
                target = TF.resize(
                    target,
                    [target_h, target_w],
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            context = (context * 0.5 + 0.5).clamp(0.0, 1.0)
            target = (target * 0.5 + 0.5).clamp(0.0, 1.0)
            tiles.append(context)
            tiles.append(target)

        grid = torchvision.utils.make_grid(tiles, nrow=2, padding=2)
        writer.add_image("Qualitative/ContextMaskedVsTarget", grid, global_step=epoch)
        return

    g = batch["global_crops"]
    l = batch["local_crops"]
    b = min(max_rows, g.shape[0])

    target_h = max(int(g.shape[-2]), int(l.shape[-2]))
    target_w = max(int(g.shape[-1]), int(l.shape[-1]))

    tiles: List[torch.Tensor] = []
    for i in range(b):
        context = g[i, 0]
        local = l[i, 0]
        if context.shape[-2:] != (target_h, target_w):
            context = TF.resize(
                context,
                [target_h, target_w],
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        if local.shape[-2:] != (target_h, target_w):
            local = TF.resize(
                local,
                [target_h, target_w],
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        context = (context * 0.5 + 0.5).clamp(0.0, 1.0)
        local = (local * 0.5 + 0.5).clamp(0.0, 1.0)
        tiles.append(context)
        tiles.append(local)

    grid = torchvision.utils.make_grid(tiles, nrow=2, padding=2)
    writer.add_image("Qualitative/ContextVsLocal", grid, global_step=epoch)


def _mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    a = mask_a > 0.5
    b = mask_b > 0.5
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    if union <= 0:
        return 0.0
    return float(inter / union)


def run_knn_diagnostics(
    model: DenseLeJepaModel,
    diag_loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    max_vis_pairs: int = 8,
) -> None:
    model.eval()

    pooled_feats: List[torch.Tensor] = []
    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    sample_ids: List[str] = []

    with torch.no_grad():
        for batch in tqdm(diag_loader, desc=f"Diag epoch {epoch + 1}", leave=False):
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)

            _, (h_f, w_f), raw_feat = model(img)

            msk_small = F.interpolate(msk, size=(h_f, w_f), mode="nearest") > 0.5
            feat_tokens = raw_feat.flatten(2).permute(0, 2, 1)  # [B, N, C]
            msk_tokens = msk_small.flatten(1)  # [B, N]

            for i in range(img.shape[0]):
                token_mask = msk_tokens[i]
                if token_mask.any():
                    feat = feat_tokens[i][token_mask].mean(dim=0)
                else:
                    feat = feat_tokens[i].mean(dim=0)
                feat = F.normalize(feat, dim=0)
                pooled_feats.append(feat.cpu())
                images.append(img[i].cpu())
                masks.append(msk[i].cpu())
                sample_ids.append(str(batch["sample_id"][i]))

    if len(pooled_feats) < 2:
        print("[WARN] Not enough diagnostic samples for KNN logging")
        return

    feat_mat = torch.stack(pooled_feats, dim=0)  # [N, C]
    sim = feat_mat @ feat_mat.t()
    sim.fill_diagonal_(-1.0)
    nn_idx = sim.argmax(dim=1)
    nn_sim = sim.gather(1, nn_idx[:, None]).squeeze(1)

    ious = []
    for i in range(len(images)):
        j = int(nn_idx[i].item())
        iou = _mask_iou(masks[i], masks[j])
        ious.append(iou)

    writer.add_scalar("DiagKNN/top1_cosine_mean", float(nn_sim.mean().item()), epoch)
    writer.add_scalar("DiagKNN/top1_mask_iou_mean", float(np.mean(ious)), epoch)

    n_vis = min(max_vis_pairs, len(images))
    fig, axes = plt.subplots(n_vis, 4, figsize=(12, 3 * n_vis))
    if n_vis == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(n_vis):
        j = int(nn_idx[row].item())
        sim_ij = float(nn_sim[row].item())
        iou_ij = float(ious[row])

        img_a = (images[row] * 0.5 + 0.5).clamp(0.0, 1.0).squeeze(0).numpy()
        img_b = (images[j] * 0.5 + 0.5).clamp(0.0, 1.0).squeeze(0).numpy()
        msk_a = masks[row].squeeze(0).numpy()
        msk_b = masks[j].squeeze(0).numpy()

        axes[row, 0].imshow(img_a, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].imshow(msk_a, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 2].imshow(img_b, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 3].imshow(msk_b, cmap="gray", vmin=0.0, vmax=1.0)

        for c in range(4):
            axes[row, c].axis("off")

        axes[row, 0].set_title(f"Anchor #{sample_ids[row]}")
        axes[row, 1].set_title("Anchor mask")
        axes[row, 2].set_title(f"NN #{sample_ids[j]} | cos={sim_ij:.3f}")
        axes[row, 3].set_title(f"NN mask | IoU={iou_ij:.3f}")

    plt.tight_layout()
    writer.add_figure("DiagKNN/top1_retrievals", fig, global_step=epoch)
    plt.close(fig)


def train_one_epoch(
    model: DenseLeJepaModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    sigreg: SIGReg,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    lambda_sigreg: float,
    sigreg_balance_mode: str,
    sigreg_balanced_ratio: float,
    sigreg_scale_min: float,
    sigreg_scale_max: float,
    context_mask_ratio: float,
    context_mask_patch: int,
    context_mask_value: float,
    deep_supervision: bool,
    deep_stage_weights: Sequence[float],
    dense_masked_weight: float,
    dense_visible_weight: float,
    normalize_sigreg_input: bool,
    lambda_varfloor: float,
    varfloor_min_std: float,
    target_stop_grad: bool,
    collapse_check_every: int,
    collapse_warmup_epochs: int,
    collapse_cos_threshold: float,
    collapse_std_threshold: float,
    collapse_patience: int,
    grad_conflict_every: int,
    grad_conflict_warmup_epochs: int,
    scaler: Optional[torch.amp.GradScaler],
) -> Tuple[float, float, float, Dict[str, torch.Tensor]]:
    model.train()

    running_total = 0.0
    running_dense = 0.0
    running_dense_masked = 0.0
    running_dense_visible = 0.0
    running_sig = 0.0
    running_varfloor = 0.0
    running_sig_contrib = 0.0
    running_mask_ratio = 0.0
    running_token_mask_ratio = 0.0
    running_grad_cos = 0.0
    running_grad_conflict = 0.0
    running_grad_obs = 0
    running_health_cos = 0.0
    running_health_std = 0.0
    running_health_obs = 0
    collapse_strikes = 0
    first_batch_cpu: Optional[Dict[str, torch.Tensor]] = None

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train epoch {epoch + 1}")
    for step, batch in pbar:
        context_view, context_boxes, target_view, target_boxes = _select_context_target_views(batch, device)

        context_masked, pixel_mask, eff_mask_ratio = _apply_random_block_mask(
            context_view,
            mask_ratio=context_mask_ratio,
            patch_size=context_mask_patch,
            mask_value=context_mask_value,
        )

        if first_batch_cpu is None:
            first_batch_cpu = {
                "context_masked": context_masked.detach().cpu(),
                "target_view": target_view.detach().cpu(),
            }

        use_amp = scaler is not None
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

        with amp_ctx:
            stage_log_items: List[Tuple[str, float, float, float, float]] = []

            if deep_supervision:
                ctx_pack = model.encode_multiscale(context_masked)
                if target_stop_grad:
                    with torch.no_grad():
                        tgt_pack = model.encode_multiscale(target_view)
                else:
                    tgt_pack = model.encode_multiscale(target_view)

                ctx_stages = ctx_pack["stages"]
                tgt_stages = tgt_pack["stages"]
                if len(ctx_stages) != len(tgt_stages):
                    raise RuntimeError(
                        f"Context/target stage count mismatch: {len(ctx_stages)} vs {len(tgt_stages)}"
                    )

                weights = _resolve_stage_weights(len(ctx_stages), deep_stage_weights)
                stage_losses: List[torch.Tensor] = []
                stage_masked_losses: List[torch.Tensor] = []
                stage_visible_losses: List[torch.Tensor] = []
                pooled_parts: List[torch.Tensor] = []
                token_mask_ratio_weighted = 0.0
                ctx_for_grad = None

                for i, (ctx_stage, tgt_stage, stage_w) in enumerate(zip(ctx_stages, tgt_stages, weights)):
                    stage_key = str(ctx_stage["stage_key"])
                    ctx_proj_stage = ctx_stage["proj_tokens"]
                    tgt_proj_stage = tgt_stage["proj_tokens"]
                    predictor = model.get_stage_predictor(stage_key)

                    pred_dense_stage, _ = predictor(
                        ctx_proj_stage,
                        context_boxes,
                        target_boxes,
                        num_target_tokens=tgt_proj_stage.shape[1],
                    )

                    target_for_dense = tgt_proj_stage.detach() if target_stop_grad else tgt_proj_stage
                    token_mask_stage = _pixel_mask_to_token_mask(pixel_mask, tgt_stage["spatial_dims"])
                    dense_stage, dense_stage_masked, dense_stage_visible, token_stage_ratio = _dense_masked_visible_loss(
                        pred_dense_stage,
                        target_for_dense,
                        token_mask_stage,
                        masked_weight=dense_masked_weight,
                        visible_weight=dense_visible_weight,
                    )

                    stage_losses.append(dense_stage * stage_w)
                    stage_masked_losses.append(dense_stage_masked * stage_w)
                    stage_visible_losses.append(dense_stage_visible * stage_w)
                    token_mask_ratio_weighted += float(stage_w) * float(token_stage_ratio)

                    stage_log_items.append(
                        (
                            stage_key,
                            float(dense_stage.item()),
                            float(dense_stage_masked.item()),
                            float(dense_stage_visible.item()),
                            float(token_stage_ratio),
                        )
                    )

                    pooled_parts.append(ctx_proj_stage.mean(dim=1))
                    pooled_parts.append(tgt_proj_stage.mean(dim=1))
                    if i == len(ctx_stages) - 1:
                        ctx_for_grad = ctx_proj_stage

                loss_dense = torch.stack(stage_losses).sum()
                loss_dense_masked = torch.stack(stage_masked_losses).sum()
                loss_dense_visible = torch.stack(stage_visible_losses).sum()
                token_mask_ratio = token_mask_ratio_weighted
                pooled = torch.cat(pooled_parts, dim=0)
            else:
                ctx_proj, _, _ = model(context_masked)
                if target_stop_grad:
                    with torch.no_grad():
                        tgt_proj, tgt_hw, _ = model(target_view)
                else:
                    tgt_proj, tgt_hw, _ = model(target_view)

                pred_dense, _ = model.predictor(
                    ctx_proj,
                    context_boxes,
                    target_boxes,
                    num_target_tokens=tgt_proj.shape[1],
                )

                token_mask = _pixel_mask_to_token_mask(pixel_mask, tgt_hw)
                target_for_dense = tgt_proj.detach() if target_stop_grad else tgt_proj
                loss_dense, loss_dense_masked, loss_dense_visible, token_mask_ratio = _dense_masked_visible_loss(
                    pred_dense,
                    target_for_dense,
                    token_mask,
                    masked_weight=dense_masked_weight,
                    visible_weight=dense_visible_weight,
                )
                pooled = torch.cat([ctx_proj.mean(dim=1), tgt_proj.mean(dim=1)], dim=0)
                ctx_for_grad = ctx_proj

            pooled_sig = F.layer_norm(pooled, (pooled.shape[-1],)) if normalize_sigreg_input else pooled
            loss_sig = sigreg(pooled_sig)
            if lambda_varfloor > 0.0:
                loss_varfloor = _variance_floor_loss(pooled_sig, min_std=varfloor_min_std)
            else:
                loss_varfloor = torch.tensor(0.0, device=device)

            if sigreg_balance_mode == "fixed":
                sig_scale = torch.tensor(1.0, device=device)
                sig_term = loss_sig
                loss = (1.0 - lambda_sigreg) * loss_dense + lambda_sigreg * sig_term
            else:
                # Keep SIGReg in the same order of magnitude as dense loss to avoid
                # dominating optimization when raw SIGReg values are large.
                sig_scale = (loss_dense.detach() / loss_sig.detach().clamp_min(1e-6)).clamp(
                    min=sigreg_scale_min,
                    max=sigreg_scale_max,
                )
                sig_term = sigreg_balanced_ratio * sig_scale * loss_sig
                loss = (1.0 - lambda_sigreg) * loss_dense + lambda_sigreg * sig_term

            loss = loss + lambda_varfloor * loss_varfloor

            sig_contrib = lambda_sigreg * sig_term
            varfloor_contrib = lambda_varfloor * loss_varfloor

            health_metrics = None
            if collapse_check_every > 0 and epoch >= collapse_warmup_epochs and (step % collapse_check_every == 0):
                health_metrics = _batch_embedding_health_metrics(pooled_sig.detach())

            grad_metrics = None
            if grad_conflict_every > 0 and epoch >= grad_conflict_warmup_epochs and (step % grad_conflict_every == 0):
                grad_metrics = _grad_conflict_metrics(loss_dense=loss_dense, loss_sig=loss_sig, wrt_tensor=ctx_for_grad)

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_total += float(loss.item())
        running_dense += float(loss_dense.item())
        running_dense_masked += float(loss_dense_masked.item())
        running_dense_visible += float(loss_dense_visible.item())
        running_sig += float(loss_sig.item())
        running_varfloor += float(loss_varfloor.item())
        running_sig_contrib += float(sig_contrib.item())
        running_mask_ratio += eff_mask_ratio
        running_token_mask_ratio += token_mask_ratio

        global_step = epoch * len(loader) + step
        writer.add_scalar("Train/LeJepaLoss", float(loss.item()), global_step)
        writer.add_scalar("Train/DenseMSE", float(loss_dense.item()), global_step)
        writer.add_scalar("Train/DenseMaskedMSE", float(loss_dense_masked.item()), global_step)
        writer.add_scalar("Train/DenseVisibleMSE", float(loss_dense_visible.item()), global_step)
        writer.add_scalar("Train/SIGReg", float(loss_sig.item()), global_step)
        writer.add_scalar("Train/VarFloor", float(loss_varfloor.item()), global_step)
        writer.add_scalar("Train/VarFloorContribution", float(varfloor_contrib.item()), global_step)
        writer.add_scalar("Train/SIGRegScale", float(sig_scale.item()), global_step)
        writer.add_scalar("Train/SIGRegContribution", float(sig_contrib.item()), global_step)
        writer.add_scalar("Train/ContextMaskRatio", float(eff_mask_ratio), global_step)
        writer.add_scalar("Train/TokenMaskRatio", float(token_mask_ratio), global_step)

        for stage_key, dense_s, dense_m_s, dense_v_s, token_m_s in stage_log_items:
            writer.add_scalar(f"TrainStage/{stage_key}/DenseMSE", dense_s, global_step)
            writer.add_scalar(f"TrainStage/{stage_key}/DenseMaskedMSE", dense_m_s, global_step)
            writer.add_scalar(f"TrainStage/{stage_key}/DenseVisibleMSE", dense_v_s, global_step)
            writer.add_scalar(f"TrainStage/{stage_key}/TokenMaskRatio", token_m_s, global_step)

        if health_metrics is not None:
            running_health_cos += health_metrics["offdiag_cos_mean"]
            running_health_std += health_metrics["std_mean"]
            running_health_obs += 1
            writer.add_scalar("Train/BatchOffdiagCosMean", health_metrics["offdiag_cos_mean"], global_step)
            writer.add_scalar("Train/BatchStdMean", health_metrics["std_mean"], global_step)

            is_collapsed_now = (
                health_metrics["offdiag_cos_mean"] >= collapse_cos_threshold
                and health_metrics["std_mean"] <= collapse_std_threshold
            )
            collapse_strikes = collapse_strikes + 1 if is_collapsed_now else 0
            writer.add_scalar("Train/CollapseStrikeCount", float(collapse_strikes), global_step)
            if collapse_patience > 0 and collapse_strikes >= collapse_patience:
                raise RuntimeError(
                    "Collapse detector triggered: "
                    f"offdiag_cos_mean={health_metrics['offdiag_cos_mean']:.5f} >= {collapse_cos_threshold:.5f} "
                    f"and std_mean={health_metrics['std_mean']:.5f} <= {collapse_std_threshold:.5f} "
                    f"for {collapse_strikes} checks"
                )

        if grad_metrics is not None:
            running_grad_cos += grad_metrics["cos"]
            running_grad_conflict += grad_metrics["conflict"]
            running_grad_obs += 1
            writer.add_scalar("Train/GradCosineDenseVsSIGReg_ctx", grad_metrics["cos"], global_step)
            writer.add_scalar("Train/GradNormDense_ctx", grad_metrics["dense_norm"], global_step)
            writer.add_scalar("Train/GradNormSIGReg_ctx", grad_metrics["sig_norm"], global_step)
            writer.add_scalar("Train/GradConflictFlag_ctx", grad_metrics["conflict"], global_step)

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Train/LR", float(lr), global_step)

        postfix = {
            "loss": f"{running_total / (step + 1):.4f}",
            "dense": f"{running_dense / (step + 1):.4f}",
            "dmask": f"{running_dense_masked / (step + 1):.4f}",
            "dvis": f"{running_dense_visible / (step + 1):.4f}",
            "sig": f"{running_sig / (step + 1):.4f}",
            "vfl": f"{running_varfloor / (step + 1):.4f}",
            "sigc": f"{running_sig_contrib / (step + 1):.4f}",
            "mratio": f"{running_mask_ratio / (step + 1):.3f}",
            "tmratio": f"{running_token_mask_ratio / (step + 1):.3f}",
            "lr": f"{lr:.2e}",
        }
        if running_health_obs > 0:
            postfix["bcos"] = f"{running_health_cos / running_health_obs:.3f}"
            postfix["bstd"] = f"{running_health_std / running_health_obs:.4f}"
        if running_grad_obs > 0:
            postfix["gcos"] = f"{running_grad_cos / running_grad_obs:.3f}"
            postfix["gconf"] = f"{running_grad_conflict / running_grad_obs:.2f}"
        pbar.set_postfix(**postfix)

    assert first_batch_cpu is not None
    n = float(len(loader))
    return running_total / n, running_dense / n, running_sig / n, first_batch_cpu


def _parse_csv_ints(csv_value: str) -> Tuple[int, ...]:
    value = csv_value.strip()
    if value == "":
        return tuple()
    out = []
    for tok in value.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(int(tok))
    return tuple(out)


def _parse_csv_floats(csv_value: str) -> Tuple[float, ...]:
    value = csv_value.strip()
    if value == "":
        return tuple()
    out = []
    for tok in value.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(float(tok))
    return tuple(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Syntax-only low-data Dense LeJEPA prototype")
    parser.add_argument("--dataset-json", type=str, default="data/ARCADE/processed/dataset.json")
    parser.add_argument("--crops-json", type=str, default="data/ARCADE/processed/dataset_crops.json")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="swinv2_tiny_window8_256")
    parser.add_argument("--proj-dim", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lambda-sigreg", type=float, default=0.0)
    parser.add_argument(
        "--sigreg-balance-mode",
        type=str,
        default="ratio",
        choices=["fixed", "ratio"],
        help="fixed: use raw SIGReg term; ratio: scale SIGReg relative to dense loss.",
    )
    parser.add_argument(
        "--sigreg-balanced-ratio",
        type=float,
        default=1.0,
        help="In ratio mode, target multiplier for balanced SIGReg before lambda weighting.",
    )
    parser.add_argument(
        "--sigreg-scale-min",
        type=float,
        default=2e-2,
        help="Lower clamp for dynamic SIGReg scaling in ratio mode.",
    )
    parser.add_argument(
        "--sigreg-scale-max",
        type=float,
        default=1.0,
        help="Upper clamp for dynamic SIGReg scaling in ratio mode.",
    )

    parser.add_argument("--num-global", type=int, default=2)
    parser.add_argument("--num-local", type=int, default=4)
    parser.add_argument("--global-size", type=int, default=224)
    parser.add_argument("--local-size", type=int, default=128)
    parser.add_argument(
        "--context-whole-image",
        action="store_true",
        help="Force context view to use the whole image as one global crop.",
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=0.8,
        help="Probability of adding Gaussian noise to each training image.",
    )
    parser.add_argument("--noise-sigma-min", type=float, default=0.01)
    parser.add_argument("--noise-sigma-max", type=float, default=0.05)
    parser.add_argument(
        "--context-mask-ratio",
        type=float,
        default=0.60,
        help="Fraction of context patches to mask (V-JEPA-style masked context).",
    )
    parser.add_argument(
        "--context-mask-patch",
        type=int,
        default=16,
        help="Pixel patch size used for random block masking of context.",
    )
    parser.add_argument(
        "--context-mask-value",
        type=float,
        default=0.0,
        help="Fill value for masked context patches in normalized [-1,1] space.",
    )
    parser.add_argument(
        "--dense-masked-weight",
        type=float,
        default=0.7,
        help="Weight for dense token loss over masked token positions.",
    )
    parser.add_argument(
        "--dense-visible-weight",
        type=float,
        default=0.3,
        help="Weight for dense token loss over visible token positions.",
    )
    parser.add_argument(
        "--normalize-sigreg-input",
        dest="normalize_sigreg_input",
        action="store_true",
        help="Apply layer normalization before feeding pooled vectors into SIGReg.",
    )
    parser.add_argument(
        "--no-normalize-sigreg-input",
        dest="normalize_sigreg_input",
        action="store_false",
        help="Disable layer normalization before SIGReg.",
    )
    parser.set_defaults(normalize_sigreg_input=True)
    parser.add_argument(
        "--lambda-varfloor",
        type=float,
        default=0.0,
        help="Weight for variance-floor anti-collapse regularizer on pooled embeddings.",
    )
    parser.add_argument(
        "--varfloor-min-std",
        type=float,
        default=0.10,
        help="Minimum desired batch std per embedding dimension for variance-floor loss.",
    )
    parser.add_argument(
        "--deep-supervision",
        action="store_true",
        help="Enable deep self-supervision over multiple backbone stages.",
    )
    parser.add_argument(
        "--deep-stages",
        type=str,
        default="2,3",
        help="Comma-separated stage indices to supervise when deep supervision is enabled.",
    )
    parser.add_argument(
        "--deep-stage-weights",
        type=str,
        default="",
        help="Optional comma-separated per-stage weights matching --deep-stages order.",
    )
    parser.add_argument(
        "--target-stop-grad",
        dest="target_stop_grad",
        action="store_true",
        help="Stop gradients through the target branch (not recommended for shared-encoder JEPA).",
    )
    parser.add_argument(
        "--no-target-stop-grad",
        dest="target_stop_grad",
        action="store_false",
        help="Allow gradients through the target branch (recommended default).",
    )
    parser.set_defaults(target_stop_grad=True)
    parser.add_argument(
        "--collapse-check-every",
        type=int,
        default=20,
        help="Run batch collapse diagnostics every N steps (0 disables).",
    )
    parser.add_argument(
        "--collapse-warmup-epochs",
        type=int,
        default=2,
        help="Start collapse checks from this epoch.",
    )
    parser.add_argument(
        "--collapse-cos-threshold",
        type=float,
        default=0.995,
        help="Collapse flag threshold for mean off-diagonal cosine similarity.",
    )
    parser.add_argument(
        "--collapse-std-threshold",
        type=float,
        default=0.01,
        help="Collapse flag threshold for mean per-dim batch std.",
    )
    parser.add_argument(
        "--collapse-patience",
        type=int,
        default=8,
        help="Number of consecutive collapse flags before stopping training (0 disables fail-fast).",
    )
    parser.add_argument(
        "--grad-conflict-every",
        type=int,
        default=20,
        help="Compute dense-vs-sigreg gradient cosine every N steps (0 disables).",
    )
    parser.add_argument(
        "--grad-conflict-warmup-epochs",
        type=int,
        default=0,
        help="Start gradient conflict diagnostics from this epoch.",
    )

    parser.add_argument("--diag-samples", type=int, default=128)
    parser.add_argument("--diag-batch-size", type=int, default=16)
    parser.add_argument("--diag-max-vis", type=int, default=8)

    parser.add_argument("--log-dir", type=str, default="runs/lejepa_syntax_lowdata_prototype")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/lejepa_syntax_lowdata_prototype")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    deep_stage_indices = _parse_csv_ints(args.deep_stages)
    deep_stage_weights = _parse_csv_floats(args.deep_stage_weights)
    if args.deep_supervision and len(deep_stage_indices) == 0:
        raise ValueError("--deep-supervision requires at least one stage index in --deep-stages")
    if args.deep_supervision and len(deep_stage_weights) > 0 and len(deep_stage_weights) != len(deep_stage_indices):
        raise ValueError(
            "--deep-stage-weights must have the same number of entries as --deep-stages"
        )
    if not args.deep_supervision:
        deep_stage_weights = tuple()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    dataset_json = (project_root / args.dataset_json).resolve()
    crops_json = (project_root / args.crops_json).resolve()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    cfg_dump = vars(args).copy()
    cfg_dump["deep_stages_resolved"] = list(deep_stage_indices)
    cfg_dump["deep_stage_weights_resolved"] = list(deep_stage_weights)
    with open(os.path.join(args.log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    records = _load_syntax_train_records(dataset_json=dataset_json, project_root=project_root)
    subset = _select_subset(records, subset_size=args.subset_size, seed=args.seed)
    diag_subset = subset[: min(args.diag_samples, len(subset))]

    crops_meta = _load_crops_meta(crops_json)

    train_ds = SyntaxDenseLowDataDataset(
        records=subset,
        dataset_json=dataset_json,
        project_root=project_root,
        crops_meta=crops_meta,
        num_global=args.num_global,
        num_local=args.num_local,
        global_size=args.global_size,
        local_size=args.local_size,
        noise_prob=args.noise_prob,
        noise_sigma_min=args.noise_sigma_min,
        noise_sigma_max=args.noise_sigma_max,
        context_whole_image=args.context_whole_image,
    )
    diag_ds = SyntaxDiagnosticsDataset(
        records=diag_subset,
        dataset_json=dataset_json,
        project_root=project_root,
        image_size=256,
    )

    worker_kwargs = {
        "num_workers": args.num_workers,
        "worker_init_fn": _worker_init_fn,
        "persistent_workers": args.num_workers > 0,
        "pin_memory": True,
    }

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
        **worker_kwargs,
    )
    diag_loader = DataLoader(
        diag_ds,
        batch_size=args.diag_batch_size,
        shuffle=False,
        drop_last=False,
        **worker_kwargs,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Train loader is empty. Check subset size and dataset paths.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DenseLeJepaModel(
        encoder_name=args.encoder_name,
        proj_dim=args.proj_dim,
        deep_supervision=args.deep_supervision,
        deep_supervision_out_indices=deep_stage_indices if len(deep_stage_indices) > 0 else (2, 3),
    ).to(device)
    sigreg = SIGReg().to(device)

    wd_params, no_wd_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Exclude ALL predictor weights (SimSiam heads) and 1D tensors (LayerNorms/Biases)
        if "predictor" in name or param.ndim <= 1:
            no_wd_params.append(param)
        else:
            wd_params.append(param)

    optimizer = AdamW(
        [
            {"params": wd_params, "weight_decay": args.weight_decay},
            {"params": no_wd_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = _make_lr_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    use_amp = (device.type == "cuda") and (not args.disable_amp)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss, dense_loss, sig_loss, first_batch = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            sigreg=sigreg,
            device=device,
            writer=writer,
            epoch=epoch,
            lambda_sigreg=args.lambda_sigreg,
            sigreg_balance_mode=args.sigreg_balance_mode,
            sigreg_balanced_ratio=args.sigreg_balanced_ratio,
            sigreg_scale_min=args.sigreg_scale_min,
            sigreg_scale_max=args.sigreg_scale_max,
            context_mask_ratio=args.context_mask_ratio,
            context_mask_patch=args.context_mask_patch,
            context_mask_value=args.context_mask_value,
            deep_supervision=args.deep_supervision,
            deep_stage_weights=deep_stage_weights,
            dense_masked_weight=args.dense_masked_weight,
            dense_visible_weight=args.dense_visible_weight,
            normalize_sigreg_input=args.normalize_sigreg_input,
            lambda_varfloor=args.lambda_varfloor,
            varfloor_min_std=args.varfloor_min_std,
            target_stop_grad=args.target_stop_grad,
            collapse_check_every=args.collapse_check_every,
            collapse_warmup_epochs=args.collapse_warmup_epochs,
            collapse_cos_threshold=args.collapse_cos_threshold,
            collapse_std_threshold=args.collapse_std_threshold,
            collapse_patience=args.collapse_patience,
            grad_conflict_every=args.grad_conflict_every,
            grad_conflict_warmup_epochs=args.grad_conflict_warmup_epochs,
            scaler=scaler,
        )

        writer.add_scalar("Epoch/LeJepaLoss", train_loss, epoch)
        writer.add_scalar("Epoch/DenseMSE", dense_loss, epoch)
        writer.add_scalar("Epoch/SIGReg", sig_loss, epoch)
        _log_epoch_crop_grid(writer, first_batch, epoch)

        run_knn_diagnostics(
            model=model,
            diag_loader=diag_loader,
            device=device,
            writer=writer,
            epoch=epoch,
            max_vis_pairs=args.diag_max_vis,
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "train_loss": train_loss,
        }
        torch.save(state, os.path.join(args.ckpt_dir, "last_model.pth"))

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(state, os.path.join(args.ckpt_dir, "best_model.pth"))

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={train_loss:.4f} dense={dense_loss:.4f} sigreg={sig_loss:.4f}"
        )

    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    main()