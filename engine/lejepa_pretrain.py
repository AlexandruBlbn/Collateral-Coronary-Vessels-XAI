"""
DenseLeJEPA pretraining for coronary X-ray angiography with Frangi vesselness.

This script trains a shared encoder using the LeJEPA objective (LeCun, 2025)
with SIGReg collapse prevention (Balestriero & LeCun, 2025) on the ARCADE
dataset. The invariance loss operates on DENSE token predictions: the predictor
predicts target token embeddings from context tokens, and MSE is computed
between predicted and actual target embeddings.

Usage:
    python engine/lejepa_pretrain.py --config config/lejepa_config.yaml
"""

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import LeJepaDenseDataset  # noqa: E402
try:
    from data.frangi_cache import FrangiCache, precompute_all  # noqa: E402
    _HAS_FRANGI_CACHE = True
except ImportError:
    FrangiCache = None
    precompute_all = None
    _HAS_FRANGI_CACHE = False
from zoo.jepa_models import DenseLeJepaModel, compute_distance_weights  # noqa: E402
from zoo.sigreg import SIGRegLoss  # noqa: E402
from utils.helpers import set_seed  # noqa: E402


# ─────────────────────────────────────────────
# Config & utilities
# ─────────────────────────────────────────────


def _load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _amp_dtype(precision: str):
    """Map precision string to torch dtype.

    Returns ``None`` for CPU training (no AMP).
    """
    precision = str(precision).lower()
    if precision in {"float16", "fp16"}:
        return torch.float16
    if precision in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return None


def _model_ref(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel if present — always returns the base module."""
    return model.module if hasattr(model, "module") else model


def _to_three_channels(x: torch.Tensor) -> torch.Tensor:
    """Convert 1- or 2-channel tensor to 3-channel for ``make_grid`` visualization.

    - 3 channels: returned unchanged.
    - 2 channels: [R=gray, G=Frangi, B=gray].
    - 1 channel:  triplicated.
    - Otherwise:  first 3 channels sliced.
    """
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 2:
        return torch.cat([x[:, :1], x[:, 1:2], x[:, :1]], dim=1)
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x[:, :3]


# ─────────────────────────────────────────────
# Build helpers
# ─────────────────────────────────────────────


def _build_dataset(cfg: dict):
    """Build :class:`LeJepaDenseDataset` from config; FrangiCache is optional."""
    data_cfg = cfg["data"]
    input_size = int(cfg["model"]["input_size"])
    max_samples = data_cfg.get("max_samples", "all")
    use_frangi = bool(data_cfg.get("use_frangi", False))

    frangi_cache = None
    if use_frangi and _HAS_FRANGI_CACHE:
        frangi_cache = FrangiCache(
            cache_dir=data_cfg["frangi_cache_dir"],
            image_size=input_size,
        )

    dataset = LeJepaDenseDataset(
        base_dataset_json=data_cfg["base_dataset_json"],
        crops_json_path=data_cfg["crops_json_path"],
        root_dir=data_cfg["root_dir"],
        num_global=int(data_cfg.get("num_global", 2)),
        num_local=int(data_cfg.get("num_local", 4)),
        global_size=int(data_cfg.get("global_size", 224)),
        local_size=int(data_cfg.get("local_size", 96)),
        max_jitter=int(data_cfg.get("max_jitter", 4)),
        num_vessel_classes=int(data_cfg.get("num_vessel_classes", 26)),
        use_frangi=use_frangi,
        frangi_cache=frangi_cache,
        max_samples=max_samples,
    )

    return dataset, frangi_cache


def _build_loader(dataset: LeJepaDenseDataset, cfg: dict) -> DataLoader:
    """Build :class:`DataLoader` from config."""
    data_cfg = cfg["data"]
    num_workers = int(data_cfg.get("num_workers", 4))

    kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 16)),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "drop_last": bool(data_cfg.get("drop_last", True)),
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))

    return DataLoader(dataset, **kwargs)


def _build_model(cfg: dict, device: torch.device) -> DenseLeJepaModel:
    """Build :class:`DenseLeJepaModel` from config."""
    model_cfg = cfg["model"]
    model = DenseLeJepaModel(
        encoder_name=model_cfg["backbone"],
        proj_dim=model_cfg["proj_dim"],
        in_channels=model_cfg["in_channels"],
        deep_supervision=model_cfg.get("deep_supervision", False),
        deep_supervision_out_indices=model_cfg.get("deep_supervision_out_indices", (2, 3)),
    ).to(device)
    return model


def _build_loss(cfg: dict, device: torch.device) -> SIGRegLoss:
    """Build :class:`SIGRegLoss` from config."""
    lejepa_cfg = cfg["lejepa"]
    sigreg = SIGRegLoss(
        knots=lejepa_cfg["sigreg_knots"],
        t_max=lejepa_cfg["sigreg_t_max"],
        num_projections=lejepa_cfg["sigreg_num_projections"],
    ).to(device)
    return sigreg


def _build_optimizer(model: nn.Module, cfg: dict) -> AdamW:
    """Build :class:`AdamW` optimizer with betas support."""
    opt_cfg = cfg["optimizer"]
    betas = opt_cfg.get("betas", (0.9, 0.999))
    if isinstance(betas, list):
        betas = tuple(betas)
    return AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 5e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
        betas=betas,
    )


def _build_scheduler(optimizer: AdamW, cfg: dict):
    """Build warmup linear LR + cosine annealing scheduler.

    When ``warmup_epochs <= 0``, only cosine annealing is used.
    """
    total_epochs = int(cfg["optimizer"].get("epochs", 300))
    warmup_epochs = int(cfg["optimizer"].get("warmup_epochs", 0))

    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=1e-6)

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=max(1, warmup_epochs),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=1e-6,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────


def _log_attention_maps(
    writer: SummaryWriter,
    model: DenseLeJepaModel,
    context_tokens: torch.Tensor,
    pred_dense: torch.Tensor,
    global_crops: torch.Tensor,
    step: int,
    max_images: int = 4,
) -> None:
    """Log 3 types of attention visualisation to TensorBoard.

    1. **Backbone saliency** — L2 channel norm from the last feature map,
       upsampled to 256×256.
    2. **Predictor interaction** — cosine-similarity between predicted target
       tokens and context tokens, averaged over the target dimension and
       reshaped to a spatial grid.
    3. **Saliency overlay** — heatmap overlay of the saliency map on the
       grayscale input (red channel emphasis).
    """
    with torch.no_grad():
        n = min(int(max_images), global_crops.size(0))
        if n <= 0:
            return

        # 1. Backbone saliency
        first_global = global_crops[:, 0, :, :, :]  # [B, 2, Hg, Wg]
        model.encode(first_global)
        saliency = model.get_backbone_saliency()  # [B, H', W']
        saliency_up = F.interpolate(
            saliency.unsqueeze(1),  # [B, 1, H', W']
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )  # [B, 1, 256, 256]
        saliency_grid = make_grid(saliency_up[:n], nrow=n, padding=2, normalize=True)
        writer.add_image("Attention/backbone_saliency", saliency_grid, step)

        # 2. Predictor interaction
        interaction = DenseLeJepaModel.get_predictor_interaction(
            context_tokens[:n], pred_dense[:n]
        )  # [n, L_t, L_c]
        avg_interaction = interaction.mean(dim=1)  # [n, L_c]

        # Reshape to spatial grid (best effort)
        L_c = context_tokens.shape[1]
        try:
            h_sp = int(math.isqrt(L_c))
            if h_sp * h_sp == L_c:
                inter_map = avg_interaction.view(n, 1, h_sp, h_sp)
            else:
                # Find best factorisation
                for factor in range(int(math.isqrt(L_c)), 0, -1):
                    if L_c % factor == 0:
                        inter_map = avg_interaction.view(n, 1, factor, L_c // factor)
                        break
                else:
                    inter_map = avg_interaction.view(n, 1, 1, L_c)
        except Exception:
            inter_map = avg_interaction.view(n, 1, 1, L_c)

        inter_grid = make_grid(inter_map, nrow=n, padding=2, normalize=True)
        writer.add_image("Attention/predictor_interaction", inter_grid, step)

        # 3. Saliency RGB overlay
        input_gray = first_global[:n, :1, :, :]  # [n, 1, Hg, Wg]
        input_gray = (input_gray + 1.0) / 2.0  # rescale [-1, 1] → [0, 1]

        if input_gray.shape[-1] != 256 or input_gray.shape[-2] != 256:
            input_gray = F.interpolate(
                input_gray, size=(256, 256), mode="bilinear", align_corners=False
            )

        overlay_rgb = input_gray.repeat(1, 3, 1, 1) * 0.6
        heatmap_rgb = torch.zeros_like(overlay_rgb)        # black base
        heatmap_rgb[:, 0:1, :, :] = saliency_up[:n]        # red channel only
        overlay = (overlay_rgb + 0.4 * heatmap_rgb).clamp(0.0, 1.0)

        overlay_grid = make_grid(overlay, nrow=n, padding=2)
        writer.add_image("Attention/saliency_overlay", overlay_grid, step)


# ─────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────


def _checkpoint_state(
    model: nn.Module,
    optimizer: AdamW,
    scheduler,
    scaler,
    epoch: int,
    best_loss: float,
    history: list,
    config: dict,
) -> dict:
    """Assemble checkpoint dictionary."""
    return {
        "epoch": int(epoch),
        "model_state_dict": _model_ref(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "best_loss": float(best_loss),
        "history": history,
        "config": config,
    }


# ─────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────


def train(config_path: str) -> None:
    """Main DenseLeJEPA pretraining loop."""
    cfg = _load_config(config_path)
    experiment_name = cfg["experiment_name"]
    lejepa_cfg = cfg["lejepa"]
    train_cfg = cfg.get("training", {})
    sys_cfg = cfg["system"]

    set_seed(int(sys_cfg.get("seed", 42)))

    device = torch.device(
        sys_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        print("Warning: CUDA not available, running on CPU.")

    precision = str(sys_cfg.get("precision", "bfloat16"))
    amp_dtype = _amp_dtype(precision)
    amp_enabled = device.type == "cuda" and amp_dtype is not None

    # ── Build components ──────────────────────────────────────────────
    dataset, frangi_cache = _build_dataset(cfg)
    train_loader = _build_loader(dataset, cfg)
    model = _build_model(cfg, device)
    sigreg_loss_fn = _build_loss(cfg, device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    # Mixed-precision scaler — only needed for fp16; bf16 and fp32 skip it
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(amp_enabled and amp_dtype == torch.float16)
    )

    # ── Hyper-parameters ──────────────────────────────────────────────
    lamb = float(lejepa_cfg["lamb"])
    accum_steps = int(train_cfg.get("accum_steps", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 20))
    attention_log_every = int(train_cfg.get("attention_log_every", 200))
    save_every = int(train_cfg.get("save_every", 10))
    epochs = int(cfg["optimizer"].get("epochs", 300))

    # ── Directories ───────────────────────────────────────────────────
    checkpoint_root = (
        Path(sys_cfg.get("save_dir", "./checkpoints/lejepa/")) / experiment_name
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path("runs") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ── Frangi cache warmup (only when Frangi is enabled) ─────────────
    if frangi_cache is not None:
        print("[FrangiCache] Precomputing Frangi responses for all samples...")
        precompute_all(dataset.samples, frangi_cache)
        print("[FrangiCache] Precomputation complete.")

    writer = SummaryWriter(log_dir=str(log_dir))

    # ── Resume ────────────────────────────────────────────────────────
    resume_path = str(sys_cfg.get("resume", "")).strip()
    start_epoch = 0
    best_loss = float("inf")
    history = []

    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        _model_ref(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_loss = float(ckpt.get("best_loss", best_loss))
        history = ckpt.get("history", [])
        print(f"Resumed from {resume_path} at epoch {start_epoch}.")

    print(
        f"Starting DenseLeJEPA pretraining: {epochs} epochs, "
        f"{len(dataset)} samples."
    )

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_inv = 0.0
        running_sigreg = 0.0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}",
        )

        for step, batch in pbar:
            # -- Move data to device -----------------------------------
            global_crops = batch["global_crops"].to(device, non_blocking=True)
            local_crops = batch["local_crops"].to(device, non_blocking=True)
            global_boxes = batch["global_boxes"].to(device, non_blocking=True)
            local_boxes = batch["local_boxes"].to(device, non_blocking=True)

            B = global_crops.shape[0]
            ng = global_crops.shape[1]
            nl = local_crops.shape[1]

            # Flatten batch + views for shared encoder
            g_flat = global_crops.reshape(B * ng, *global_crops.shape[2:])
            l_flat = local_crops.reshape(B * nl, *local_crops.shape[2:])

            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with amp_ctx:
                # ── Encode global crops (context) ──
                # model.encode() returns list of [B*ng, N, proj_dim] per level
                ctx_tokens_list, _, _ = model.encode(g_flat)
                num_levels = len(ctx_tokens_list)
                D = ctx_tokens_list[0].shape[-1]
                L_ctx = ctx_tokens_list[0].shape[1]

                # Reshape each level: [B*ng, N, D] → [B, ng, N, D]
                ctx_tokens_by_level = [
                    t.reshape(B, ng, L_ctx, D) for t in ctx_tokens_list
                ]

                # ── Encode local crops (target) ──
                tgt_tokens_list, _, _ = model.encode(l_flat)
                L_tgt = tgt_tokens_list[0].shape[1]

                tgt_tokens_by_level = [
                    t.reshape(B, nl, L_tgt, D) for t in tgt_tokens_list
                ]

                # ── Multi-level prediction loss ──
                inv_loss = torch.tensor(0.0, device=device)
                ctx_boxes_for_pred = global_boxes[:, 0, :]  # [B, 4]

                # Compute distance weights once (same across levels)
                dist_weights = compute_distance_weights(
                    ctx_boxes_for_pred, local_boxes
                )  # [B, nl, L_tgt]

                for level_idx in range(num_levels):
                    ctx_tokens = ctx_tokens_by_level[level_idx]   # [B, ng, L_ctx, D]
                    tgt_tokens = tgt_tokens_by_level[level_idx]   # [B, nl, L_tgt, D]

                    # Predict: first global crop → all local crops
                    ctx_for_pred = ctx_tokens[:, 0, :, :]  # [B, L_ctx, D]

                    pred_dense_list = []
                    for li in range(nl):
                        tgt_box = local_boxes[:, li, :]  # [B, 4]
                        pd, _ = model.predictors[level_idx](
                            ctx_for_pred, ctx_boxes_for_pred, tgt_box, L_tgt
                        )
                        pred_dense_list.append(pd)

                    pred_dense = torch.stack(pred_dense_list, dim=1)  # [B, nl, L_tgt, D]

                    # V-JEPA 2.1: distance-weighted MSE
                    tgt_flat = tgt_tokens.reshape(B * nl, L_tgt, D)
                    pred_flat = pred_dense.reshape(B * nl, L_tgt, D)
                    level_mse = F.mse_loss(pred_flat, tgt_flat, reduction='none')  # [B*nl, L_tgt, D]

                    # Apply distance weights: weight per target token
                    w = dist_weights.reshape(B * nl, L_tgt, 1)  # [B*nl, L_tgt, 1]
                    level_loss = (level_mse * w).mean()

                    # Level weighting: earlier layers get lower weight
                    level_weight = 1.0 / (2 ** (num_levels - 1 - level_idx))
                    inv_loss = inv_loss + level_weight * level_loss

                inv_loss = inv_loss / num_levels

                # ── SIGReg on ALL projected tokens (last level only) ──
                last_ctx = ctx_tokens_by_level[-1]  # [B, ng, L_ctx, D]
                last_tgt = tgt_tokens_by_level[-1]  # [B, nl, L_tgt, D]
                all_tokens = torch.cat(
                    [
                        last_ctx.reshape(B * ng * L_ctx, D),
                        last_tgt.reshape(B * nl * L_tgt, D),
                    ],
                    dim=0,
                )
                sigreg_loss_val = sigreg_loss_fn(all_tokens)

                # ── Combined loss ──
                loss = (1.0 - lamb) * inv_loss + lamb * sigreg_loss_val

            # -- Backward ----------------------------------------------
            if not torch.isfinite(loss):
                print(
                    f"WARNING: Non-finite loss at epoch {epoch + 1}, "
                    f"step {step + 1}. Skipping."
                )
                continue

            scaled_loss = loss / accum_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = ((step + 1) % accum_steps == 0) or (
                (step + 1) == len(train_loader)
            )
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    )

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # -- Metrics & logging -------------------------------------
            inv_val = float(inv_loss.detach().item())
            sigreg_val = float(sigreg_loss_val.detach().item())
            loss_val = float(loss.detach().item())
            running_loss += loss_val
            running_inv += inv_val
            running_sigreg += sigreg_val

            global_step = epoch * len(train_loader) + step
            pbar.set_postfix(
                {
                    "loss": f"{running_loss / (step + 1):.4f}",
                    "inv": f"{inv_val:.4f}",
                    "sigreg": f"{sigreg_val:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            if (step + 1) % max(1, log_every) == 0:
                writer.add_scalar(
                    "Train/Loss", running_loss / (step + 1), global_step
                )
                writer.add_scalar(
                    "Train/Invariance", running_inv / (step + 1), global_step
                )
                writer.add_scalar(
                    "Train/SIGReg", running_sigreg / (step + 1), global_step
                )
                writer.add_scalar(
                    "Train/LR", optimizer.param_groups[0]["lr"], global_step
                )

            if attention_log_every > 0 and (global_step + 1) % attention_log_every == 0:
                with torch.no_grad():
                    _log_attention_maps(
                        writer=writer,
                        model=model,
                        context_tokens=ctx_for_pred,
                        pred_dense=pred_dense_list[0],
                        global_crops=global_crops,
                        step=global_step,
                    )

        # -- End of epoch ----------------------------------------------
        scheduler.step()

        epoch_loss = running_loss / max(1, len(train_loader))
        epoch_inv = running_inv / max(1, len(train_loader))
        epoch_sigreg = running_sigreg / max(1, len(train_loader))

        writer.add_scalar("Epoch/Loss", epoch_loss, epoch)
        writer.add_scalar("Epoch/Invariance", epoch_inv, epoch)
        writer.add_scalar("Epoch/SIGReg", epoch_sigreg, epoch)
        writer.add_scalar("Epoch/LR", optimizer.param_groups[0]["lr"], epoch)

        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "inv_loss": epoch_inv,
                "sigreg_loss": epoch_sigreg,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | "
            f"Inv: {epoch_inv:.4f} | SIGReg: {epoch_sigreg:.4f}"
        )

        # -- Checkpointing ---------------------------------------------
        state = _checkpoint_state(
            model, optimizer, scheduler, scaler, epoch, best_loss, history, cfg
        )
        torch.save(state, checkpoint_root / "last_model.pth")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(state, checkpoint_root / "best_model.pth")
            torch.save(
                _model_ref(model).backbone.state_dict(),
                checkpoint_root / "best_backbone.pth",
            )

        if (epoch + 1) % max(1, save_every) == 0:
            torch.save(
                _model_ref(model).backbone.state_dict(),
                checkpoint_root / f"backbone_ep{epoch + 1}.pth",
            )

        with open(checkpoint_root / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────
    summary = {
        "best_loss": best_loss,
        "final_loss": history[-1]["loss"] if history else None,
        "epochs": len(history),
        "dataset_size": len(dataset),
    }
    with open(checkpoint_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    writer.close()
    print("DenseLeJEPA pretraining complete.")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DenseLeJEPA on ARCADE coronary X-ray angiography."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/lejepa_config.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
