import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
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

from data.MIM import MaskGenerator, XA170KImageDataset  # noqa: E402
from utils.helpers import set_seed  # noqa: E402
from zoo.mim import SimMIM  # noqa: E402


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _amp_dtype(precision: str):
    precision = str(precision).lower()
    if precision in {"float16", "fp16"}:
        return torch.float16
    if precision in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return None


def _model_ref(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _build_dataset(cfg: dict):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    sys_cfg = cfg["system"]

    dataset_style = str(data_cfg.get("dataset_style", "xa170k")).lower()
    if dataset_style != "xa170k":
        raise ValueError(
            f"Unsupported dataset_style '{dataset_style}'. This trainer currently supports 'xa170k'."
        )

    dataset = XA170KImageDataset(
        base_path=data_cfg["data_path"],
        input_size=int(model_cfg["input_size"]),
        sources=data_cfg.get("sources"),
        require_all_sources=bool(data_cfg.get("require_all_sources", True)),
    )

    full_size = len(dataset)
    max_samples = int(data_cfg.get("max_samples", 0) or 0)
    subset_seed = int(data_cfg.get("subset_seed", sys_cfg.get("seed", 42)))
    subset_indices = None

    if max_samples > 0 and max_samples < full_size:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(subset_seed)
        subset_indices = torch.randperm(full_size, generator=gen)[:max_samples].tolist()
        subset_indices.sort()
        dataset.image_paths = [dataset.image_paths[i] for i in subset_indices]

    meta = {
        "full_size": full_size,
        "selected_size": len(dataset),
        "max_samples": max_samples,
        "subset_seed": subset_seed,
        "subset_indices": subset_indices,
    }
    return dataset, meta


def _build_loader(dataset: XA170KImageDataset, cfg: dict) -> DataLoader:
    data_cfg = cfg["data"]
    num_workers = int(data_cfg.get("num_workers", 4))

    kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 32)),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "drop_last": bool(data_cfg.get("drop_last", True)),
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))

    return DataLoader(dataset, **kwargs)


def _build_scheduler(optimizer: AdamW, cfg: dict):
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


def _build_mask_generator(cfg: dict) -> MaskGenerator:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    sys_cfg = cfg["system"]

    return MaskGenerator(
        input_size=int(model_cfg["input_size"]),
        mask_block_size=int(data_cfg.get("mask_block_size", 32)),
        mask_ratio=float(data_cfg.get("mask_ratio", 0.65)),
        fixed_mask_blocks=data_cfg.get("effective_mask_blocks", None),
        seed=int(sys_cfg.get("seed", 42)),
    )


def _checkpoint_state(model, optimizer, scheduler, scaler, epoch, best_loss, history, config):
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


def _to_three_channels(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x[:, :1].repeat(1, 3, 1, 1)


def _log_reconstruction_triplet(
    writer: SummaryWriter,
    images: torch.Tensor,
    masks: torch.Tensor,
    reconstructions: torch.Tensor,
    step: int,
    max_images: int = 4,
) -> None:
    with torch.no_grad():
        n = min(int(max_images), images.size(0))
        if n <= 0:
            return

        img = images[:n].detach().float()
        rec_head = reconstructions[:n].detach().float()
        mask = masks[:n].detach().float().unsqueeze(1)
        masked = img * (1.0 - mask)
        rec_mask_only = rec_head * mask
        rec_composite = masked + rec_mask_only

        img = _to_three_channels(img).clamp(0.0, 1.0).cpu()
        masked = _to_three_channels(masked).clamp(0.0, 1.0).cpu()
        rec_head = _to_three_channels(rec_head).clamp(0.0, 1.0).cpu()
        rec_mask_only = _to_three_channels(rec_mask_only).clamp(0.0, 1.0).cpu()
        rec_composite = _to_three_channels(rec_composite).clamp(0.0, 1.0).cpu()

        vis_items = []
        for i in range(n):
            vis_items.extend([img[i], masked[i], rec_composite[i]])

        grid = make_grid(vis_items, nrow=3, padding=2)
        writer.add_image("Train/Reconstruction_inpainted_triplet", grid, step)

        # Keep this view to inspect exactly what the head predicts inside masked regions.
        mask_only_grid = make_grid(rec_mask_only, nrow=n, padding=2)
        writer.add_image("Train/Reconstruction_head_masked_pixels", mask_only_grid, step)

        # Additional view: raw output of reconstruction head before masking.
        head_grid = make_grid(rec_head, nrow=n, padding=2)
        writer.add_image("Train/Reconstruction_head_full", head_grid, step)


def train(config_path: str) -> None:
    cfg = _load_config(config_path)

    experiment_name = cfg["experiment_name"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimizer"]
    data_cfg = cfg["data"]
    loss_cfg = cfg.get("loss", {})
    train_cfg = cfg.get("training", {})
    sys_cfg = cfg["system"]

    set_seed(int(sys_cfg.get("seed", 42)))

    device = torch.device(sys_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available, running on CPU.")

    precision = str(sys_cfg.get("precision", "bfloat16"))
    amp_dtype = _amp_dtype(precision)
    amp_enabled = device.type == "cuda" and amp_dtype is not None

    mask_generator = _build_mask_generator(cfg)
    print(
        "Mask setup:",
        {
            "block_size": mask_generator.mask_block_size,
            "mask_blocks": mask_generator.mask_count,
            "token_count": mask_generator.token_count,
            "target_ratio": float(data_cfg.get("mask_ratio", 0.65)),
            "effective_ratio": mask_generator.effective_mask_ratio,
        },
    )

    dataset, subset_meta = _build_dataset(cfg)
    train_loader = _build_loader(dataset, cfg)

    model = SimMIM(
        backbone_name=model_cfg["backbone"],
        in_channels=int(model_cfg.get("in_channels", 1)),
        input_size=int(model_cfg.get("input_size", 256)),
        decoder_type=str(model_cfg.get("decoder_type", "pixelshuffle")),
        decoder_hidden_dim=int(model_cfg.get("decoder_hidden_dim", 512)),
        output_activation=str(model_cfg.get("output_activation", "sigmoid")),
        pixel_loss_weight=float(loss_cfg.get("pixel_weight", 1.0)),
        gradient_loss_weight=float(loss_cfg.get("gradient_weight", 0.2)),
        vessel_focus_weight=float(loss_cfg.get("vessel_focus_weight", 1.5)),
        vessel_prior_kernel_size=int(loss_cfg.get("vessel_prior_kernel_size", 9)),
    ).to(device)

    if torch.cuda.device_count() > 1 and bool(train_cfg.get("use_data_parallel", True)):
        model = nn.DataParallel(model)

    optimizer = AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
    )
    scheduler = _build_scheduler(optimizer, cfg)

    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp_enabled and amp_dtype == torch.float16))

    accum_steps = int(train_cfg.get("accum_steps", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 20))
    reconstruction_log_every = int(train_cfg.get("reconstruction_log_every", 200))
    reconstruction_max_images = int(train_cfg.get("reconstruction_max_images", 4))
    save_every = int(train_cfg.get("save_every", 10))
    epochs = int(opt_cfg.get("epochs", 300))

    checkpoint_root = Path(sys_cfg.get("save_dir", "./checkpoints/mim/")) / experiment_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path("runs") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    if subset_meta["subset_indices"] is not None:
        with open(checkpoint_root / "subset_indices.json", "w", encoding="utf-8") as f:
            json.dump(subset_meta["subset_indices"], f)
        with open(checkpoint_root / "subset_info.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "full_size": subset_meta["full_size"],
                    "selected_size": subset_meta["selected_size"],
                    "max_samples": subset_meta["max_samples"],
                    "subset_seed": subset_meta["subset_seed"],
                },
                f,
                indent=2,
            )
        print(
            "Using deterministic subset:",
            f"{subset_meta['selected_size']}/{subset_meta['full_size']} samples",
            f"(seed={subset_meta['subset_seed']}).",
        )
    else:
        print(f"Using full dataset: {subset_meta['selected_size']} samples.")

    writer = SummaryWriter(log_dir=str(log_dir))

    resume_path = str(sys_cfg.get("resume", "")).strip()
    start_epoch = 0
    best_loss = float("inf")
    history = []

    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        _model_ref(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None and scaler.is_enabled():
            scaler.load_state_dict(scaler_state)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))
        history = checkpoint.get("history", history)
        print(f"Resumed from {resume_path} at epoch {start_epoch}.")

    print(f"Start SimMIM training for {epochs} epochs on {len(dataset)} samples.")

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mask_generator.set_epoch(epoch=epoch, rank=0)

        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")

        for step, images in pbar:
            images = images.to(device, non_blocking=True)
            masks = mask_generator(batch_size=images.shape[0], device=device, dtype=images.dtype)
            loss_terms = {}

            amp_context = (
                torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
            )
            with amp_context:
                model_output = model(images, masks)
                if isinstance(model_output, (tuple, list)) and len(model_output) == 3:
                    loss, reconstructions, loss_terms = model_output
                elif isinstance(model_output, (tuple, list)) and len(model_output) == 2:
                    loss, reconstructions = model_output
                else:
                    raise RuntimeError(
                        "SimMIM forward must return (loss, reconstructions) or "
                        f"(loss, reconstructions, loss_terms), got type={type(model_output)}"
                    )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Encountered non-finite loss at epoch {epoch + 1}, step {step + 1}")

            scaled_loss = loss / accum_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader))
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().item())
            avg_loss = running_loss / float(step + 1)
            global_step = epoch * len(train_loader) + step
            pbar.set_postfix({"loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})

            if (step + 1) % max(1, log_every) == 0:
                writer.add_scalar("Train/Loss_iter", avg_loss, global_step)
                if isinstance(loss_terms, dict):
                    for name, value in loss_terms.items():
                        if torch.is_tensor(value):
                            scalar = float(value.detach().mean().item())
                        else:
                            scalar = float(value)
                        writer.add_scalar(f"Train/{name}", scalar, global_step)

            if reconstruction_log_every > 0 and (global_step + 1) % reconstruction_log_every == 0:
                _log_reconstruction_triplet(
                    writer=writer,
                    images=images,
                    masks=masks,
                    reconstructions=reconstructions,
                    step=global_step,
                    max_images=reconstruction_max_images,
                )

        scheduler.step()

        epoch_loss = running_loss / max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Train/Loss_epoch", epoch_loss, epoch)
        writer.add_scalar("Train/LR", current_lr, epoch)

        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "lr": current_lr,
                "effective_mask_ratio": mask_generator.effective_mask_ratio,
                "mask_blocks": mask_generator.mask_count,
            }
        )

        state = _checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_loss=best_loss,
            history=history,
            config=cfg,
        )
        torch.save(state, checkpoint_root / "last_model.pth")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state["best_loss"] = best_loss
            torch.save(state, checkpoint_root / "best_model.pth")
            torch.save(_model_ref(model).encoder.state_dict(), checkpoint_root / "best_backbone.pth")

        if (epoch + 1) % max(1, save_every) == 0:
            torch.save(_model_ref(model).encoder.state_dict(), checkpoint_root / f"backbone_ep{epoch + 1}.pth")

        with open(checkpoint_root / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    summary = {
        "best_loss": best_loss,
        "final_loss": history[-1]["loss"] if history else None,
        "epochs": len(history),
        "dataset_size": len(dataset),
        "dataset_full_size": subset_meta["full_size"],
        "dataset_selected_size": subset_meta["selected_size"],
        "subset_seed": subset_meta["subset_seed"],
        "max_samples": subset_meta["max_samples"],
        "effective_mask_ratio": mask_generator.effective_mask_ratio,
        "mask_blocks": mask_generator.mask_count,
        "token_count": mask_generator.token_count,
    }
    with open(checkpoint_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    writer.close()
    print("SimMIM training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SimMIM with SwinV2 on XA-170K style data.")
    parser.add_argument("--config", type=str, default="config/mim_config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
