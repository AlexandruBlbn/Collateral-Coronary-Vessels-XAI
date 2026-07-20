"""
train_test.py — VasoJEPA Dry-Run on 10k Balanced Subset
Run from project root: python train_test.py [flags]

Flags:
  --no-ema            EMA-free (LeJEPA style, single encoder)
  --vessel-masking    Vessel-aware target patch selection
  --vessel-anchor     Vesselness anchor head (anti-collapse + vessel encoding)
  --epochs N          Override epoch count (default 80)
  --checkpoint-dir D  Override checkpoint directory (auto-generated if omitted)

Ablation grid:
  python train_test.py                                            # Baseline (EMA)
  python train_test.py --vessel-masking                           # A: EMA + vmask
  python train_test.py --no-ema                                   # B: noema (expect collapse)
  python train_test.py --no-ema --vessel-masking                  # C: noema + vmask
  python train_test.py --no-ema --vessel-anchor                   # D: noema + vanchor
  python train_test.py --no-ema --vessel-masking --vessel-anchor  # E: full VesselJEPA
"""
import sys
sys.path.append(".")

import os
import json
import random
import argparse
from tqdm import tqdm
import math
import torch
import matplotlib
matplotlib.use("Agg")
from torch.utils.data import DataLoader, Subset

from vasojepa.model import Model
from data.data import pretrain_dataset, paths

# ── Config ────────────────────────────────────────────────────────────────────

EPOCHS          = 80
BATCH_SIZE      = 32
ACCUM_STEPS     = 8
LR              = 1.5e-4
WEIGHT_DECAY    = 0.05
WARMUP_EPOCHS   = 5
NUM_WORKERS     = 4
SAVE_EVERY      = 20
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--no-ema", action="store_true", help="EMA-free (LeJEPA style)")
parser.add_argument("--vessel-masking", action="store_true", help="Vessel-aware target masking")
parser.add_argument("--vessel-anchor", action="store_true", help="Vesselness anchor head")
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--checkpoint-dir", type=str, default=None)
args = parser.parse_args()

EPOCHS = args.epochs

#auto checkpoint dir from config
if args.checkpoint_dir:
    CHECKPOINT_DIR = args.checkpoint_dir
else:
    tags = ["ema" if not args.no_ema else "noema"]
    if args.vessel_masking:
        tags.append("vmask")
    if args.vessel_anchor:
        tags.append("vanchor")
    CHECKPOINT_DIR = "checkpoints_" + "_".join(tags)

USE_EMA         = not args.no_ema
VESSEL_MASKING  = args.vessel_masking
VESSEL_ANCHOR   = args.vessel_anchor


# ── Balanced subset builder ────────────────────────────────────────────────────

def build_balanced_subset(total_n=10000, seed=42):
    """Randomly sample ~total_n images balanced across all subset keys."""
    with open(paths["Pretraining"], "r") as f:
        data = json.load(f)

    keys  = list(data.keys())
    counts = {k: len(data[k]) for k in keys}

    per_key = total_n // len(keys)                       # 2000

    indices = []
    offset = 0
    for k in keys:
        n_avail = counts[k]
        n_take  = min(per_key, n_avail)
        pool    = list(range(offset, offset + n_avail))
        indices.extend(random.sample(pool, n_take))
        offset += n_avail

    # Distribute remaining slots to subsets that have more images available
    remaining = total_n - len(indices)
    for k in keys:
        if remaining <= 0:
            break
        n_avail = counts[k]
        offset_start = offset - n_avail if k == keys[-1] else sum(counts[kk] for kk in keys[:keys.index(k)])
        taken = sum(1 for i in indices if offset_start <= i < offset_start + n_avail)
        if n_avail > taken:
            extra = min(remaining, n_avail - taken)
            pool = [i for i in range(offset_start, offset_start + n_avail) if i not in indices]
            indices.extend(random.sample(pool, extra))
            remaining -= extra

    random.shuffle(indices)
    return Subset(pretrain_dataset(), indices)


def report_subset_balance(indices, data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    keys = list(data.keys())
    offset = 0
    for k in keys:
        n = len(data[k])
        n_in = sum(1 for i in indices if offset <= i < offset + n)
        print(f"  {k:<25s}: {n_in:>5,} / {n:>6,}")
        offset += n


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return LR * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
    return 1e-5 + (LR - 1e-5) * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, epoch):
    lr = get_lr(epoch)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, avg_loss):
    path = os.path.join(CHECKPOINT_DIR, f"vasojepa_test_epoch{epoch:03d}.pt")
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "avg_loss":  avg_loss,
    }, path)
    print(f"  Saved checkpoint -> {path}")


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Resumed from epoch {ckpt['epoch']} (loss {ckpt['avg_loss']:.4f})")
    return ckpt["epoch"] + 1


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, epoch, total_epochs, global_step, total_steps):
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    n_batches  = 0
    n_grad_steps = 0
    last_x     = None
    last_prior = None

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True)

    for step, (x, prior) in enumerate(pbar):
        x     = x.to(DEVICE, non_blocking=True)
        prior = prior.to(DEVICE, non_blocking=True)
        last_x, last_prior = x, prior

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, loss_dict = model(x, prior, epoch, total_epochs)

        (loss / ACCUM_STEPS).backward()

        total_loss += loss.item()
        n_batches  += 1
        postfix = {
            "loss": f"{loss.item():.3f}",
            "dense": f"{loss_dict['dense']:.3f}",
            "f2_std": f"{loss_dict['f2_std']:.3f}",
            "tf2_std": f"{loss_dict['tf2_std']:.3f}",
        }
        if "anchor" in loss_dict:
            postfix["anchor"] = f"{loss_dict['anchor']:.3f}"
            postfix["cons"] = f"{loss_dict['consistency']:.3f}"
        pbar.set_postfix(**postfix)

        if (step + 1) % ACCUM_STEPS == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0) + \
                        torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            if hasattr(model, 'vessel_head'):
                grad_norm += torch.nn.utils.clip_grad_norm_(model.vessel_head.parameters(), 1.0)
            total_grad_norm += grad_norm
            n_grad_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            progress = min(global_step / total_steps, 1.0)
            model.update_target_encoder(progress)

    if (len(loader) % ACCUM_STEPS) != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0) + \
                    torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
        if hasattr(model, 'vessel_head'):
            grad_norm += torch.nn.utils.clip_grad_norm_(model.vessel_head.parameters(), 1.0)
        total_grad_norm += grad_norm
        n_grad_steps += 1
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        progress = min(global_step / total_steps, 1.0)
        model.update_target_encoder(progress)

    avg_loss = total_loss / max(n_batches, 1)
    avg_grad_norm = total_grad_norm / max(n_grad_steps, 1)
    return avg_loss, avg_grad_norm, last_x, last_prior, global_step


# ── Main ──────────────────────────────────────────────────────────────────────

def main(resume_from=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = Model(
        use_ema=USE_EMA,
        vessel_masking=VESSEL_MASKING,
        vessel_anchor=VESSEL_ANCHOR,
    ).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total:,}  |  Device: {DEVICE}")
    print(f"Config: ema={USE_EMA}  vessel_masking={VESSEL_MASKING}  vessel_anchor={VESSEL_ANCHOR}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Build balanced 10k subset
    random.seed(42)
    dataset = build_balanced_subset(total_n=10000, seed=42)
    print(f"\nBalanced subset ({len(dataset):,} images) break-down:")
    report_subset_balance(dataset.indices, paths["Pretraining"])

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    total_steps = EPOCHS * math.ceil(len(loader) / ACCUM_STEPS)
    print(f"\nBatches/epoch: {len(loader):,}  |  Effective batch: {BATCH_SIZE * ACCUM_STEPS}")
    print(f"Total optimizer steps: {total_steps:,}")

    start_epoch = 0
    global_step = 0
    if resume_from:
        start_epoch = load_checkpoint(model, optimizer, resume_from)
        # global_step reset to 0 for simplicity on resume; EMA schedule will ramp back up

    for epoch in range(start_epoch, EPOCHS):
        lr = set_lr(optimizer, epoch)
        print(f"\nEpoch {epoch:03d}/{EPOCHS}  lr={lr:.2e}")

        avg_loss, avg_grad_norm, last_x, last_prior, global_step = train_one_epoch(
            model, loader, optimizer, epoch, EPOCHS, global_step, total_steps
        )
        print(f"  avg loss: {avg_loss:.4f}  |  avg grad_norm (pre-clip): {avg_grad_norm:.2f}")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
            save_checkpoint(model, optimizer, epoch, avg_loss)


if __name__ == "__main__":
    main()
