"""
train.py — VasoJEPA v2 Pre-training
Run from project root: python train.py
"""

import sys
sys.path.append(".")

import os
from tqdm import tqdm
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from vasojepa.model import Model
from data.data import pretrain_dataset


# ── Config ────────────────────────────────────────────────────────────────────

EPOCHS          = 200
BATCH_SIZE      = 32
ACCUM_STEPS     = 8          # effective batch = 32 × 8 = 256
LR              = 1.5e-4
WEIGHT_DECAY    = 0.05
WARMUP_EPOCHS   = 10
NUM_WORKERS     = 4
SAVE_EVERY      = 25         # checkpoint + visualization every N epochs
LOG_EVERY       = 50         # log loss every N optimizer steps
CHECKPOINT_DIR  = "checkpoints"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ── Visualization ─────────────────────────────────────────────────────────────

def save_vessel_score(x, vessel_score, epoch):
    image = x[0, 0].detach().float().cpu()
    score = vessel_score[0].reshape(14, 14).detach().float().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap="gray");  axes[0].set_title("Input");       axes[0].axis("off")
    axes[1].imshow(score, cmap="hot");   axes[1].set_title("Vessel Score"); axes[1].axis("off")
    plt.tight_layout()
    path = os.path.join(CHECKPOINT_DIR, f"vessel_score_epoch{epoch:03d}.png")
    plt.savefig(path, dpi=100)
    plt.close()


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, avg_loss):
    path = os.path.join(CHECKPOINT_DIR, f"vasojepa_epoch{epoch:03d}.pt")
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "avg_loss":  avg_loss,
    }, path)
    print(f"  Saved checkpoint → {path}")


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Resumed from epoch {ckpt['epoch']} (loss {ckpt['avg_loss']:.4f})")
    return ckpt["epoch"] + 1


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    opt_steps  = 0
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

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            opt_steps += 1
            total_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                dense=f"{loss_dict['dense']:.3f}",
                cglt=f"{loss_dict['cglt']:.3f}",
                lds=f"{loss_dict['lds']:.3f}",
            )

    # Handle leftover steps (if dataset not divisible by ACCUM_STEPS)
    if (len(loader) % ACCUM_STEPS) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(opt_steps, 1)
    return avg_loss, last_x, last_prior


# ── Main ──────────────────────────────────────────────────────────────────────

def main(resume_from=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Model
    model = Model().to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total:,}  |  Device: {DEVICE}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Data
    dataset = pretrain_dataset()
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset):,} images  |  Batches/epoch: {len(loader):,}")
   
    start_epoch = 0
    if resume_from:
        start_epoch = load_checkpoint(model, optimizer, resume_from)

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        lr = set_lr(optimizer, epoch)
        print(f"\nEpoch {epoch:03d}/{EPOCHS}  lr={lr:.2e}")

        avg_loss, last_x, last_prior = train_one_epoch(
            model, loader, optimizer, epoch, EPOCHS
        )
        print(f"  → avg loss: {avg_loss:.4f}")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
            save_checkpoint(model, optimizer, epoch, avg_loss)

            # Vessel score visualization — run encoder first, then LDS
            with torch.no_grad():
                f0, f1, f2, f3 = model.encoder(last_x[:1])
                _, vs = model.lds(f2.detach(), last_prior[:1], epoch, EPOCHS)
            save_vessel_score(last_x[:1], vs, epoch)


if __name__ == "__main__":
    # To resume: main(resume_from="checkpoints/vasojepa_epoch050.pt")
    main()
