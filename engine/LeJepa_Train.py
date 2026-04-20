import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from torchmetrics.classification import MultilabelF1Score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import LeJepaDenseDataset
from utils.helpers import set_seed
from zoo.jepa_models import DenseLeJepaModel, LinearClsProbe

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GradScaler IS needed even for bfloat16: bf16 has only 7 mantissa bits;
# without scaling, small SIGReg gradients are rounded to zero.
scaler = torch.amp.GradScaler('cuda')


def loader(json_path, crops_json, num_classes=26, batch_size=128):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    ds = LeJepaDenseDataset(
        base_dataset_json=json_path,
        crops_json_path=crops_json,
        num_global=2,
        num_local=4,
        num_vessel_classes=num_classes,
        local_size=128,          # 128 → 256 is 2× zoom; much less distortion than 96 → 256
        root_dir=project_root
    )
    g = torch.Generator()
    g.manual_seed(42)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4, persistent_workers=True,
        worker_init_fn=seed_worker, generator=g
    )


def configCreate(path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)


class SIGReg(nn.Module):
    """
    Signature Regularisation — the sole collapse-prevention mechanism in LeJEPA.
    No epsilon stabilisation needed: the SIGReg statistic is inherently well-conditioned.
    (Balestriero & LeCun, 2025 — LeJEPA, Section 3)
    """
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
        # proj: (N, D) — pooled token vectors from any set of views/paths
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-1)
        return statistic.mean()


def train_epoch(model, probe, dataloader, optimiser, scheduler,
                sigreg, criterion_probe, f1_metric, epoch, config, writer):
    model.train()
    probe.train()
    running_loss, running_dense, running_sig = 0.0, 0.0, 0.0
    running_probe = 0.0
    running_probe_cnt = 1e-4   # avoid div-by-zero on non-syntax batches

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
    for batch_idx, batch in pbar:
        g_crops    = batch['global_crops'].cuda()   # (B, num_g, 1, 224, 224)
        g_boxes    = batch['global_boxes'].cuda()   # (B, num_g, 4)
        l_crops    = batch['local_crops'].cuda()    # (B, num_l, 1, 128, 128)
        l_boxes    = batch['local_boxes'].cuda()    # (B, num_l, 4)
        is_syntax  = batch['is_syntax'].cuda()
        cls_target = batch['cls_target'].cuda()

        _, num_l, _, _, _ = l_crops.shape
        is_syntax = is_syntax.bool()  # safety: ensure bool dtype for indexing

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):

            # ── 1. CONTEXT ENCODE ────────────────────────────────────────
            # GPT fix: randomly rotate which global crop is context each step
            ctx_idx   = random.randint(0, g_crops.shape[1] - 1)
            ctx_view  = g_crops[:, ctx_idx, ...]
            ctx_boxes = g_boxes[:, ctx_idx, :]
            ctx_proj, _, raw_feat = model(ctx_view)   # (B, L_c, D)

            # ── 2. PROBE (detached — no gradients back into backbone) ──────
            probe_pred = probe(raw_feat.detach())
            if is_syntax.any():
                loss_probe = criterion_probe(probe_pred[is_syntax], cls_target[is_syntax])
            else:
                loss_probe = torch.tensor(0.0, device='cuda', requires_grad=True)

            # GPT fix: DataParallel awareness — build a unified encode fn that respects DP
            # model() already goes through DP for context. For local crops we must do the same.
            # We replicate by looping through the DataParallel-aware model call.
            tgt_proj_list   = []
            tgt_pooled_list = []

            with torch.no_grad():
                for li in range(num_l):
                    # Call model (not model.module) so DataParallel distributes the work
                    t_proj, _, _ = model(l_crops[:, li, ...])
                    tgt_proj_list.append(t_proj)
                    tgt_pooled_list.append(t_proj.mean(dim=1))   # (B, D)

            # ── 4. PREDICTOR ─────────────────────────────────────────────
            L_t = tgt_proj_list[0].shape[1]
            pred_proj_list = []
            model_ref = model.module if hasattr(model, 'module') else model

            for li in range(num_l):
                p_dense, _ = model_ref.predictor(
                    ctx_proj, ctx_boxes, l_boxes[:, li, :], num_target_tokens=L_t
                )
                pred_proj_list.append(p_dense)

            # ── 5. DENSE LOSS (SimSiam Cosine Similarity) ───────────────────
            loss_dense = torch.tensor(0.0, device='cuda')
            for p_d, t_d in zip(pred_proj_list, tgt_proj_list):
                # SimSiam uses negative cosine similarity. We use 1 - cos_sim for a positive loss.
                sim = F.cosine_similarity(p_d, t_d.detach(), dim=-1)
                loss_dense = loss_dense + (1.0 - sim).mean()
            loss_dense = loss_dense / num_l

            loss_lejepa = loss_dense

        # ── BACKWARD ──────────────────────────────────────────────────────
        optimiser.zero_grad()
        scaler.scale(loss_lejepa).backward()
        if is_syntax.any() and loss_probe.requires_grad:
            scaler.scale(loss_probe).backward()

        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(probe.parameters()), max_norm=1.0
        )
        scaler.step(optimiser)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        # ── LOGGING ───────────────────────────────────────────────────────
        running_loss  += loss_lejepa.item()
        running_dense += loss_dense.item()
        # SIGReg removed logically
        if is_syntax.any():
            running_probe += loss_probe.item()
            running_probe_cnt += 1

        if is_syntax.any():
            prob = torch.sigmoid(probe_pred[is_syntax])
            f1_metric.update(prob, cls_target[is_syntax].int())

        pbar.set_postfix({
            'LeJEPA': f"{running_loss  / (batch_idx + 1):.3f}",
            'DenseCos':  f"{running_dense / (batch_idx + 1):.3f}",
            'ProbeL': f"{running_probe / running_probe_cnt:.3f}",
        })

        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("Train/LeJepa_Loss", loss_lejepa.item(), global_step)
        writer.add_scalar("Train/Dense_Cos",   loss_dense.item(),  global_step)
        
        if is_syntax.any():
            writer.add_scalar("Train/Probe_Loss", loss_probe.item(), global_step)

    f1_val = f1_metric.compute().item()
    f1_metric.reset()
    writer.add_scalar("Train/Probe_F1", f1_val, epoch)
    print(f"Epoch {epoch+1} Probe F1: {f1_val:.4f}")
    return running_loss / len(dataloader)


def trainScript():
    experiment_name = "dense_lejepa_swinv2"
    checkpoint_dir  = f"checkpoints/{experiment_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = {
        'experiment_name': experiment_name,
        'logging': {
            'log_dir':        f'runs/{experiment_name}',
            'checkpoint_dir': checkpoint_dir,
        },
        'training': {
            'batch_size':    32,
            'epochs':        400,
            'lr_probe':      1e-4,
            'lr_model':      1e-4,
            'weight_decay':  0.04,
            'labda':         0.05,   # Raised: gives SIGReg enough gradient weight vs bf16 rounding floor
            'warmup_epochs': 10,
        },
        'model': {
            'encoder_name': 'swinv2_tiny_window8_256',
            'proj_dim':     384,
        }
    }

    writer = SummaryWriter(log_dir=config['logging']['log_dir'])
    configCreate(os.path.join(config['logging']['log_dir'], 'config.yaml'), config)

    dataset_json = "data/ARCADE/processed/dataset.json"
    crops_json   = "data/ARCADE/processed/dataset_crops.json"
    num_classes  = 26

    train_loader = loader(
        dataset_json, crops_json,
        num_classes=num_classes,
        batch_size=config['training']['batch_size']
    )

    model = DenseLeJepaModel(
        encoder_name=config['model']['encoder_name'],
        proj_dim=config['model']['proj_dim'],
    ).cuda()
    probe   = LinearClsProbe(encoder_name=config['model']['encoder_name'], num_classes=num_classes).cuda()
    sigreg  = SIGReg().cuda()

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
        probe = nn.DataParallel(probe)

    lr1 = {"params": probe.parameters(), "lr": config['training']['lr_probe'],
            "weight_decay": config['training']['weight_decay']}
    lr2 = {"params": model.parameters(), "lr": config['training']['lr_model'],
            "weight_decay": config['training']['weight_decay']}
    opt = torch.optim.AdamW([lr1, lr2])

    total_iters_per_epoch = len(train_loader)
    if total_iters_per_epoch == 0:
        print("DataLoader is empty — run offline_preprocess.py first.")
        return

    warmup_iters = config['training']['warmup_epochs'] * total_iters_per_epoch
    total_iters  = config['training']['epochs']        * total_iters_per_epoch

    scheduler1 = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
    scheduler2 = CosineAnnealingLR(opt, T_max=total_iters - warmup_iters, eta_min=1e-6)
    scheduler  = SequentialLR(opt, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])

    criterion_probe = nn.BCEWithLogitsLoss()
    f1_metric       = MultilabelF1Score(num_labels=num_classes).cuda()

    for epoch in range(config['training']['epochs']):
        train_epoch(
            model, probe, train_loader, opt, scheduler,
            sigreg, criterion_probe, f1_metric, epoch, config, writer
        )

        ckpt = {
            'epoch':     epoch,
            'model':     model.module.state_dict() if num_gpus > 1 else model.state_dict(),
            'probe':     probe.module.state_dict() if num_gpus > 1 else probe.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, "last_model.pth"))

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(checkpoint_dir, f"model_ep{epoch+1}.pth"))
            backbone = model.module.backbone if num_gpus > 1 else model.backbone
            torch.save(backbone.state_dict(),
                       os.path.join(checkpoint_dir, f"backbone_ep{epoch+1}.pth"))


if __name__ == "__main__":
    trainScript()