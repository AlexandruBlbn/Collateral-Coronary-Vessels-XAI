import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import yaml
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from data.dataloader import ArcadeDataset
from zoo.backbones import get_backbone
from zoo.dino_components.dino_head import DINOHead
from zoo.dino_components.koleo_loss import KoLeoLoss
from utils.logger import TensorboardLogger


class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs, teacher_outputs, teacher_temp):
        teacher_probs = []
        for t in teacher_outputs:
            centered = (t.float() - self.center) / teacher_temp
            teacher_probs.append(F.softmax(centered, dim=-1))

        total_loss = 0
        n_terms = 0
        for t_idx, tp in enumerate(teacher_probs):
            for s_idx, so in enumerate(student_outputs):
                if s_idx == t_idx:
                    continue
                s_log = F.log_softmax(so.float() / self.student_temp, dim=-1)
                total_loss += torch.sum(-tp.detach() * s_log, dim=-1).mean()
                n_terms += 1

        total_loss /= n_terms
        self._update_center(teacher_outputs)
        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_outputs):
        batch_center = torch.cat(teacher_outputs).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MultiCropAugmentation:
    def __init__(self, input_size=256, n_global=2, n_local=4,
                 global_scale=(0.4, 1.0), local_scale=(0.05, 0.4)):
        base_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0)], p=0.8
            ),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=global_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            *base_aug
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=local_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            *base_aug
        ])
        self.n_global = n_global
        self.n_local = n_local

    def __call__(self, image):
        crops = []
        for _ in range(self.n_global):
            crops.append(self.global_transform(image))
        for _ in range(self.n_local):
            crops.append(self.local_transform(image))
        return crops


def cosine_schedule(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):
    schedule = []
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            val = start_warmup_value + (base_value - start_warmup_value) * epoch / max(warmup_epochs, 1)
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            val = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
        schedule.append(val)
    return schedule


def pool_features(feat):
    if feat.dim() == 4:
        return feat.mean(dim=[-2, -1])
    if feat.dim() == 3:
        return feat.mean(dim=1)
    return feat


def enable_grad_checkpointing(backbone):
    inner = getattr(backbone, 'model', backbone)
    if hasattr(inner, 'set_grad_checkpointing'):
        inner.set_grad_checkpointing(True)
        print("Gradient checkpointing enabled")


def train_one_epoch(student_backbone, student_head, teacher_backbone, teacher_head,
                    dino_loss, koleo_loss, dataloader, optimizer, device, epoch,
                    logger, scaler, config, lr_schedule, wd_schedule,
                    momentum_schedule, teacher_temp_schedule):
    student_backbone.train()
    student_head.train()
    running_loss = 0.0
    n_global = config["data"]["n_global_crops"]
    koleo_w = config["dino"]["koleo_weight"]
    clip_val = config["dino"]["clip_grad"]

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for step, crops in enumerate(progress_bar):
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = lr_schedule[epoch]
            if i == 0:
                pg["weight_decay"] = wd_schedule[epoch]

        crops = [c.to(device, non_blocking=True) for c in crops]
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            student_outputs = []
            for crop in crops:
                feat = pool_features(student_backbone(crop))
                student_outputs.append(student_head(feat))

            with torch.no_grad():
                teacher_outputs = []
                for crop in crops[:n_global]:
                    feat = pool_features(teacher_backbone(crop))
                    teacher_outputs.append(teacher_head(feat))

            t_temp = teacher_temp_schedule[min(epoch, len(teacher_temp_schedule) - 1)]
            loss_dino = dino_loss(student_outputs, teacher_outputs, t_temp)
            loss_koleo = koleo_loss(student_outputs[0])
            loss = loss_dino + koleo_w * loss_koleo

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student_backbone.parameters(), clip_val)
        nn.utils.clip_grad_norm_(student_head.parameters(), clip_val)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            m = momentum_schedule[epoch]
            for ps, pt in zip(student_backbone.parameters(), teacher_backbone.parameters()):
                pt.data.mul_(m).add_((1 - m) * ps.data)
            for ps, pt in zip(student_head.parameters(), teacher_head.parameters()):
                pt.data.mul_(m).add_((1 - m) * ps.data)

        running_loss += loss_dino.item()
        progress_bar.set_postfix({"dino_loss": f"{loss_dino.item():.4f}"})
        global_step = epoch * len(dataloader) + step
        logger.log_scalar("Train/Step_DINOLoss", loss_dino.item(), global_step)

    return running_loss / len(dataloader)


def main():
    config_path = "config/dino_config.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["system"]["seed"])
    print(f"Using device: {device}")

    backbone_name = config["model"]["backbone"]
    experiment_name = config.get("experiment_name", f"dino_{backbone_name}")
    logger = TensorboardLogger(log_dir="runs", experiment_name=experiment_name)

    transform = MultiCropAugmentation(
        input_size=config["model"]["input_size"],
        n_global=config["data"]["n_global_crops"],
        n_local=config["data"]["n_local_crops"],
        global_scale=(config["data"]["global_scale_min"], config["data"]["global_scale_max"]),
        local_scale=(config["data"]["local_scale_min"], config["data"]["local_scale_max"]),
    )

    json_path = "/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json"
    root_dir = "/workspace/Collateral-Coronary-Vessels-XAI"
    dataset = ArcadeDataset(json_path=json_path, split='train', transform=transform,
                            mode='pretrain', root_dir=root_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    feature_dim = config["model"]["feature_dim"]
    head_dim = config["model"]["head_dim"]

    student_backbone = get_backbone(backbone_name, in_channels=config["data"]["in_channels"]).to(device)
    student_head = DINOHead(
        in_dim=feature_dim, out_dim=head_dim,
        hidden_dim=config["model"]["head_hidden_dim"],
        bottleneck_dim=config["model"]["head_bottleneck_dim"],
    ).to(device)
    student_head.init_weights()

    teacher_backbone = copy.deepcopy(student_backbone)
    teacher_head = copy.deepcopy(student_head)
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False

    if config["system"].get("gradient_checkpointing", False):
        enable_grad_checkpointing(student_backbone)

    dino_loss_fn = DINOLoss(
        out_dim=head_dim,
        student_temp=config["dino"]["student_temp"],
        center_momentum=config["dino"]["center_momentum"],
    ).to(device)

    koleo_loss_fn = KoLeoLoss().to(device)

    all_params = list(student_backbone.named_parameters()) + list(student_head.named_parameters())
    params = [
        {"params": [p for n, p in all_params if p.requires_grad and not (n.endswith(".bias") or len(p.shape) == 1)]},
        {"params": [p for n, p in all_params if p.requires_grad and (n.endswith(".bias") or len(p.shape) == 1)],
         "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(params, lr=float(config["optimizer"]["lr"]),
                            weight_decay=config["optimizer"]["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    epochs = config["optimizer"]["epochs"]

    lr_schedule = cosine_schedule(
        float(config["optimizer"]["lr"]), float(config["optimizer"]["final_lr"]),
        epochs, config["optimizer"]["warmup_epochs"]
    )
    wd_schedule = cosine_schedule(
        config["optimizer"]["weight_decay"], config["optimizer"]["weight_decay_end"], epochs
    )
    momentum_schedule = cosine_schedule(
        config["dino"]["ema_momentum"], 1.0, epochs
    )
    teacher_temp_schedule = cosine_schedule(
        config["dino"]["teacher_warmup_temp"], config["dino"]["teacher_temp"],
        config["dino"]["teacher_temp_warmup_epochs"]
    )
    teacher_temp_schedule += [config["dino"]["teacher_temp"]] * (epochs - len(teacher_temp_schedule))

    save_dir = os.path.join(config["system"]["save_dir"], experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            student_backbone, student_head, teacher_backbone, teacher_head,
            dino_loss_fn, koleo_loss_fn, dataloader, optimizer, device, epoch,
            logger, scaler, config, lr_schedule, wd_schedule,
            momentum_schedule, teacher_temp_schedule
        )
        logger.log_scalar("Train/Epoch_DINOLoss", train_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs} | DINO Loss: {train_loss:.4f}")

        torch.save(student_backbone.state_dict(), os.path.join(save_dir, "last_backbone.pth"))

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(student_backbone.state_dict(), os.path.join(save_dir, "best_backbone.pth"))
            print(f"New best backbone saved with loss: {best_loss:.4f}")

    logger.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
