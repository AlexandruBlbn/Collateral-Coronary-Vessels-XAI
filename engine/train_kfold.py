import os
import sys
import csv
import yaml
import random
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from tqdm import tqdm
import torchvision

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import TverskyLoss
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataloader import ArcadeDataset
from data.transformWrapper import TransformsWrapper as TW
from utils.helpers import set_seed


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_create(path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)


def make_loader(dataset, batch_size, shuffle, seed=42, num_workers=4):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )


def build_syntax_trainval_base(json_path="data/ARCADE/processed/dataset.json", root_dir="."):
    train_ds = ArcadeDataset(split="train", transform=None, mode="syntax", root_dir=root_dir, json_path=json_path)
    val_ds = ArcadeDataset(split="validation", transform=None, mode="syntax", root_dir=root_dir, json_path=json_path)
    return ConcatDataset([train_ds, val_ds])


def build_syntax_test_loader(img_size, batch_size, json_path="data/ARCADE/processed/dataset.json", root_dir="."):
    test_base = ArcadeDataset(split="test", transform=None, mode="syntax", root_dir=root_dir, json_path=json_path)
    test_ds = TW(test_base, input_size=img_size, mode="test")
    return make_loader(test_ds, batch_size=batch_size, shuffle=False)


def make_kfold_indices(num_samples, n_splits, seed=42):
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits > num_samples:
        raise ValueError("n_splits cannot be greater than number of samples")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    fold_sizes = np.full(n_splits, num_samples // n_splits, dtype=int)
    fold_sizes[: num_samples % n_splits] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop

    return folds


def train_epoch(model, dataloader, criterion, optimizer, epoch, writer, fold_idx):
    model.train()
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Fold {fold_idx + 1} | Epoch {epoch + 1}")
    for batch_idx, (images, masks) in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        output = model(images)
        loss = criterion(output, masks)
        if not torch.isfinite(loss):
            print(f"[Warn] Fold {fold_idx + 1} non-finite loss at batch {batch_idx}, skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"train_loss": running_loss / (batch_idx + 1)})

    avg_train_loss = running_loss / max(1, len(dataloader))
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    return avg_train_loss


def validate_epoch(model, dataloader, criterion, f1_metric, epoch, writer, fold_idx):
    model.eval()
    val_f1 = 0.0
    val_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Fold {fold_idx + 1} | Epoch {epoch + 1} - Validation",
        )
        for batch_idx, (images, masks) in pbar:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            loss = criterion(output, masks)

            val_loss += loss.item()
            val_f1 += f1_metric(output.sigmoid(), masks.int()).item()
            pbar.set_postfix(
                {
                    "val_loss": val_loss / (batch_idx + 1),
                    "val_f1": val_f1 / (batch_idx + 1),
                }
            )

            if batch_idx == 0:
                img_vis = images * 0.5 + 0.5
                num_samples = min(4, img_vis.size(0))
                grid_images = []
                preds_vis = torch.sigmoid(output)

                for i in range(num_samples):
                    grid_images.append(img_vis[i].cpu())
                    grid_images.append(preds_vis[i].float().cpu())
                    grid_images.append(masks[i].float().cpu())

                grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
                writer.add_image("Val/Predictions", grid, epoch)

    avg_f1 = val_f1 / max(1, len(dataloader))
    avg_val_loss = val_loss / max(1, len(dataloader))
    writer.add_scalar("Val/F1", avg_f1, epoch)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    print(f"Fold {fold_idx + 1} Validation F1: {avg_f1:.4f}")
    return avg_f1, avg_val_loss


def find_best_f1_threshold(model, dataloader, f1_metric):
    model.eval()
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_thresh = 0.5
    best_f1 = 0.0

    with torch.no_grad():
        all_probs = []
        all_masks = []
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device).int()
            probs = torch.sigmoid(model(images))
            all_probs.append(probs)
            all_masks.append(masks)

        all_probs = torch.cat(all_probs, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        for t in thresholds:
            preds_bin = (all_probs > t).int()
            f1_metric.reset()
            current_f1 = f1_metric(preds_bin, all_masks).item()
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_thresh = t

    return float(best_thresh), float(best_f1)


def evaluate_all_thresholds(model, dataloader, f1_metric, iou_metric, csv_path):
    model.eval()
    thresholds = np.arange(0.1, 0.95, 0.05)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    all_probs = []
    all_masks = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            probs = model(images).sigmoid()
            all_probs.append(probs)
            all_masks.append(masks.int())

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Threshold", "F1_Score", "IoU_Score"])
        for t in thresholds:
            preds_bin = (all_probs > t).int()
            f1_metric.reset()
            iou_metric.reset()
            current_f1 = f1_metric(preds_bin, all_masks).item()
            current_iou = iou_metric(preds_bin, all_masks).item()
            writer.writerow([f"{t:.2f}", f"{current_f1:.4f}", f"{current_iou:.4f}"])


def test_model(model, dataloader, f1_metric, iou_metric, writer, prefix="Test"):
    model.eval()
    test_f1 = 0.0
    test_iou = 0.0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=prefix)
        for batch_idx, (images, masks) in pbar:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            preds = (output.sigmoid() > 0.5).int()

            test_f1 += f1_metric(preds, masks.int()).item()
            test_iou += iou_metric(preds, masks.int()).item()
            pbar.set_postfix(
                {
                    "test_f1": test_f1 / (batch_idx + 1),
                    "test_iou": test_iou / (batch_idx + 1),
                }
            )

    test_f1 = test_f1 / max(1, len(dataloader))
    test_iou = test_iou / max(1, len(dataloader))
    writer.add_scalar(f"{prefix}/F1", test_f1)
    writer.add_scalar(f"{prefix}/IoU", test_iou)
    return float(test_f1), float(test_iou)


def test_model_with_threshold(model, dataloader, f1_metric, iou_metric, writer, threshold):
    model.eval()
    test_f1 = 0.0
    test_iou = 0.0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing (best-threshold)")
        for batch_idx, (images, masks) in pbar:
            images, masks = images.to(device), masks.to(device)
            probs = model(images).sigmoid()
            preds = (probs > threshold).int()

            test_f1 += f1_metric(preds, masks.int()).item()
            test_iou += iou_metric(preds, masks.int()).item()
            pbar.set_postfix(
                {
                    "test_f1": test_f1 / (batch_idx + 1),
                    "test_iou": test_iou / (batch_idx + 1),
                }
            )

    test_f1 = test_f1 / max(1, len(dataloader))
    test_iou = test_iou / max(1, len(dataloader))
    writer.add_scalar("Test_Youden/F1", test_f1)
    writer.add_scalar("Test_Youden/IoU", test_iou)
    return float(test_f1), float(test_iou)


def create_model(config):
    model = smp.DPT(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"].get("encoder_weights", None),
        in_channels=config["model"].get("in_channels", 1),
        classes=config["model"].get("classes", 1),
        decoder_readout=config["model"].get("decoder_readout", "ignore"),
    ).to(device)
    return model


def run_fold(
    fold_idx,
    train_loader,
    val_loader,
    test_loader,
    config,
    fold_log_dir,
    fold_ckpt_dir,
    aggregate_writer,
):
    os.makedirs(fold_log_dir, exist_ok=True)
    os.makedirs(fold_ckpt_dir, exist_ok=True)

    fold_writer = SummaryWriter(log_dir=fold_log_dir)
    model = create_model(config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )

    tversky_loss_fn = TverskyLoss(
        mode="binary",
        alpha=float(config["training"].get("tversky_alpha", 0.3)),
        beta=float(config["training"].get("tversky_beta", 0.7)),
        gamma=1.0,
        log_loss=False,
    )
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    tversky_weight = float(config["training"].get("tversky_weight", 0.5))
    bce_weight = float(config["training"].get("bce_weight", 0.5))

    def criterion(pred, target):
        probs = torch.sigmoid(pred.float()).clamp(min=1e-6, max=1.0 - 1e-6)
        target = target.float().clamp(min=0.0, max=1.0)
        tversky_loss = tversky_loss_fn(probs, target)
        bce_loss = bce_loss_fn(pred.float(), target)

        loss = tversky_weight * tversky_loss + bce_weight * bce_loss
        if not torch.isfinite(loss):
            # Safe fallback on rare unstable batches.
            return bce_loss
        return loss

    warmup_epochs = config["training"].get("warmup_epochs", 5)
    total_epochs = config["training"]["epochs"]
    cosine_t_max = max(1, total_epochs - warmup_epochs)

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_t_max)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    f1_metric = BinaryF1Score().to(device)
    iou_metric = BinaryJaccardIndex().to(device)

    best_model_path = os.path.join(fold_ckpt_dir, "best_model.pth")
    last_model_path = os.path.join(fold_ckpt_dir, "last_model.pth")
    best_val_f1 = 0.0

    for epoch in range(total_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, fold_writer, fold_idx)
        val_f1, val_loss = validate_epoch(model, val_loader, criterion, f1_metric, epoch, fold_writer, fold_idx)
        scheduler.step()

        aggregate_writer.add_scalar(f"fold_{fold_idx + 1}/Loss_train", train_loss, epoch)
        aggregate_writer.add_scalar(f"fold_{fold_idx + 1}/Loss_val", val_loss, epoch)
        aggregate_writer.add_scalar(f"fold_{fold_idx + 1}/Val_F1", val_f1, epoch)

        torch.save(model.state_dict(), last_model_path)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Fold {fold_idx + 1}: saved BEST model (F1={best_val_f1:.4f})")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    default_f1, default_iou = test_model(model, test_loader, f1_metric, iou_metric, fold_writer, prefix="Test")

    best_threshold, best_fold_val_f1 = find_best_f1_threshold(model, val_loader, f1_metric)
    thresh_f1, thresh_iou = test_model_with_threshold(
        model,
        test_loader,
        f1_metric,
        iou_metric,
        fold_writer,
        threshold=best_threshold,
    )

    threshold_csv = os.path.join(fold_log_dir, "thresholds_results.csv")
    evaluate_all_thresholds(model, test_loader, f1_metric, iou_metric, threshold_csv)

    aggregate_writer.add_scalar("kfold/Test_F1_default", default_f1, fold_idx + 1)
    aggregate_writer.add_scalar("kfold/Test_IoU_default", default_iou, fold_idx + 1)
    aggregate_writer.add_scalar("kfold/Test_F1_best_threshold", thresh_f1, fold_idx + 1)
    aggregate_writer.add_scalar("kfold/Test_IoU_best_threshold", thresh_iou, fold_idx + 1)
    aggregate_writer.add_scalar("kfold/Val_Best_F1", best_val_f1, fold_idx + 1)
    aggregate_writer.add_scalar("kfold/Val_Best_Threshold", best_threshold, fold_idx + 1)

    fold_writer.close()

    return {
        "fold": fold_idx + 1,
        "best_val_f1": float(best_val_f1),
        "best_val_f1_at_threshold_search": float(best_fold_val_f1),
        "best_threshold": float(best_threshold),
        "test_f1_default": float(default_f1),
        "test_iou_default": float(default_iou),
        "test_f1_best_threshold": float(thresh_f1),
        "test_iou_best_threshold": float(thresh_iou),
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
    }


def main():
    config = {
        "experiment_name": "DPT_convnextv2_pico_no_pretrain_MCC_kfold",
        "data": {
            "json_path": "data/ARCADE/processed/dataset.json",
            "root_dir": ".",
            "mode": "syntax",
            "n_splits": 5,
            "seed": 42,
        },
        "logging": {
            "log_dir": "runs/{experiment_name}",
            "checkpoint_dir": "checkpoints/{experiment_name}",
            "fold_prefix": "fold",
        },
        "training": {
            "img_size": 256,
            "batch_size": 16,
            "epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 5,
            "loss_function": "BCE + Tversky",
            "tversky_alpha": 0.3,
            "tversky_beta": 0.7,
            "tversky_weight": 0.5,
            "bce_weight": 0.5,
            "scheduler": "LinearWarmup + CosineAnnealingLR",
            "precision": "bfloat16",
        },
        "model": {
            "name": "DPT",
            "encoder_name": "tu-convnextv2_pico",
            "encoder_weights": None,
            "in_channels": 1,
            "classes": 1,
            "decoder_readout": "ignore",
        },
    }

    log_root = config["logging"]["log_dir"].format(experiment_name=config["experiment_name"])
    ckpt_root = config["logging"]["checkpoint_dir"].format(experiment_name=config["experiment_name"])
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    config_create(os.path.join(log_root, "config.yaml"), config)
    aggregate_writer = SummaryWriter(log_dir=os.path.join(log_root, "main"))

    base_trainval = build_syntax_trainval_base(
        json_path=config["data"]["json_path"],
        root_dir=config["data"]["root_dir"],
    )
    test_loader = build_syntax_test_loader(
        img_size=config["training"]["img_size"],
        batch_size=config["training"]["batch_size"],
        json_path=config["data"]["json_path"],
        root_dir=config["data"]["root_dir"],
    )

    n_total = len(base_trainval)
    n_splits = config["data"]["n_splits"]
    folds = make_kfold_indices(n_total, n_splits=n_splits, seed=config["data"]["seed"])
    print(f"Total syntax train+validation samples used for k-fold: {n_total}")

    all_fold_results = []
    for fold_idx in range(n_splits):
        val_indices = folds[fold_idx]
        train_indices = np.concatenate([folds[i] for i in range(n_splits) if i != fold_idx])

        train_subset = Subset(base_trainval, train_indices.tolist())
        val_subset = Subset(base_trainval, val_indices.tolist())

        train_ds = TW(train_subset, input_size=config["training"]["img_size"], mode="train")
        val_ds = TW(val_subset, input_size=config["training"]["img_size"], mode="validation")

        train_loader = make_loader(
            train_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            seed=config["data"]["seed"] + fold_idx,
        )
        val_loader = make_loader(
            val_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            seed=config["data"]["seed"] + fold_idx,
        )

        fold_name = f"{config['logging']['fold_prefix']}{fold_idx + 1}"
        fold_log_dir = os.path.join(log_root, fold_name)
        fold_ckpt_dir = os.path.join(ckpt_root, fold_name)

        print("\n" + "=" * 60)
        print(f"Starting {fold_name} | train={len(train_subset)} | val={len(val_subset)}")
        print("=" * 60)

        fold_result = run_fold(
            fold_idx=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            fold_log_dir=fold_log_dir,
            fold_ckpt_dir=fold_ckpt_dir,
            aggregate_writer=aggregate_writer,
        )
        all_fold_results.append(fold_result)

    mean_default_f1 = float(np.mean([r["test_f1_default"] for r in all_fold_results]))
    mean_default_iou = float(np.mean([r["test_iou_default"] for r in all_fold_results]))
    mean_thresh_f1 = float(np.mean([r["test_f1_best_threshold"] for r in all_fold_results]))
    mean_thresh_iou = float(np.mean([r["test_iou_best_threshold"] for r in all_fold_results]))
    mean_best_val = float(np.mean([r["best_val_f1"] for r in all_fold_results]))

    aggregate_writer.add_scalar("kfold_summary/mean_test_f1_default", mean_default_f1, n_splits)
    aggregate_writer.add_scalar("kfold_summary/mean_test_iou_default", mean_default_iou, n_splits)
    aggregate_writer.add_scalar("kfold_summary/mean_test_f1_best_threshold", mean_thresh_f1, n_splits)
    aggregate_writer.add_scalar("kfold_summary/mean_test_iou_best_threshold", mean_thresh_iou, n_splits)
    aggregate_writer.add_scalar("kfold_summary/mean_best_val_f1", mean_best_val, n_splits)

    summary_csv = os.path.join(log_root, "kfold_summary.csv")
    with open(summary_csv, mode="w", newline="") as f:
        fieldnames = [
            "fold",
            "best_val_f1",
            "best_val_f1_at_threshold_search",
            "best_threshold",
            "test_f1_default",
            "test_iou_default",
            "test_f1_best_threshold",
            "test_iou_best_threshold",
            "best_model_path",
            "last_model_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_fold_results:
            writer.writerow(row)

        writer.writerow(
            {
                "fold": "mean",
                "best_val_f1": mean_best_val,
                "test_f1_default": mean_default_f1,
                "test_iou_default": mean_default_iou,
                "test_f1_best_threshold": mean_thresh_f1,
                "test_iou_best_threshold": mean_thresh_iou,
            }
        )

    aggregate_writer.close()

    print("\nK-fold training completed.")
    print(f"Summary CSV: {summary_csv}")
    print(f"TensorBoard root: {log_root}")
    print(f"Checkpoint root: {ckpt_root}")


if __name__ == "__main__":
    main()