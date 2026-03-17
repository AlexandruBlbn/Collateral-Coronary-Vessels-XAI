import os
import sys
import yaml
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from PIL import Image 
import timm as timm
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataloader import ArcadeDataset
import segmentation_models_pytorch as smp
import cv2
from torchinfo import summary
import monai
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.transformWrapper import TransformsWrapper as TW
from torch.utils.tensorboard import SummaryWriter
from utils.helpers import set_seed
import numpy as np
import random

from segmentation_models_pytorch.metrics import sensitivity, specificity, iou_score
from segmentation_models_pytorch.losses import TverskyLoss, SoftBCEWithLogitsLoss
from monai.networks.nets.unet import UNet
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.losses import DiceCELoss

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loader(img_size, batch_size, split='train'):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    base = ArcadeDataset(split=split, transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
    ds = TW(base, input_size=img_size, mode=split)
    g = torch.Generator()
    g.manual_seed(42)
    
    return DataLoader(
        ds,
        batch_size=batch_size, 
        shuffle=(split=='train'),
        num_workers=4, 
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )
        
def configCreate(path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)

config = {}

def modelChange(model, old_layer, new_layer):
    for k, v in model.named_children():
        if isinstance(v, old_layer):
            setattr(model, k, new_layer)
    return model.cuda()

def load_pretrained_backbone(model, backbone_path):
    """Load LeJEPA best_backbone.pth into the SMP model encoder.
    Handles both tu- encoders (timm keys need 'model.' prefix) and
    standard SMP encoders (direct key match for layer1-layer4).
    """
    if backbone_path is None or not os.path.isfile(backbone_path):
        print(f"[Backbone] No pretrained weights loaded (path: {backbone_path})")
        return model
    lejepa_sd = torch.load(backbone_path, map_location='cpu')
    encoder_sd = model.encoder.state_dict()
    compatible = {}
    for k, v in lejepa_sd.items():
        if k in encoder_sd and encoder_sd[k].shape == v.shape:
            compatible[k] = v
        elif f'model.{k}' in encoder_sd and encoder_sd[f'model.{k}'].shape == v.shape:
            compatible[f'model.{k}'] = v
    encoder_sd.update(compatible)
    model.encoder.load_state_dict(encoder_sd)
    print(f"[Backbone] Loaded {len(compatible)}/{len(encoder_sd)} layers from {backbone_path}")
    return model

def train_epoch(model, dataloader, criterion, optimiser, f1_metric, epoch):
    model.train()
    running_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
    for batch_idx, (images, masks) in pbar:
        images, masks = images.cuda(), masks.cuda()
        optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model(images)
            loss = criterion(output, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        running_loss += loss.item()
        pbar.set_postfix({'Loss': running_loss / (batch_idx + 1)})
    writer.add_scalar('Loss/train', running_loss / len(dataloader), epoch)

def validate_epoch(model, dataloader, criterion, f1_metric, epoch):
    model.eval()
    val_f1 = 0.0
    val_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1} - Validation")
        for batch_idx, (images, masks) in pbar:
            images, masks = images.cuda(), masks.cuda()
            output = model(images)
            loss = criterion(output, masks) 
            val_loss += loss.item()
            val_f1 += f1_metric(output.sigmoid(), masks.int()).item()
            pbar.set_postfix({'val_loss': val_loss / (batch_idx + 1), 'val_f1': val_f1 / (batch_idx + 1)})
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

        avg_f1 = val_f1 / len(dataloader)
        writer.add_scalar("Val/F1", avg_f1, epoch)
        writer.add_scalar("Loss/val", val_loss / len(dataloader), epoch)
        print(f"Validation F1: {avg_f1:.4f}")
        return avg_f1
    
def find_best_f1_threshold(model, loader, f1_metric):
    model.eval()
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_thresh = 0.5
    best_f1 = 0.0
    
    with torch.no_grad():
        all_probs = []
        all_masks = []
        for img, masks in loader:
            img, masks = img.cuda(), masks.cuda().int()
            probs = torch.sigmoid(model(img))
            all_probs.append(probs)
            all_masks.append(masks)
            
        all_probs = torch.cat(all_probs, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        for t in thresholds:
            preds_b = (all_probs > t).int()
            f1_metric.reset()
            current_f1 = f1_metric(preds_b, all_masks).item()
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_thresh = t
                
    return best_thresh

def evaluate_all_thresholds(model, dataloader, f1_metric, iou_metric, config):
    model.eval()
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    csv_path = os.path.join(
        config['logging']['log_dir'].format(experiment_name=config['experiment_name']), 
        'thresholds_results.csv'
    )
    
    all_probs = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.cuda(), masks.cuda()
            probs = model(images).sigmoid()
            all_probs.append(probs)
            all_masks.append(masks.int())
            
        all_probs = torch.cat(all_probs, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Threshold', 'F1_Score', 'IoU_Score'])
        
        for t in thresholds:
            preds_bin = (all_probs > t).int()
            f1_metric.reset()
            iou_metric.reset()
            current_f1 = f1_metric(preds_bin, all_masks).item()
            current_iou = iou_metric(preds_bin, all_masks).item()
            
            # print(f"Prag: {t:.2f} | F1: {current_f1:.4f} | IoU: {current_iou:.4f}")
            writer.writerow([f"{t:.2f}", f"{current_f1:.4f}", f"{current_iou:.4f}"])

def test_model(model, dataloader, f1_metric, iou_metric, tb_writer):
    model.eval()
    test_f1 = 0.0
    test_iou = 0.0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")
        for batch_idx, (images, masks) in pbar:
            images, masks = images.cuda(), masks.cuda()
            output = model(images)
            preds = (output.sigmoid() > 0.5).int()
            test_f1 += f1_metric(preds, masks.int()).item()
            test_iou += iou_metric(preds, masks.int()).item()
            pbar.set_postfix({'test_f1': test_f1 / (batch_idx + 1), 'test_iou': test_iou / (batch_idx + 1)})
    test_f1 = test_f1 / len(dataloader)
    test_iou = test_iou / len(dataloader)
    print(f"Test F1: {test_f1:.4f}, Test IoU: {test_iou:.4f}")
    tb_writer.add_scalar("Test/F1", test_f1)
    tb_writer.add_scalar("Test/IoU", test_iou)
    return test_f1

def test_model_youden(model, dataloader, f1_metric, iou_score, tb_writer, optimal_threshold):
    model.eval()
    test_f1 = 0.0
    test_iou = 0.0  
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing (Youden)")
        for batch_idx, (images, masks) in pbar:
            images, masks = images.cuda(), masks.cuda()
            probs = model(images).sigmoid()
            preds_bin = (probs > optimal_threshold).int()
            test_f1 += f1_metric(preds_bin, masks.int()).item()
            test_iou += iou_score(preds_bin, masks.int()).item()
            pbar.set_postfix({
                'test_f1': test_f1 / (batch_idx + 1), 
                'test_iou': test_iou / (batch_idx + 1)
            })       
    test_f1 = test_f1 / len(dataloader)
    test_iou = test_iou / len(dataloader)
    print(f"Test F1 (Youden): {test_f1:.4f}, Test IoU (Youden): {test_iou:.4f}")
    tb_writer.add_scalar("Test_Youden/F1", test_f1)
    tb_writer.add_scalar("Test_Youden/IoU", test_iou)
    
    return test_f1, test_iou
    
def trainScript(model,
                train_loader,
                val_loader,
                test_loader,
                criterion,
                optimiser,
                scheduler,
                f1_metric,
                iou_metric,
                num_epochs,
                config,         
                tb_writer     
                ):

    checkpoint_dir = config['logging']['checkpoint_dir'].format(experiment_name=config['experiment_name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    best_val_f1 = 0.0
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, criterion, optimiser, f1_metric, epoch)
        val_f1 = validate_epoch(model, val_loader, criterion, f1_metric, epoch)
        scheduler.step()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"model salvat cu F1 = {best_val_f1:.4f}")

    print("\n" + "="*50)
    print("testare")
    print("="*50 + "\n")

    model.load_state_dict(torch.load(best_model_path))
    print("Test cu sigmoid")
    test_model(model, test_loader, f1_metric, iou_metric=iou_metric, tb_writer=tb_writer)
    
    best_thresh = find_best_f1_threshold(model, val_loader, f1_metric)
    print(f"Best threshold: {best_thresh:.4f}")
    
    test_model_youden(
        model=model, 
        dataloader=test_loader, 
        f1_metric=f1_metric, 
        iou_score=iou_metric, 
        tb_writer=tb_writer, 
        optimal_threshold=best_thresh
    )
    
    evaluate_all_thresholds(
        model=model, 
        dataloader=test_loader, 
        f1_metric=f1_metric, 
        iou_metric=iou_metric, 
        config=config
    )
    
if __name__ == "__main__":
    
    config = {
    'experiment_name': 'resnet50_unetplusplus',
    'logging': {
        'log_dir': 'runs/{experiment_name}',
        'checkpoint_dir': 'checkpoints/{experiment_name}'
    },
    'training': {
        'img_size': 256,
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 2e-4,
        'loss_function': "twersky + BCE",
        'scheduler': 'CosineAnnealingLR',
        'precision': 'bfloat16',
    },
    'model': {
        'model': 'summary',
        'pretrained_backbone': None,
    }
    }
    writer = SummaryWriter(
        log_dir=config['logging']['log_dir'].format(experiment_name=config['experiment_name'])
    )
    
    train_loader = loader(config['training']['img_size'], config['training']['batch_size'], split='train')
    val_loader = loader(config['training']['img_size'], config['training']['batch_size'], split='validation')
    test_loader = loader(config['training']['img_size'], config['training']['batch_size'], split='test')

    # model = UNet(
    # spatial_dims=2,
    # in_channels=1,
    # out_channels=1,
    # channels=(32, 64, 128, 256, 512, 1024),
    # strides=(2, 2, 2, 2, 2),                
    # num_res_units=0,
    # norm='instance',
    # act='leakyrelu'
    # ).cuda()
    
    
    model = smp.UnetPlusPlus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=1,
    classes=1,
    encoder_depth=5,
    decoder_channels=(512, 256, 128, 64,32),
    decoder_use_batchnorm=True,
    decoder_attention_type='scse'
).cuda()

    if config['model'].get('pretrained_backbone'):
        model = load_pretrained_backbone(model, config['model']['pretrained_backbone'])

    optimiser = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)

    tversky_loss_fn = TverskyLoss(
        mode='binary',
        beta=0.7,
        gamma=0.75,
        log_loss=False,
    )
    bce_loss_fn = SoftBCEWithLogitsLoss()

    def criterion(pred, target):
        return tversky_loss_fn(pred, target) + bce_loss_fn(pred, target)

    warmup = LinearLR(optimiser, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine = CosineAnnealingLR(optimiser, T_max=config['training']['epochs'] - 5)
    scheduler = SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[5])
    f1_metric = BinaryF1Score().cuda()
    iou_metric = BinaryJaccardIndex().cuda()
    configCreate(os.path.join(config['logging']['log_dir'].format(experiment_name=config['experiment_name']), 'config.yaml'), config)
    
    trainScript(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion ,
        optimiser=optimiser,
        scheduler=scheduler,
        f1_metric=f1_metric,
        iou_metric=iou_metric,
        num_epochs=config['training']['epochs'],
        config=config,         
        tb_writer=writer     
    )