import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import  CosineAnnealingLR
from tqdm import tqdm
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryF1Score
import torchvision.transforms.functional  as tf
from PIL import Image 
import timm as timm
import torchvision
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataloader import ArcadeDataset
from utils.logger import TensorboardLogger
import segmentation_models_pytorch as smp
import cv2
from monai.networks.nets.basic_unet import BasicUNet
from torchinfo import summary

import monai


from torchmetrics.classification import (
    BinaryJaccardIndex, 
    BinaryF1Score
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from zoo.models import SegmentatorCoronare, SegmentatorCoronarePlusPlus
from data.dataloader import ArcadeDataset
from torch.utils.tensorboard import SummaryWriter
from utils.helpers import set_seed
import numpy as np



set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(encoder_name="tu-focalnet_small_srf", encoder_weights=None,in_channels=1, classes=1, activation=None).to(device)
Name = model.__class__.__name__ + "_" + model.encoder.name #decoder-encoder
#---------------------
batch_size = 16
epochs = 100
learning_rate = 1e-4
min_lr = 1e-6

checkpoint_dir = f'checkpoints/{Name}'
log_dir = f'runs/{Name}'
os.makedirs(checkpoint_dir, exist_ok=True)
img_size = 256



# model = BasicUNet(
#     spatial_dims=2,
#     in_channels=1,
#     out_channels=1,
#     features=(32, 64, 128, 256, 512, 32),
#     act="LeakyReLU",
#     norm="instance",
#     dropout=0.2
# ).to(device) - nnU-Net


#---------------------

def saveConfig(path):
    config = {
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'min_lr': min_lr,
        'model': Name
        
    }
    with open(path, 'w') as f:
        yaml.dump(config, f)
    
saveConfig(os.path.join(checkpoint_dir, 'config.yaml'))

class TransformsWrapper():
    def __init__(self, dataset, input_size=224, mode='train'):
        self.dataset = dataset
        self.input_size = input_size
        self.mode = mode
        self.root_dir = '.'
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        mask = label.resize((self.input_size, self.input_size), resample=Image.NEAREST)

        if self.mode == 'train':
            if torch.rand(1).item() > 0.5:
                img = tf.hflip(img)
                mask = tf.hflip(mask)
            if torch.rand(1).item() > 0.5:
                img = tf.vflip(img)
                mask = tf.vflip(mask)
            
            angle = torch.randint(-15, 15, (1,)).item()
            img = tf.rotate(img, angle)
            mask = tf.rotate(mask, angle)
            
        if self.mode == "lejepa":
                img = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(transforms.RandomVerticalFlip()(img), p=0.5),
                transforms.RandomApply(transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))(img), p=0.2),
                transforms.RandomApply(transforms.RandomSolarize(threshold=0.5)(img), p=0.2)])
                
            

        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(np.array(img))
        img = tf.to_tensor(img)
        img = tf.normalize(img, [0.5], [0.5])
        mask = tf.to_tensor(mask)

        return img, mask
            
        


train_base = ArcadeDataset(split='train', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
val_base   = ArcadeDataset(split='validation', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
test_base  = ArcadeDataset(split='test', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')

train_ds = TransformsWrapper(train_base, input_size=img_size, mode='train')
val_ds   = TransformsWrapper(val_base, input_size=img_size, mode='val')
test_ds  = TransformsWrapper(test_base, input_size=img_size, mode='val')

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)




optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
bce_loss = smp.losses.SoftBCEWithLogitsLoss()
criterion = dice_loss
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

writer = SummaryWriter(log_dir=log_dir)

iou = BinaryJaccardIndex().to(device)
f1 = BinaryF1Score().to(device)

#----------------------
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)
    for batch_idx, (images, masks) in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type = device_type, dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        current_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': current_loss})
        writer.add_scalar("Train Loss", current_loss, pbar.n + 1)
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device, epoch_num):
    model.eval()
    epoch_loss = 0
    iou_score = 0
    f1_score = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validare", leave=False)
    with torch.no_grad():
        for (batch_idx, (images, masks)) in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            
            preds = (outputs > 0.0).float()
            iou_score += iou(preds, masks.int()).item()
            f1_score += f1(preds, masks.int()).item()
            
            if batch_idx == 0:
                img = images * 0.5 + 0.5
                num_samples = min(4, images.size(0))
                grid_images = []
                for i in range(num_samples):
                    grid_images.append(img[i].cpu())
                    grid_images.append(preds[i].float().cpu())
                    grid_images.append(masks[i].float().cpu())
                grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
                writer.add_image("Val Predictions", grid, epoch_num)
            
            pbar.set_postfix({
                'loss': epoch_loss / (pbar.n + 1),
                'iou': iou_score / (pbar.n + 1),
                'f1': f1_score / (pbar.n + 1)
            })
        print(f"Val Loss: {epoch_loss / len(dataloader):.4f} | IoU: {iou_score / len(dataloader):.4f} | F1: {f1_score / len(dataloader):.4f}")
        writer.add_scalar("Val F1", f1_score / len(dataloader), epoch_num)
        writer.add_scalar("Val IoU", iou_score / len(dataloader), epoch_num)        
    return epoch_loss / len(dataloader), iou_score / len(dataloader), f1_score / len(dataloader)


def test_epoch(model, dataloader, device):
    model.eval()
    iou_score = 0
    f1_score = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testare", leave=False)
    with torch.no_grad():
        for (batch_idx, (images, masks)) in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            preds = (outputs > 0.0).float()
            iou_score += iou(preds, masks.int()).item()
            f1_score += f1(preds, masks.int()).item()
            
            if batch_idx == 0:
                img = images * 0.5 + 0.5
                num_samples = min(4, images.size(0))
                grid_images = []
                for i in range(num_samples):
                    grid_images.append(img[i].cpu())
                    grid_images.append(preds[i].float().cpu())
                    grid_images.append(masks[i].float().cpu())
                grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
                writer.add_image("Test Predictions", grid, 0)
            
            pbar.set_postfix({
                'iou': iou_score / (pbar.n + 1),
                'f1': f1_score / (pbar.n + 1)
            })
        print(f"Test IoU: {iou_score / len(dataloader):.4f} | F1: {f1_score / len(dataloader):.4f}")
        writer.add_scalar("Test F1", f1_score / len(dataloader), 0)
        writer.add_scalar("Test IoU", iou_score / len(dataloader), 0)
    
    return iou_score / len(dataloader), f1_score / len(dataloader)


if __name__ == "__main__":
    best_f1 = 0
    # summary(model, input_size=(16, 1, 256,256))
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou, val_f1 = validate_epoch(model, val_loader, criterion, device, epoch)
        scheduler.step()
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"Model cu F1 nou salvat: {best_f1:.4f}")
    
    #test
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))
    test_iou, test_f1 = test_epoch(model, test_loader, device)
    
