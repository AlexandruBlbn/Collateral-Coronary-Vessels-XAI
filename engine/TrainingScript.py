import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryF1Score
import torchvision.transforms.functional as tf
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
from monai.networks.nets import attentionunet as AttUnet
from zoo.unext import UNeXt_S
from frangiPreproces import FrangiFilter
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from zoo.models import SegmentatorCoronare, SegmentatorCoronarePlusPlus
from data.dataloader import ArcadeDataset
from torch.utils.tensorboard import SummaryWriter
from utils.helpers import set_seed
import numpy as np
from monai.networks.nets import basic_unet as monai_unet

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = 256
batch_size = 8
epochs = 100
lr = 5e-4
loss_fn = smp.losses.TverskyLoss(mode='binary', log_loss=True, from_logits=True)
f1_metric = BinaryF1Score().to(device)
scaler = torch.amp.GradScaler()

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
                
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        # img_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_np)
        
        frangi = FrangiFilter(img_np, 256)
        frangi_tensor = torch.from_numpy(frangi).unsqueeze(0).float() 
        
        img_tensor = tf.to_tensor(img_np)
        img_tensor = tf.normalize(img_tensor, [0.5], [0.5])
        
        # combined_img = img_tensor
        combined_img = torch.cat([img_tensor, frangi_tensor], dim=0)
        mask = tf.to_tensor(mask)

        return combined_img, mask

train_base = ArcadeDataset(split='train', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
val_base   = ArcadeDataset(split='validation', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
test_base  = ArcadeDataset(split='test', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')

train_ds = TransformsWrapper(train_base, input_size=img_size, mode='train')
val_ds   = TransformsWrapper(val_base, input_size=img_size, mode='val')
test_ds  = TransformsWrapper(test_base, input_size=img_size, mode='val')

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16, persistent_workers=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=16, persistent_workers=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=16, persistent_workers=True)


#----------

from monai.networks.nets import SwinUNETR
# model = SwinUNETR( 
#     in_channels=2,     
#     out_channels=1,     
#     spatial_dims=2,       
#     feature_size=24,     
#     use_checkpoint=True  
# ).cuda()
#---------

model = smp.Unet(
    encoder_name='tu-convnextv2_tiny',
    encoder_weights=None,
    in_channels=2,
    classes=1
)

#relu - leaky, batchnorm - instance norm
for k, v in model.named_children():
    if isinstance(v, nn.ReLU):
        setattr(model, k, nn.LeakyReLU(inplace=True))
    elif isinstance(v, nn.BatchNorm2d):
        setattr(model, k, nn.InstanceNorm2d(v.num_features, affine=True))
        
model = model.cuda()

optimiser = optim.AdamW(model.parameters(), lr=lr)
criterion = smp.losses.FocalLoss(mode='binary', alpha=0.5, gamma=1)
criterion2 = smp.losses.TverskyLoss(mode='binary', log_loss=True, from_logits=True)
scheduler = CosineAnnealingLR(optimiser, T_max=epochs)
writer = SummaryWriter(log_dir='runs/ConvNext_UNET_Frangi')

def train_epoch(model, dataloader, criterion, optimiser, f1_metric, epoch):
    model.train()
    running_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs} - Training")
    for batch_idx, (images, masks) in pbar:
        images, masks = images.cuda(), masks.cuda()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model(images)
            loss = criterion(output, masks)
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    writer.add_scalar('Loss/train', running_loss / len(dataloader), epoch)
    
def validate_epoch(model, dataloader, criterion, f1_metric, epoch):
    model.eval()
    val_f1 = 0.0
    val_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs} - Validation")
        for batch_idx, (images, masks) in pbar:
            images, masks = images.cuda(), masks.cuda()
            output = model(images)
            loss = criterion(output, masks) 
            val_loss += loss.item()
            val_f1 += f1_metric(output.sigmoid(), masks.int()).item()
            pbar.set_postfix({'val_loss': val_loss / (batch_idx + 1), 'val_f1': val_f1 / (batch_idx + 1)})
            if batch_idx == 0:
                img_vis = images[:, 0:1, :, :] * 0.5 + 0.5
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
        
def test_model(model, dataloader, f1_metric, tb_writer):
    model.eval()
    test_f1 = 0.0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")
        for batch_idx, (images, masks) in pbar:
            images, masks = images.cuda(), masks.cuda()
            output = model(images)
            test_f1 += f1_metric(output.sigmoid(), masks.int()).item()
            pbar.set_postfix({'test_f1': test_f1 / (batch_idx + 1)})
    test_f1 = test_f1 / len(dataloader)
    print(f"Test F1: {test_f1:.4f}")
    tb_writer.add_text("Test/F1", f"{test_f1:.4f}")
    return test_f1
    
if __name__ == '__main__':
    encoder_name = encoder_name = getattr(model, 'encoder_name', None) or getattr(model, 'encoder', None)
    check_path = 'checkpoints/Convnext_unet_frangi'
    os.makedirs(check_path, exist_ok=True)
    best_val_f1 = 0.0
    
    with open(os.path.join(check_path, 'config.yaml'), 'w') as f:
        yaml.dump({
            'in_channels': 2,
            'classes': 1,
            'decoder_attention_type': 'scse',
            'optimizer': 'AdamW',
            'learning_rate': lr,
            'loss_function': "Focal Loss",
            'scheduler': 'CosineAnnealingLR',
            'epochs': epochs,
            'batch_size': batch_size,
        }, f)

    for epoch in range(epochs):
        train_epoch(model, train_loader, criterion, optimiser, f1_metric, epoch)
        avg_f1 = validate_epoch(model, val_loader, criterion, f1_metric, epoch)

        is_best = avg_f1 > best_val_f1
        if is_best:
            best_val_f1 = avg_f1

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
        }

        torch.save(checkpoint, os.path.join(check_path, f'last_checkpoint.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(check_path, 'best_model.pth'))
        scheduler.step()

    final_test_f1 = test_model(model, test_loader, f1_metric, tb_writer=writer)
    writer.add_scalar('Test F1', final_test_f1, epochs)
    print(f"Final Test F1: {final_test_f1:.4f}")