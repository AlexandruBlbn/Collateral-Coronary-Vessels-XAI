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
from zoo.models import SegmentatorCoronare, SegmentatorCoronarePlusPlus
from torch.utils.tensorboard import SummaryWriter
from utils.helpers import set_seed
import numpy as np
from deeplab import TransformsWrapper

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


#-------------------------------#
epochs = 100
batch_size = 16
img_size = 224
lr1 = 5e-4
lr2 = 1e-4

#-------------------------------#

train_base = ArcadeDataset(split='train', mode='syntax', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
train_ds = TransformsWrapper(train_base, input_size=img_size, mode='train')
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

val_base = ArcadeDataset(split='validation', mode='syntax', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
val_ds = TransformsWrapper(val_base, input_size=img_size, mode='validation')
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

test_base = ArcadeDataset(split='test', mode='syntax', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
test_ds = TransformsWrapper(test_base, input_size=img_size, mode='test')
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

def loadModel(weights_path, device='cuda'):
    weights = torch.load(weights_path, map_location='cpu')
    # if 'model_state_dict' in weights:
    #     weights = weights['model_state_dict']
    
    # new_weights = {}
    # for k, v in weights.items():
    #     if k.startswith("module."): k = k.replace("module.", "")
    #     if k.startswith("backbone."): k = k.replace("backbone.", "")
    #     new_weights[k] = v
    model = timm.create_model('coatnet_1_rw_224', pretrained=False, in_chans=1, features_only=True,out_indices=(0, 1, 2, 3, 4), num_classes=0, global_pool='')
    model.load_state_dict(weights, strict=False)
    return model.to(device)

model = loadModel('/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/LeJepa_coatnet_detach/best_backbone.pth', device='cuda')

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsampling prin convoluție transpusă (dublăm rezoluția spațială)
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        # Convoluțiile după concatenarea cu skip connection-ul
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Asigurăm alinierea dimensiunilor în caz de rotunjiri (ex: 7x7 -> 14x14 vs 15x15)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        # Conectăm informația de detaliu (skip) cu informația semantică (x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes=1):
        super().__init__()
        # encoder_channels pentru CoAtNet_1 sunt de obicei [64, 96, 192, 384, 768]
        
        # Trecem ultimul strat printr-un bloc "Center"
        self.center = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )
        
        # Construim traseul de urcare (decodare)
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=encoder_channels[3], out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=encoder_channels[2], out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=encoder_channels[1], out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64,  skip_channels=encoder_channels[0], out_channels=32)
        
        # Ultimul strat de upsampling către rezoluția originală (224x224)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, features):
        # features este o listă de 5 tensori returnată de model(images)
        x = self.center(features[-1]) 
        x = self.dec4(x, features[3])
        x = self.dec3(x, features[2]) 
        x = self.dec2(x, features[1]) 
        x = self.dec1(x, features[0]) 
        x = self.final_up(x) 
        
        return x

dummy_input = torch.randn(1, 1, 224, 224).cuda()
with torch.no_grad():
    feats = model(dummy_input)
enc_channels = [f.shape[1] for f in feats]

decoder = UNetDecoder(encoder_channels=enc_channels, num_classes=1).cuda()

scaler = torch.amp.GradScaler()
optimiser = optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=lr1)
scheduler = CosineAnnealingLR(optimiser, T_max=epochs)
criterion = smp.losses.TverskyLoss(mode='binary', log_loss=True)
f1_metric = BinaryF1Score().cuda()
writer = SummaryWriter(log_dir='runs/finetune_deeplabv3')



def train_epoch(model, decoder, dataloader, optimiser, criterion, f1_metric, epoch):
    model.train(), decoder.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs} - Training")
    for batch_idx, (images, masks) in pbar:
        images, masks = images.cuda(), masks.cuda()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            features = model(images)
            outputs = decoder(features)
            loss = criterion(outputs, masks)
            
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    writer.add_scalar('Loss/Train', running_loss / len(dataloader), epoch)
    
def validate_epoch(model, decoder, dataloader, criterion, f1_metric, epoch):
    model.eval(), decoder.eval()
    
    val_f1 = 0.0
    with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs} - Validation")
            for batch_idx, (images, masks) in pbar:
                images, masks = images.cuda(), masks.cuda()
                features = model(images)
                outputs = decoder(features)

                preds = (torch.sigmoid(outputs) > 0.5).int()
                f1_val = f1_metric(preds, masks.int()).item()
                val_f1 += f1_val

                if batch_idx == 0:
                    img_vis = images * 0.5 + 0.5
                    num_samples = min(4, img_vis.size(0))
                    grid_images = []
                    preds_vis = torch.sigmoid(outputs)
                    for i in range(num_samples):
                        grid_images.append(img_vis[i].cpu())
                        grid_images.append(preds_vis[i].float().cpu())
                        grid_images.append(masks[i].float().cpu())
                    grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
                    writer.add_image("Val/Predictions", grid, epoch)

            avg_f1 = val_f1 / len(dataloader)
            writer.add_scalar("Val/F1", avg_f1, epoch)
            print(f"Validation F1: {avg_f1:.4f}")
            return avg_f1


def test_model(model, decoder, dataloader, criterion, f1_metric, epoch, tb_writer=None):
        model.eval(), decoder.eval()
        test_f1 = 0.0
        with torch.inference_mode():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")
            for batch_idx, (images, masks) in pbar:
                images, masks = images.cuda(), masks.cuda()
                features = model(images)
                outputs = decoder(features)

                preds = (torch.sigmoid(outputs) > 0.5).int()
                f1_val = f1_metric(preds, masks.int()).item()
                test_f1 += f1_val

                if batch_idx == 0 and tb_writer is not None:
                    img_vis = images * 0.5 + 0.5
                    num_samples = min(4, img_vis.size(0))
                    grid_images = []
                    preds_vis = torch.sigmoid(outputs)
                    for i in range(num_samples):
                        grid_images.append(img_vis[i].cpu())
                        grid_images.append(preds_vis[i].float().cpu())
                        grid_images.append(masks[i].float().cpu())
                    grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
                    tb_writer.add_image("Test/Predictions", grid, epoch)

        avg_f1 = test_f1 / len(dataloader)
        if tb_writer is not None:
            tb_writer.add_scalar("Test/F1", avg_f1, epoch)
        print(f"Test F1: {avg_f1:.4f}")
        return avg_f1


if __name__ == '__main__':
        check_path = 'checkpoints/finetune_unet_coatnet'
        os.makedirs(check_path, exist_ok=True)
        best_val_f1 = 0.0

        for epoch in range(epochs):
            train_epoch(model, decoder, train_loader, optimiser, criterion, f1_metric, epoch)
            avg_f1 = validate_epoch(model, decoder, val_loader, criterion, f1_metric, epoch)

            is_best = avg_f1 > best_val_f1
            if is_best:
                best_val_f1 = avg_f1

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
            }

            torch.save(checkpoint, os.path.join(check_path, f'last_checkpoint.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(check_path, 'best_model.pth'))
                torch.save(model.state_dict(), os.path.join(check_path, 'best_backbone.pth'))
                torch.save(decoder.state_dict(), os.path.join(check_path, 'best_decoder.pth'))

            print(f"Epoch {epoch+1}: best F1 so far {best_val_f1:.4f}")
            scheduler.step()

        final_test_f1 = test_model(model, decoder, test_loader, criterion, f1_metric, epochs, tb_writer=writer)
        writer.add_scalar('Test/Final_F1', final_test_f1, epochs)
        print(f"Final Test F1: {final_test_f1:.4f}")

