from utils.helpers import set_seed
from utils.zoo import UNet
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.ARCADE.dataloader import ArcadeCoronarySegmentation
from tqdm import tqdm
from utils.metrics import dice_coefficient, iou, DiceLoss
import matplotlib.pyplot as plt
import numpy as np
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import os

set_seed(42)


dataset_path = 'D:\Collateral Coronary Vessels XAI\data\ARCADE\processed\dataset.json'

dataset = ArcadeCoronarySegmentation(dataset_path)
train_dataset = ArcadeCoronarySegmentation(dataset_path, split='train', augment=False)
val_dataset = ArcadeCoronarySegmentation(dataset_path, split='test', augment=False)  # Folosim test pentru validare
test_dataset = ArcadeCoronarySegmentation(dataset_path, split='test', augment=False)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(in_channels=1, out_channels=1).to(device)
# Reduce pos_weight to 3 (less aggressive) and combine with DiceLoss for fine structures
pos_weight = torch.tensor([3.0], device=device)
bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
dice_loss = DiceLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
use_amp = (device == 'cuda')
scaler = GradScaler('cuda', enabled=use_amp)
epochs = 20
