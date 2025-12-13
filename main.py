from utils.helpers import set_seed
from utils.zoo import UNet
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.ARCADE.dataloader import ARCADEDataset
from tqdm import tqdm
from utils.metrics import dice_coefficient, iou, DiceLoss
import matplotlib.pyplot as plt
import numpy as np
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import os

set_seed(42)

dataset_path = 'data/ARCADE/processed/dataset.json'

'''
taskuri:
SegStenoza
SegCoronare
Clasificare
'''

train_dataset = ARCADEDataset(dataset_path, split='train', task='SegCoronare')
val_dataset = ARCADEDataset(dataset_path, split='test', task='SegCoronare')
test_dataset = ARCADEDataset(dataset_path, split='test', task='SegCoronare')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
