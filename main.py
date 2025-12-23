# from utils.helpers import set_seed
# from utils.zoo import UNet
# import torch
# from torch.utils.data import DataLoader
# import torch.nn as nn
# from data.ARCADE.dataloader import ARCADEDataset
# from tqdm import tqdm
# from utils.metrics import dice_coefficient, iou, DiceLoss
# import matplotlib.pyplot as plt
# import numpy as np
# from torch.amp import autocast, GradScaler
# import torch.backends.cudnn as cudnn
# import os

# set_seed(42)

# dataset_path = 'data/ARCADE/processed/dataset.json'

# '''
# taskuri:
# SegStenoza
# SegCoronare
# Clasificare
# Unsupervised
# '''

# train_dataset = ARCADEDataset(dataset_path, split='train', task='SegCoronare')
# val_dataset = ARCADEDataset(dataset_path, split='test', task='SegCoronare')
# test_dataset = ARCADEDataset(dataset_path, split='test', task='SegCoronare')

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)


import sys
import os
import torch
import yaml
from torch.utils.data import DataLoader

# Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from data.ARCADE.MIM import MaskGenerator, ArcadeDatasetMIM
from data.ARCADE.dataloader import ARCADEDataset

# --- INSEREAZĂ FUNCȚIA inspect_input_data AICI ---
# (Copiază codul funcției de mai sus)

