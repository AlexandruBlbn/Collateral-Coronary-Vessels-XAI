import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import  CosineAnnealingLR, LinearLR, SequentialLR
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
from torchvision.ops import MLP

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

#------------------------
from deeplab import TransformsWrapper

#------------------------

model = torch.load("/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/LeJepa_coatnet/best_model.pth", map_location=torch.device('cpu'))
backbone = os.path.join("/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/LeJepa_coatnet/backbone_best.pth")
probe = os.path.join("/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/LeJepa_coatnet/probe_best.pth")
torch.save(model['model_state_dict'], backbone)
torch.save(model['probe_state_dict'], probe)
