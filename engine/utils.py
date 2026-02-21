import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
import segmentation_models_pytorch as smp
import cv2
from torchinfo import summary
import monai
from frangiPreproces import FrangiFilter
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from data.dataloader import ArcadeDataset
from data.transformWrapper import TransformsWrapper as TW
from torch.utils.tensorboard import SummaryWriter
from utils.helpers import set_seed
import numpy as np
import random
scaler = torch.amp.GradScaler()



set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def loader(img_size, batch_size, split='train'):
    """_returneaza direct dataloaderii_

    Parametrii:
        img_size (_int_): _Dimensiunea imaginilor_
        batch_size (_int_): _batchsize_
        split (_str_): _Split pentru datalaoder_. Tipuri: train, validation, test.

    Returns:
        _dataloader_: => _Dataloader pentru split-ul specificat_"""
        
    #seed pentru DL
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
    """_Salveaza configuratia intr-un fisier yaml_

    Parametrii:
        path (_str_): _Calea catre fisierul yaml_
        config (_dict_): _Dictionar cu configuratia_
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)
        

        
#-------------------------------------
# Logging si configuratie
# Train:
# Loss.
# Validation:
# Loss, F1 pt > 0.5, F1 pt < 0.5 (yonden procent dif valori), Sampleuri: Data | GT | Pred > 0.5 | Pred < 0.5, Sampleuri yonden ( pt procente 1 pacient, acelasi)
# Test:
# F1 pt > 0.5, F1 pt preds <0.5, youden cu best pe de pe validare. Sampleuri: Data | GT | Pred > 0.5 | Pred < 0.5, Sampleuri yonden ( pt procente 1 pacient, acelasi)


# Configuratie model, optimizator, scheduler, criterii, metrici, etc.
#optimiser -> WD salvat in config
#criterion name -> salvat
#scheduler -> propietati salvate

#-------------------------------------

config = {
    'experiment_name': 'ConvNext_Model',
    'logging': {
        'log_dir': 'runs/{experiment_name}',
        'checkpoint_dir': 'checkpoints/{experiment_name}'
    },
    'training': {
        'img_size': 256,
        'batch_size': 8,
        'epochs': 100,
        'learning_rate': 5e-4,
        'loss_function': "Focal Loss",
        'scheduler': 'CosineAnnealingLR',
        'precision': 'bfloat16',
    },
    'model': {
        'model': 'summary',
    }
    }

writer = SummaryWriter(log_dir=config['logging']['log_dir'].format(experiment_name=config['experiment_name']))

    
def modelChange(model, old_layer, new_layer):
    for k, v in model.named_children():
        if isinstance(v, old_layer):
            setattr(model, k, new_layer)
    return model.cuda()

    
def train_epoch(model, dataloader, criterion, optimiser, f1_metric, epoch):
    model.train()
    running_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
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
        
        
#todo:
#youden index







# def test_model(model, dataloader, f1_metric, tb_writer):
#     model.eval()
#     test_f1 = 0.0
#     with torch.no_grad():
#         pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")
#         for batch_idx, (images, masks) in pbar:
#             images, masks = images.cuda(), masks.cuda()
#             output = model(images)
#             test_f1 += f1_metric(output.sigmoid(), masks.int()).item()
#             pbar.set_postfix({'test_f1': test_f1 / (batch_idx + 1)})
#     test_f1 = test_f1 / len(dataloader)
#     print(f"Test F1: {test_f1:.4f}")
#     tb_writer.add_text("Test/F1", f"{test_f1:.4f}")
#     return test_f1
    
# if __name__ == '__main__':
#     encoder_name = encoder_name = getattr(model, 'encoder_name', None) or getattr(model, 'encoder', None)
#     check_path = 'checkpoints/Convnext_unet_frangi'
#     os.makedirs(check_path, exist_ok=True)
#     best_val_f1 = 0.0
    
#     with open(os.path.join(check_path, 'config.yaml'), 'w') as f:
#         yaml.dump({
#             'in_channels': 2,
#             'classes': 1,
#             'decoder_attention_type': 'scse',
#             'optimizer': 'AdamW',
#             'learning_rate': lr,
#             'loss_function': "Focal Loss",
#             'scheduler': 'CosineAnnealingLR',
#             'epochs': epochs,
#             'batch_size': batch_size,
#         }, f)

#     for epoch in range(epochs):
#         train_epoch(model, train_loader, criterion, optimiser, f1_metric, epoch)
#         avg_f1 = validate_epoch(model, val_loader, criterion, f1_metric, epoch)

#         is_best = avg_f1 > best_val_f1
#         if is_best:
#             best_val_f1 = avg_f1

#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimiser_state_dict': optimiser.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'best_val_f1': best_val_f1,
#         }

#         torch.save(checkpoint, os.path.join(check_path, f'last_checkpoint.pth'))
#         if is_best:
#             torch.save(checkpoint, os.path.join(check_path, 'best_model.pth'))
#         scheduler.step()

#     final_test_f1 = test_model(model, test_loader, f1_metric, tb_writer=writer)
#     writer.add_scalar('Test F1', final_test_f1, epochs)
#     print(f"Final Test F1: {final_test_f1:.4f}")


