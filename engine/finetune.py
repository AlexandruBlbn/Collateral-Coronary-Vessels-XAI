import os
import sys
import yaml
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision
import segmentation_models_pytorch as smp
from torchinfo import summary
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
import monai
from monai.losses import DiceCELoss
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import gc

# Ajustează căile către dataset și modulele tale
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import ArcadeDataset
from data.transformWrapper import TransformsWrapper as TW
from utils.helpers import set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler()

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

# Funcție pentru încărcarea backbone-ului preantrenat
def load_pretrained_backbone(smp_model, backbone_path):
    if not os.path.exists(backbone_path):
        print(f"⚠️ Nu s-a găsit backbone la {backbone_path}. Se va folosi modelul cu greutăți inițiale.")
        return smp_model

    print(f"🔄 Se încarcă backbone-ul de la: {backbone_path}")
    state_dict = torch.load(backbone_path, map_location='cuda')

    # Accept either a bare backbone state_dict (best_backbone.pth) or a full
    # training checkpoint containing model_state_dict.
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        full_state = state_dict['model_state_dict']
        # Keep only backbone weights if this is a full SparK checkpoint.
        if any(k.startswith('backbone.') for k in full_state.keys()):
            state_dict = {
                k.replace('backbone.', '', 1): v
                for k, v in full_state.items()
                if k.startswith('backbone.')
            }
        else:
            state_dict = full_state
    
    # SMP stochează modelul timm original în `smp_model.encoder.model` pentru encoderele 'tu-*'
    try:
        missing, unexpected = smp_model.encoder.model.load_state_dict(state_dict, strict=False)
        print(f"✅ Backbone încărcat cu succes! Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    except Exception as e:
        print(f"⚠️ Eroare la încărcarea directă, se încearcă fallback-ul manual... Detalii: {e}")
        # Fallback dacă structura variază ușor
        new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        smp_model.encoder.load_state_dict(new_state_dict, strict=False)
        print("✅ Fallback aplicat cu succes.")
        
    return smp_model


def get_decoder_config(base_encoder):
    if base_encoder == 'resnet50':
        return 5, (512, 256, 128, 64, 32)
    if base_encoder == 'convnextv2_tiny':
        # For SMP Unet++ with timm universal encoders, using depth=5 avoids
        # zero-channel nested decoder blocks seen with depth=4 on ConvNeXtV2.
        return 5, (256, 128, 64, 32, 16)
    raise ValueError(f"Encoder nesuportat pentru fine-tuning SparK: {base_encoder}")


def build_segmentation_model(base_encoder):
    smp_encoder_name = f"tu-{base_encoder}"
    enc_depth, dec_channels = get_decoder_config(base_encoder)
    model = smp.UnetPlusPlus(
        encoder_name=smp_encoder_name,
        encoder_weights=None,
        in_channels=1,
        classes=1,
        encoder_depth=enc_depth,
        decoder_channels=dec_channels,
        decoder_use_batchnorm=True,
        decoder_attention_type='scse'
    )

    # SMP + tu-convnextv2_tiny currently exposes an intermediate skip with
    # 0 channels (encoder.out_channels contains [..., 0, ...]), which makes
    # Unet++ instantiate invalid conv layers (out_channels=0). If detected,
    # fall back to Unet so the experiment suite can run end-to-end.
    bad_skip = any(ch == 0 for ch in getattr(model.encoder, 'out_channels', []))
    if bad_skip:
        print(f"⚠️  Unet++ incompatibil cu {smp_encoder_name} (skip cu 0 canale). "
              "Fallback la Unet pentru acest encoder.")
        model = smp.Unet(
            encoder_name=smp_encoder_name,
            encoder_weights=None,
            in_channels=1,
            classes=1,
            encoder_depth=enc_depth,
            decoder_channels=dec_channels,
            decoder_use_batchnorm=True,
            decoder_attention_type='scse'
        )

    return model.cuda()


def build_optimiser(model, config):
    mode = config['training']['mode']
    if mode == 'frozen':
        print("❄️  Mod FROZEN: Backbone-ul este înghețat, se antrenează doar decoder-ul.")
        for param in model.encoder.parameters():
            param.requires_grad = False

        return optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['learning_rate_decoder'],
            weight_decay=1e-4
        )

    print("🔥 Mod UNFROZEN: Fine-tuning complet (Backbone LR mai mic, Decoder LR mai mare).")
    for param in model.encoder.parameters():
        param.requires_grad = True

    return optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config['training']['learning_rate_encoder']},
        {'params': model.decoder.parameters(), 'lr': config['training']['learning_rate_decoder']},
        {'params': model.segmentation_head.parameters(), 'lr': config['training']['learning_rate_decoder']}
    ], weight_decay=1e-4)

# Funcțiile de validare, testare și evaluare praguri rămân la fel ca în scriptul tău original
def train_epoch(model, dataloader, criterion, optimiser, f1_metric, epoch, writer, current_step):
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
        writer.add_scalar('Loss/train_step', loss.item(), current_step)
        current_step += 1
        
    writer.add_scalar('Loss/train_epoch', running_loss / len(dataloader), epoch)
    return current_step

def validate_epoch(model, dataloader, criterion, f1_metric, epoch, writer):
    model.eval()
    val_f1 = 0.0
    val_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1} - Validation")
        for batch_idx, (images, masks) in pbar:
            images, masks = images.cuda(), masks.cuda()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
        config['logging']['log_dir'], 
        'thresholds_results.csv'
    )
    
    all_probs, all_masks = [], []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.cuda(), masks.cuda()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
            writer.writerow([f"{t:.2f}", f"{current_f1:.4f}", f"{current_iou:.4f}"])

def test_model(model, dataloader, f1_metric, iou_metric, tb_writer, prefix="Test"):
    model.eval()
    test_f1 = 0.0
    test_iou = 0.0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Testing ({prefix})")
        for batch_idx, (images, masks) in pbar:
            images, masks = images.cuda(), masks.cuda()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(images)
            preds = (output.sigmoid() > 0.5).int()
            test_f1 += f1_metric(preds, masks.int()).item()
            test_iou += iou_metric(preds, masks.int()).item()
            pbar.set_postfix({'f1': test_f1 / (batch_idx + 1), 'iou': test_iou / (batch_idx + 1)})
            
    test_f1 = test_f1 / len(dataloader)
    test_iou = test_iou / len(dataloader)
    print(f"{prefix} F1: {test_f1:.4f}, IoU: {test_iou:.4f}")
    tb_writer.add_scalar(f"{prefix}/F1", test_f1)
    tb_writer.add_scalar(f"{prefix}/IoU", test_iou)
    return test_f1, test_iou

def test_model_youden(model, dataloader, f1_metric, iou_score, tb_writer, optimal_threshold):
    model.eval()
    test_f1 = 0.0
    test_iou = 0.0  
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing (Youden)")
        for batch_idx, (images, masks) in pbar:
            images, masks = images.cuda(), masks.cuda()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                probs = model(images).sigmoid()
            preds_bin = (probs > optimal_threshold).int()
            test_f1 += f1_metric(preds_bin, masks.int()).item()
            test_iou += iou_score(preds_bin, masks.int()).item()
            pbar.set_postfix({'f1': test_f1 / (batch_idx + 1), 'iou': test_iou / (batch_idx + 1)})       
            
    test_f1 = test_f1 / len(dataloader)
    test_iou = test_iou / len(dataloader)
    print(f"Test F1 (Youden): {test_f1:.4f}, Test IoU (Youden): {test_iou:.4f}")
    tb_writer.add_scalar("Test_Youden/F1", test_f1)
    tb_writer.add_scalar("Test_Youden/IoU", test_iou)
    return test_f1, test_iou
    
def trainScript(model, train_loader, val_loader, test_loader, criterion, optimiser, scheduler, f1_metric, iou_metric, config, tb_writer):
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    best_val_f1 = 0.0
    current_step = 0
    
    for epoch in range(config['training']['epochs']):
        current_step = train_epoch(model, train_loader, criterion, optimiser, f1_metric, epoch, tb_writer, current_step)
        val_f1 = validate_epoch(model, val_loader, criterion, f1_metric, epoch, tb_writer)
        scheduler.step()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Model salvat cu F1 = {best_val_f1:.4f}")

    print("\n" + "="*50)
    print("Începe Testarea...")
    print("="*50 + "\n")

    model.load_state_dict(torch.load(best_model_path))
    print("--> Test cu prag implicit (0.5)")
    test_model(model, test_loader, f1_metric, iou_metric, tb_writer, prefix="Test_Sigmoid")
    
    best_thresh = find_best_f1_threshold(model, val_loader, f1_metric)
    print(f"--> Cel mai bun prag obținut pe validare: {best_thresh:.4f}")
    
    test_model_youden(model, test_loader, f1_metric, iou_metric, tb_writer, best_thresh)
    evaluate_all_thresholds(model, test_loader, f1_metric, iou_metric, config)


if __name__ == "__main__":

    encoders = ['convnextv2_tiny']
    modes = ['frozen', 'unfrozen']
    
    # Detalii generale training
    img_size = 256
    batch_size = 16
    epochs = 100
    
    # Încărcăm Dataloaders o singură dată pentru a fi eficienți
    train_loader = loader(img_size, batch_size, split='train')
    val_loader = loader(img_size, batch_size, split='validation')
    test_loader = loader(img_size, batch_size, split='test')
    
    for base_encoder in encoders:
        for mode in modes:
            
            experiment_name = f"{base_encoder}_{mode}_finetune"
            print(f"\n{'='*60}\n🚀 Începe antrenamentul pentru: {experiment_name}\n{'='*60}")
            
            config = {
                'experiment_name': experiment_name,
                'logging': {
                    'log_dir': f'runs/{experiment_name}',
                    'checkpoint_dir': f'checkpoints/{experiment_name}'
                },
                'training': {
                    'img_size': img_size,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'learning_rate_decoder': 5e-4, # LR normal pentru decoder
                    'learning_rate_encoder': 1e-5, # LR mic pentru encoder (folosit doar în unfrozen)
                    'mode': mode
                }
            }
            
            writer = SummaryWriter(log_dir=config['logging']['log_dir'])
            configCreate(os.path.join(config['logging']['log_dir'], 'config.yaml'), config)
            
            model = build_segmentation_model(base_encoder)

            # Calea către backbone-ul preantrenat cu SparK
            pretrained_backbone_path = f"checkpoints/{base_encoder}_spark/best_backbone.pth"
            model = load_pretrained_backbone(model, pretrained_backbone_path)

            optimiser = build_optimiser(model, config)

            criterion = DiceCELoss(include_background=True, sigmoid=True, lambda_ce=1, lambda_dice=1)
            scheduler = CosineAnnealingLR(optimiser, T_max=config['training']['epochs'])
            
            f1_metric = BinaryF1Score().cuda()
            iou_metric = BinaryJaccardIndex().cuda()
            
            # Executare Training Script
            trainScript(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                criterion=criterion,
                optimiser=optimiser,
                scheduler=scheduler,
                f1_metric=f1_metric,
                iou_metric=iou_metric,
                config=config,        
                tb_writer=writer    
            )
            
            
            # Cleanup memorie pentru a evita OOM înainte de următorul experiment
            writer.close()
            del model, optimiser, scheduler, criterion, f1_metric, iou_metric
            torch.cuda.empty_cache()
            gc.collect()

    print("\n🎉 Toate experimentele au fost finalizate cu succes!")