import os
import sys
import math
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import timm

from skimage.morphology import skeletonize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. DATA PREPARATION (Vector Field & Centerline Ground Truth Generation)
# ==============================================================================
def generate_vector_field_and_centerline(mask_np: np.ndarray):
    """
    Transformă o mască binară (0, 1) într-o hartă de centerline și un câmp vectorial 2D.
    Vectorii vor indica direcția dinspre marginea vasului către centrul acestuia (Centerline).
    """
    # 1. Scheletizare (Centerline)
    centerline = skeletonize(mask_np > 0).astype(np.float32)
    
    # 2. Distance Transform (Distanța fiecărui pixel până la fundal)
    mask_u8 = (mask_np * 255).astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    
    # 3. Gradientul distanței (Aflăm direcția în care crește distanța -> spre centru)
    dy, dx = np.gradient(dist)
    
    # 4. Normalizare pentru a obține un Vector de Direcție Unitar (dx, dy)
    magnitude = np.sqrt(dx**2 + dy**2 + 1e-8)
    dx_norm = dx / magnitude
    dy_norm = dy / magnitude
    
    # Păstrăm vectorii exclusiv în interiorul vaselor de sânge, pe fundal punem 0
    dx_norm = dx_norm * mask_np
    dy_norm = dy_norm * mask_np
    
    # Returnăm: Câmpul vectorial [2, H, W] și Scheletul [1, H, W]
    vector_field = np.stack([dx_norm, dy_norm], axis=0)
    return vector_field.astype(np.float32), np.expand_dims(centerline, axis=0).astype(np.float32)


class DummyHybridDataset(Dataset):
    """ Un dataset dummy pentru a demonstra funcționarea arhitecturii în afara ecosistemului tău. """
    def __init__(self, size=8, img_size=256):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Imagine Dummy X-Ray
        img = torch.randn(1, self.img_size, self.img_size)
        
        # Mască Binară Dummy (Un dreptunghi înclinat ca vas de sânge)
        mask_np = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        cv2.line(mask_np, (50, 50), (200, 200), 1, thickness=15)
        
        # Generare Ground Truth Avansat
        vector_field, centerline = generate_vector_field_and_centerline(mask_np)
        
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()
        centerline_t = torch.from_numpy(centerline)
        vector_t = torch.from_numpy(vector_field)
        
        return img, mask_t, centerline_t, vector_t


# ==============================================================================
# 2. MODEL ARCHITECTURE (Deformable Convolutions + Directional U-Net)
# ==============================================================================
class DeformableConvBlock(nn.Module):
    """
    Înlocuiește convoluția standard 3x3. 
    Rețeaua învață 'offset'-uri pentru a deforma grid-ul filtrului să urmărească vasele!
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # Offset-ul are 2 * kernel_size^2 canale (dx și dy pentru fiecare element din kernel)
        offset_channels = 2 * kernel_size * kernel_size
        
        self.offset_conv = nn.Conv2d(in_channels, offset_channels, kernel_size=kernel_size, padding=padding)
        # Inițializăm offset-urile cu 0 (la început e o convoluție normală, apoi învață să se deformeze)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        self.dcn = ops.DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.dcn(x, offset)
        return self.relu(self.bn(x))


class DCN_UNetDecoder(nn.Module):
    """ Decodor U-Net care folosește Deformable Convolutions pentru a reconstrui hărțile de trăsături. """
    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        # encoder_channels vin inversate (de la bottleneck spre stratul inițial)
        self.stages = nn.ModuleList()
        
        in_ch = encoder_channels[0]
        for i in range(len(decoder_channels)):
            out_ch = decoder_channels[i]
            skip_ch = encoder_channels[i+1] if (i+1) < len(encoder_channels) else 0
            
            # Deformable Block pentru a procesa imaginea upsamplată concatenată cu skip-ul
            block = nn.Sequential(
                DeformableConvBlock(in_ch + skip_ch, out_ch),
                DeformableConvBlock(out_ch, out_ch)
            )
            self.stages.append(block)
            in_ch = out_ch
            
        self.out_channels = in_ch

    def forward(self, features):
        # features = [f1, f2, f3, f4, f5], f5 e bottleneck-ul
        features = features[::-1] 
        x = features[0]
        
        for i, stage in enumerate(self.stages):
            # Upsampling
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Skip Connection (dacă mai există hărți în encoder)
            if (i + 1) < len(features):
                skip = features[i + 1]
                # Asigurare dimensiuni egale
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
                
            x = stage(x)
            
        return x


class HybridSOTAVesselNetwork(nn.Module):
    """
    Arhitectură hibridă State-of-the-Art (Encoder TIMM + Decodor DCN + Capete Multi-Task)
    """
    def __init__(self, encoder_name="tu-efficientnetv2_s", in_channels=1):
        super().__init__()
        
        timm_model_name = encoder_name.replace("tu-", "") if encoder_name.startswith("tu-") else encoder_name
        # 1. Extractor de Trăsături (Encoder)
        self.encoder = timm.create_model(timm_model_name, pretrained=False, in_chans=in_channels, features_only=True)
        # Obținem dimensiunile canalelor generate de timm (ex: [24, 48, 64, 160, 256])
        enc_channels = self.encoder.feature_info.channels()
        
        # 2. Decodor bazat pe Deformable Convolutions
        self.decoder = DCN_UNetDecoder(encoder_channels=enc_channels[::-1])
        dec_out_ch = self.decoder.out_channels
        
        # 3. Capete (Heads) Inteligente (Aici e "magia" lucrării)
        # A. Masca principală de vase de sânge
        self.mask_head = nn.Conv2d(dec_out_ch, 1, kernel_size=1)
        
        # B. Centerline (Scheletul - forțează rețeaua să găsească miezul vasului)
        self.centerline_head = nn.Conv2d(dec_out_ch, 1, kernel_size=1)
        
        # C. Vector Field (Direcția curgerii/formei, 2 canale: dx, dy)
        self.vector_head = nn.Sequential(
            nn.Conv2d(dec_out_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Tanh() # Tanh forțează output-ul între -1 și 1 (direcții)
        )

    def forward(self, x):
        # Mărim imaginea inițială pentru upsample-ul final (dacă e nevoie)
        orig_size = x.shape[2:]
        
        # Forward Encoder
        feats = self.encoder(x)
        
        # Forward Decoder DCN
        dec_out = self.decoder(feats)
        
        # Dacă decoderul nu a ajuns la dimensiunea originală a imaginii (din cauza pooling-urilor encoderului)
        if dec_out.shape[2:] != orig_size:
            dec_out = F.interpolate(dec_out, size=orig_size, mode='bilinear', align_corners=False)
            
        # Predicții
        mask_logits = self.mask_head(dec_out)
        centerline_logits = self.centerline_head(dec_out)
        vector_field = self.vector_head(dec_out)
        
        return mask_logits, centerline_logits, vector_field


# ==============================================================================
# 3. LOSS FUNCTIONS (Hybrid Directional Loss)
# ==============================================================================
def focal_tversky_loss(pred_logits, true_mask, alpha=0.3, beta=0.7, gamma=1.33):
    probs = torch.sigmoid(pred_logits)
    p = probs.flatten(1)
    t = true_mask.flatten(1)
    
    tp = (p * t).sum(dim=1)
    fp = (p * (1.0 - t)).sum(dim=1)
    fn = ((1.0 - p) * t).sum(dim=1)
    
    tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    return torch.pow((1.0 - tversky), gamma).mean()

def bce_loss(pred_logits, true_mask):
    return F.binary_cross_entropy_with_logits(pred_logits, true_mask)

def vector_direction_loss(pred_vector, true_vector, true_mask):
    """
    Pierderea pentru Câmpul Vectorial. Calculăm asemănarea dintre vectorul prezis (dx, dy)
    și cel real, DOAR în zonele unde există vas de sânge. (Cosine Similarity)
    """
    # pred_vector: [B, 2, H, W]
    # Cosine Similarity = (A dot B) / (norm(A) * norm(B))
    dot_product = (pred_vector * true_vector).sum(dim=1, keepdim=True)
    
    norm_pred = torch.norm(pred_vector, p=2, dim=1, keepdim=True)
    norm_true = torch.norm(true_vector, p=2, dim=1, keepdim=True)
    
    cos_sim = dot_product / (norm_pred * norm_true + 1e-8)
    
    # Cos_sim e 1 dacă sunt la fel, -1 dacă sunt opuse. 
    # Loss-ul este (1 - cos_sim) -> minim 0.
    loss = 1.0 - cos_sim
    
    # Aplicăm loss-ul doar pe pixelii care aparțin vasului de sânge
    loss = loss * true_mask
    
    # Media pierderii doar pentru pixelii vizați
    num_vessel_pixels = true_mask.sum() + 1e-8
    return loss.sum() / num_vessel_pixels


class HybridMultiTaskLoss(nn.Module):
    def __init__(self, w_mask=1.0, w_center=0.5, w_vector=0.5):
        super().__init__()
        self.w_mask = w_mask
        self.w_center = w_center
        self.w_vector = w_vector
        
    def forward(self, preds, targets):
        mask_logits, centerline_logits, pred_vector = preds
        true_mask, true_centerline, true_vector = targets
        
        # 1. Mask Loss (Standard Tversky + BCE)
        loss_m = bce_loss(mask_logits, true_mask) + focal_tversky_loss(mask_logits, true_mask)
        
        # 2. Centerline Loss (BCE focusat, ca un pre-clDice)
        loss_c = bce_loss(centerline_logits, true_centerline)
        
        # 3. Vector Loss (Topological continuity via Direction)
        loss_v = vector_direction_loss(pred_vector, true_vector, true_mask)
        
        total_loss = (self.w_mask * loss_m) + (self.w_center * loss_c) + (self.w_vector * loss_v)
        
        return total_loss, loss_m, loss_c, loss_v


# ==============================================================================
# 4. TEST DE FUNCȚIONARE (Sanity Check)
# ==============================================================================
if __name__ == "__main__":
    print(f"[{device.type.upper()}] Pornire script arhitectura hibridă SOTA...")
    
    # 1. Creare Model
    print("\n--- 1. Construire Model (EfficientNetV2 + DCN Decoder + Multi-Head) ---")
    model = HybridSOTAVesselNetwork(encoder_name="tu-efficientnetv2_s", in_channels=1).to(device)
    print("Modelul a fost construit cu succes!")
    
    # 2. Creare Dataloader & Loss
    ds = DummyHybridDataset(size=4, img_size=256)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    criterion = HybridMultiTaskLoss()
    
    # 3. Forward & Backward Pass Test
    print("\n--- 2. Testare Flux (Forward & Backward) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for batch_idx, (imgs, masks, centerlines, vectors) in enumerate(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        centerlines = centerlines.to(device)
        vectors = vectors.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        preds = model(imgs)
        mask_out, center_out, vector_out = preds
        
        print(f"Batch {batch_idx+1}:")
        print(f" - Image shape:       {imgs.shape}")
        print(f" - Mask Out shape:    {mask_out.shape}")
        print(f" - Center Out shape:  {center_out.shape}")
        print(f" - Vector Out shape:  {vector_out.shape} (dx, dy)")
        
        # Loss Calculation
        targets = (masks, centerlines, vectors)
        total_loss, lm, lc, lv = criterion(preds, targets)
        
        # Backward
        total_loss.backward()
        optimizer.step()
        
        print(f" - Loss Total: {total_loss.item():.4f} (Mask: {lm.item():.4f} | Center: {lc.item():.4f} | Vector: {lv.item():.4f})")
        print(" -> Forward & Backward executat cu succes!\n")
        break
        
    print("===================================================================")
    print("PROTOTIP FINALIZAT. Gata de integrare în scriptul tău de Training!")
    print("===================================================================")