import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm

# Setup path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from zoo.backbones import get_backbone
from data.dataloader import ArcadeDataset

def compute_pca_visualizations(backbone, dataloader, device, save_dir, n_images=5):
    backbone.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Luăm un batch
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)
    
    print(f"Generating PCA for {n_images} images...")
    
    with torch.no_grad():
        # 1. Extragem Feature-urile: (B, C, H, W) -> ex: (B, 384, 16, 16)
        features = backbone(imgs[:n_images])
    
    # Procesăm fiecare imagine individual
    for i in range(n_images):
        feat = features[i] # (C, H_grid, W_grid)
        C, H_grid, W_grid = feat.shape
        
        # 2. Reshape pentru PCA: (N_pixels, C)
        feat_flat = feat.flatten(1).transpose(0, 1).cpu().numpy() # (H*W, C)
        
        # 3. Aplicăm PCA pentru a reduce la 3 dimensiuni (ca să facem o imagine RGB)
        # Dacă modelul a învățat bine, prima componentă va separa fundalul de obiecte
        pca = PCA(n_components=3)
        pca_feat = pca.fit_transform(feat_flat) # (H*W, 3)
        
        # 4. Normalizare Min-Max la [0, 1] pentru afișare RGB
        pca_feat_min = pca_feat.min(axis=0)
        pca_feat_max = pca_feat.max(axis=0)
        pca_feat = (pca_feat - pca_feat_min) / (pca_feat_max - pca_feat_min)
        
        # 5. Reshape înapoi la grid 2D: (H_grid, W_grid, 3)
        pca_img = pca_feat.reshape(H_grid, W_grid, 3)
        
        # 6. Pregătire imagine originală
        orig_img = imgs[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        orig_img = np.clip(orig_img, 0, 1)
        
        # 7. Plotare
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # A. Imagine Originală
        axes[0].imshow(orig_img, cmap='gray' if orig_img.shape[2]==1 else None)
        axes[0].set_title("Original Input")
        axes[0].axis('off')
        
        # B. PCA Principal (RGB) - Relații Semantice
        # Facem upsampling (resize) doar pentru vizualizare fină, 
        # dar culorile sunt date de feature-urile brute
        pca_upscaled = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_upscaled = pca_upscaled.resize((256, 256), resample=Image.NEAREST)
        axes[1].imshow(pca_upscaled)
        axes[1].set_title("PCA Features (Semantic Colors)")
        axes[1].axis('off')

        # C. Doar prima componentă principală (PC1) - De obicei Foregound vs Background
        pc1 = pca_feat[:, 0].reshape(H_grid, W_grid)
        axes[2].imshow(pc1, cmap='viridis')
        axes[2].set_title("PC1 (Dominant Feature)")
        axes[2].axis('off')
        
        # D. A doua componentă (PC2) - Detalii secundare
        pc2 = pca_feat[:, 1].reshape(H_grid, W_grid)
        axes[3].imshow(pc2, cmap='magma')
        axes[3].set_title("PC2 (Secondary Feature)")
        axes[3].axis('off')
        
        save_path = os.path.join(save_dir, f"pca_vis_{i}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

def main():
    # --- Config ---
    config_path = "config/lejepa_config.yaml"
    if not os.path.exists(config_path): config_path = "../../config/lejepa_config.yaml"
    with open(config_path, "r") as f: config = yaml.safe_load(f)
        
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    backbone_name = config["model"]["backbone"]
    
    # Load Backbone
    print(f"Loading backbone: {backbone_name}")
    checkpoint_path = f"./checkpoints/lejepa/lejepa_pretrain_{backbone_name}/best_backbone.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"../../checkpoints/lejepa/lejepa_pretrain_{backbone_name}/best_backbone.pth"
        
    backbone = get_backbone(model_name=backbone_name, in_channels=1, pretrained=False)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location='cpu')
        clean_state = {k.replace("backbone.", "").replace("encoder.", ""): v for k, v in state.items()}
        backbone.load_state_dict(clean_state, strict=False)
        print("Backbone weights loaded.")
    else:
        print("WARNING: Using random weights. Colors will be random noise.")
        
    backbone.to(device)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # Folosim Validation set pentru imagini noi
    json_path = "/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json"
    ds = ArcadeDataset(json_path, split='validation', transform=transform, mode='syntax', root_dir="/workspace/Collateral-Coronary-Vessels-XAI")
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    
    compute_pca_visualizations(backbone, loader, device, save_dir=f"./vis_pca/{backbone_name}")

if __name__ == "__main__":
    main()