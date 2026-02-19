import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import timm
from torch.utils.data import DataLoader
from captum.attr import Saliency

# --- Importuri specifice proiectului ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataloader import ArcadeDataset
from deeplab import TransformsWrapper

# =========================================================================
# 1. LOGICĂ ÎNCĂRCARE MODEL (REPARARE CHEI STAGES_ -> STAGES.)
# =========================================================================

def load_repaired_model(ckpt_path, device='cuda'):
    weights = torch.load(ckpt_path, map_location='cpu')
    
    if 'model_state_dict' in weights:
        weights = weights['model_state_dict']

    new_weights = {}
    for k, v in weights.items():
        if k.startswith("module."): k = k.replace("module.", "")
        if k.startswith("backbone."): k = k.replace("backbone.", "")
        
        # REPARARE: timm folosește punct (stages.0), nu underscore (stages_0)
        if "stages_" in k:
            k = k.replace("stages_", "stages.")
        
        new_weights[k] = v

    model = timm.create_model('coatnet_1_rw_224', pretrained=False, in_chans=1, num_classes=0, global_pool='')
    
    # SCHIMBARE: strict=False pentru a ignora norm.weight/bias lipsă
    msg = model.load_state_dict(new_weights, strict=False)
    
    # VERIFICARE: Ne asigurăm că NU avem "stages" în lista de chei lipsă
    missing_stages = [key for key in msg.missing_keys if "stages" in key]
    
    if len(missing_stages) > 0:
        print(f"ATENȚIE: Următoarele straturi critice lipsesc: {missing_stages[:5]}...")
        raise RuntimeError("Nu s-au putut încărca etapele principale ale modelului!")
    else:
        print("Backbone-ul s-a încărcat corect (lipsesc doar norm.weight/bias, ceea ce este OK).")
    
    return model.to(device)

# =========================================================================
# 2. FUNCȚIE VIZUALIZARE SALIENCY OVERLAY
# =========================================================================

def visualize_saliency_overlay_threshold(model, val_loader, device='cuda', save_path='saliency_overlay.png', threshold=0.3, overlay_alpha=0.6):
    # Extrage un batch pentru vizualizare
    images, _ = next(iter(val_loader))
    num_images = min(4, images.size(0))
    
    img_batch = images[:num_images].to(device)
    img_batch.requires_grad = True
    
    model.eval()

    # Wrapper pentru encoder (Saliency are nevoie de un scalar la ieșire)
    def forward_enc(img):
        features = model(img)
        # Global Average Pooling manual
        pooled = torch.mean(features, dim=(2, 3)) 
        return pooled.sum(dim=1)

    # Inițializare Captum Saliency
    slc = Saliency(forward_enc)
    
    # Calculăm gradienții (atribuțiile)
    attributions = slc.attribute(img_batch, target=None)
    attributions_np = torch.abs(attributions).cpu().detach().numpy()
    
    # Plotting
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1: axes = np.expand_dims(axes, axis=0)

    for i in range(num_images):
        # 1. Imaginea originală (denormalizare x*0.5 + 0.5)
        orig_img = img_batch[i, 0].detach().cpu().numpy() * 0.5 + 0.5
        
        # 2. Harta Saliency - Normalizare și Thresholding
        sal_map = attributions_np[i, 0]
        # Normalizare [0, 1]
        sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8) * 2
        
        # Aplicăm threshold (valorile mici devin 0)
        sal_map_thresholded = sal_map.copy()
        sal_map_thresholded[sal_map < threshold] = 0
        
        # --- Coloana A: Original ---
        axes[i, 0].imshow(orig_img, cmap='gray')
        axes[i, 0].axis('off')
        
        # --- Coloana B: Saliency Map ---
        axes[i, 1].imshow(sal_map_thresholded, cmap='hot', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        
        # --- Coloana C: Overlay (Original + Saliency) ---
        axes[i, 2].imshow(orig_img, cmap='gray')
        # Heatmap-ul este suprapus folosind parametrul alpha pentru transparență
        axes[i, 2].imshow(sal_map_thresholded, cmap='hot', alpha=overlay_alpha, vmin=0, vmax=1)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Imaginea salvată cu succes: {save_path}")
    plt.close()

# =========================================================================
# 3. EXECUȚIE SCRIPT
# =========================================================================

if __name__ == "__main__":
    # Configurații
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_PATH = "/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/LeJepa_coatnet_detach/best_backbone.pth"
    IMG_SIZE = 224
    BATCH_SIZE = 16
    JSON_PATH = 'data/ARCADE/processed/dataset.json'

    # 1. Încarcă modelul reparat
    model = load_repaired_model(CKPT_PATH, device=DEVICE)

    # 2. Setup Dataloader
    val_base = ArcadeDataset(
        split='validation', 
        mode='syntax', 
        transform=None, 
        root_dir='.', 
        json_path=JSON_PATH
    )
    val_ds = TransformsWrapper(val_base, input_size=IMG_SIZE, mode='validation')
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Generează vizualizarea
    # Threshold 0.3 elimină zgomotul slab, Alpha 0.6 permite vederea vaselor prin heatmap
    visualize_saliency_overlay_threshold(
        model, 
        val_loader, 
        device=DEVICE, 
        save_path='saliency_overlay_result.png', 
        threshold=0, 
        overlay_alpha=0.6
    )