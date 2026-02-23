import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import ArcadeDataset
from data.transformWrapper import TransformsWrapper
from engine.LeJepa_Train import LeJepaModel

def load_model(checkpoint_path, encoder_name='swinv2_tiny_window8_256', proj_dim=256, device='cuda'):
    print(f"Încărcăm modelul din: {checkpoint_path}")
    model = LeJepaModel(encoder_name=encoder_name, proj_dim=proj_dim)
    
    if os.path.exists(checkpoint_path):
        weights = torch.load(checkpoint_path, map_location='cpu')
        # Dacă ai salvat doar backbone-ul, trebuie să îl încarci în model.backbone
        # Dacă ai salvat modelul complet (LeJepaModel), încarci normal
        if 'model_state_dict' in weights:
            model.load_state_dict(weights['model_state_dict'])
        else:
            model.backbone.load_state_dict(weights, strict=False)
        print("Greutăți încărcate cu succes!")
    else:
        print("Avertisment: Nu s-a găsit checkpoint-ul. Se folosesc greutăți random.")
        
    model.to(device)
    model.eval()
    return model

def generate_saliency_map(model, img_tensor, device='cuda'):
    # 1. Ne asigurăm că imaginea necesită gradienți
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_()

    # 2. Forward pass prin backbone
    features, _ = model(img_tensor)
    
    # 3. Preluăm ultima hartă de trăsături (cea mai profundă semantic)
    last_feature_map = features[-1]
    
    # 4. Funcția de "scor" - vrem să maximizăm activarea globală a trăsăturilor
    score = last_feature_map.sum()
    
    # 5. Calculăm gradienții (Backward pass până la imagine)
    model.zero_grad()
    score.backward()
    
    # 6. Extragem gradienții imaginii de intrare
    saliency = img_tensor.grad.data.abs().squeeze().cpu().numpy()
    
    # 7. Normalizare 0 - 1
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency

def save_visualization(original_img, saliency_map, mask, save_path):
    # original_img vine normalizat [-1, 1], îl trecem în [0, 1]
    img_vis = original_img.squeeze().cpu().numpy() * 0.5 + 0.5
    mask_vis = mask.squeeze().cpu().numpy()
    
    # Aplicăm un colormap (JET) peste harta de saliency
    saliency_heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    saliency_heatmap = cv2.cvtColor(saliency_heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Suprapunem harta de saliency peste imaginea originală
    img_rgb = np.stack((img_vis,)*3, axis=-1)
    overlay = 0.5 * img_rgb + 0.5 * saliency_heatmap
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_vis, cmap='gray')
    plt.title('Imagine Originală')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(overlay)
    plt.title('Saliency Map (Backbone)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask_vis, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Salvat vizualizare în: {save_path}")

if __name__ == "__main__":
    # --- CONFIGURARE ---
    IMG_SIZE = 256
    CHECKPOINT_PATH = os.path.join(project_root, "checkpoints", "Swin_v2_tiny_w8_256_lejepa_linear_probe", "best_backbone.pth")
    SAVE_DIR = os.path.join(project_root, "saliency_results")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- PREGĂTIRE DATE ---
    dataset = ArcadeDataset(json_path=os.path.join(project_root, 'data/ARCADE/processed/dataset.json'), split='validation', mode='syntax', root_dir=project_root)
    wrapper = TransformsWrapper(dataset, input_size=IMG_SIZE, mode='validation')
    dataloader = DataLoader(wrapper, batch_size=1, shuffle=True)
    
    # --- ÎNCĂRCARE MODEL ---
    model = load_model(CHECKPOINT_PATH)
    
    # --- GENERARE ---
    num_samples = 5
    iterator = iter(dataloader)
    
    for i in range(num_samples):
        # Despachetăm datele (ignorăm is_syntax)
        batch = next(iterator)
        if len(batch) == 3:
            img, mask, _ = batch
        else:
            img, mask = batch
            
        # Generăm harta
        saliency = generate_saliency_map(model, img)
        
        # Salvăm imaginea
        save_path = os.path.join(SAVE_DIR, f"saliency_sample_{i+1}.png")
        save_visualization(img, saliency, mask, save_path)
        
    print("Generarea hărților Saliency s-a finalizat cu succes!")