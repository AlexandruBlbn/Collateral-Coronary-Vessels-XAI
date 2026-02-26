import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import timm

# Importuri specifice proiectului
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import ArcadeDataset
from data.transformWrapper import TransformsWrapper
from engine.train import loader

# def get_val_loader(img_size=256, batch_size=4):
#     """Creează un loader de validare cu câteva imagini."""
#     base = ArcadeDataset(split='validation', mode='syntax', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
#     ds = TransformsWrapper(base, input_size=img_size, mode='validation')
#     return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

# class BackboneExtractor(torch.nn.Module):
#     """Încapsulăm crearea modelului pentru a scoate strict lista de features."""
#     def __init__(self, encoder_name='resnet50'):
#         super().__init__()
#         self.backbone = timm.create_model(
#             encoder_name, 
#             pretrained=False, 
#             in_chans=1, 
#             features_only=True,
#         )
#         self.channels_list = self.backbone.feature_info.channels()
        
#     def forward(self, x):
#         features = list(self.backbone(x))
#         for i in range(len(features)):
#             # Reparăm permute-ul fix ca în codul de antrenare
#             if features[i].dim() == 4 and features[i].shape[-1] == self.channels_list[i]:
#                 features[i] = features[i].permute(0, 3, 1, 2).contiguous()
#         return features

# def extract_and_visualize(checkpoint_path, save_path="feature_maps.png"):
#     print(f"Încărcăm modelul din: {checkpoint_path}")
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 1. Creăm și încărcăm backbone-ul
#     model = BackboneExtractor(encoder_name='resnet50').to(device)
    
#     if os.path.exists(checkpoint_path):
#         state_dict = torch.load(checkpoint_path, map_location=device)
#         model.backbone.load_state_dict(state_dict)
#         print("-> Greutăți încărcate cu succes!")
#     else:
#         print(f"Eroare: Nu găsesc fișierul {checkpoint_path}")
#         return

#     model.eval()

#     # 2. Luăm un batch de imagini
#     val_loader = get_val_loader(img_size=256, batch_size=3) # Luăm 3 imagini pentru a vedea varietate
#     images, masks = next(iter(val_loader))
#     images = images.to(device)

#     # 3. Extragem Feature Maps
#     with torch.no_grad():
#         features_list = model(images)

#     # Parametri grafică
#     num_images = images.shape[0]
#     num_stages = len(features_list)
#     fig, axes = plt.subplots(num_images, num_stages + 2, figsize=(3 * (num_stages + 2), 3 * num_images))
    
#     # Dacă avem o singură imagine, axes e 1D. Îl facem 2D forțat pentru cod curat
#     if num_images == 1:
#         axes = np.expand_dims(axes, axis=0)

#     for i in range(num_images):
#         img_np = images[i, 0].cpu().numpy()
#         mask_np = masks[i, 0].cpu().numpy()
        
#         # Denormalizare imagine (Presupunând mean 0.5, std 0.5)
#         img_np = img_np * 0.5 + 0.5
        
#         # Originală
#         axes[i, 0].imshow(img_np, cmap='gray')
#         axes[i, 0].set_title(f"IMG {i+1} Original")
#         axes[i, 0].axis('off')
        
#         # Cele 4 Hărți de Atenție (Stages)
#         for stage_idx, feat in enumerate(features_list):
#             # feat = [B, C, H, W]
#             single_feat = feat[i:i+1] # Luăm feat-ul pentru imaginea 'i'
            
#             # Calculăm "energia" (media activărilor pe axa canalelor)
#             activation = torch.mean(torch.abs(single_feat), dim=1, keepdim=True)
            
#             # Interpolăm pentru afișare corectă
#             act_resized = F.interpolate(activation, size=(256, 256), mode='bilinear', align_corners=False)
#             act_np = act_resized[0, 0].cpu().numpy()
            
#             # Normalizare Min-Max pe heatmap
#             act_np = (act_np - act_np.min()) / (act_np.max() - act_np.min() + 1e-8)
            
#             # Afișare cu colormap-ul Jet
#             axes[i, stage_idx + 1].imshow(img_np, cmap='gray') # Fundalul
#             axes[i, stage_idx + 1].imshow(act_np, cmap='jet', alpha=0.5) # Heatmap suprapus
#             axes[i, stage_idx + 1].set_title(f"Stage {stage_idx+1}")
#             axes[i, stage_idx + 1].axis('off')
            
#         # Ground Truth
#         axes[i, -1].imshow(mask_np, cmap='gray')
#         axes[i, -1].set_title("Masca GT")
#         axes[i, -1].axis('off')

#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches='tight', dpi=150)
#     print(f"==> Vizualizările au fost salvate în: {save_path}")
#     plt.close()

# if __name__ == "__main__":
#     # Schimbă calea dacă este necesar
#     CHECKPOINT = "checkpoints/resnet50_lejepa_strict_probe/best_backbone.pth"
#     extract_and_visualize(CHECKPOINT, save_path="Feature_Activations.png")


