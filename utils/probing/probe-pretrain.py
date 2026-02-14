import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torchvision.transforms.functional as TF

# --- SETUP CĂI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from zoo.models import SegmentatorCoronare, SegmentatorCoronarePlusPlus # Importă modelul tău
from data.dataloader import ArcadeDataset

# --- CONFIGURARE ---
CHECKPOINT_PATH = "/workspace/Collateral-Coronary-Vessels-XAI/checkpoints/segmentation/swinv2_tiny_simmim_FROZEN/best.pth" # <--- PUNE CALEA CĂTRE MODELUL TĂU AICI
BACKBONE_TYPE = "swinv2_tiny_window16_256"             # <--- BACKBONE-UL FOLOSIT
IMAGE_INDEX = 6                                       # Indexul imaginii din validare pe care vrei s-o inspectezi
INPUT_SIZE = 256                                       # Sau 512, cum ai antrenat

# --- CLASA PENTRU HOOKS ---
class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.activations = {}
        for name, layer in target_layers.items():
            layer.register_forward_hook(self._get_hook(name))

    def _get_hook(self, name):
        def hook(model, input, output):
            self.activations[name] = output
        return hook

def normalize_heatmap(heatmap):
    """Normalizează o hartă de trăsături între 0 și 1 pentru vizualizare"""
    heatmap = heatmap.float()
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-7)
    return heatmap.cpu().numpy()

def analyze_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Încărcare Model
    print(f"🔍 Încărcare model din: {CHECKPOINT_PATH}")
    # NOTĂ: Schimbă clasa aici dacă folosești DeepLab sau altceva
    model = SegmentatorCoronare(backbone=BACKBONE_TYPE, in_channels=1, num_classes=1, input_size=INPUT_SIZE)
    
    # Încărcare ponderi (cu ignorarea erorilor mici de strictness dacă e cazul)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. Setup Hooks (Unde "ascultăm" în rețea)
    # Putem accesa straturile prin nume.
    # Pentru SegmentatorCoronare: 
    #   - backbone: model.backbone
    #   - decoder final: model.up1 (ultimul bloc de upsample înainte de conv finală)
    
    layers_to_hook = {
        'Backbone (Encoder)': model.backbone,
        'Decoder (Final Layer)': model.up1
    }
    extractor = FeatureExtractor(model, layers_to_hook)

    # 3. Încărcare Imagine
    # Folosim dataset-ul pentru a lua imaginea preprocesată corect
    json_path = os.path.join(project_root, 'data/ARCADE/processed/dataset.json')
    ds = ArcadeDataset(json_path, split='validation', mode='syntax')
    
    img_pil, label_path = ds[IMAGE_INDEX]
    full_label_path = os.path.join(project_root, label_path)
    mask_pil = Image.open(full_label_path).convert('L')

    # Transformare manuală (resize + normalize)
    img_tensor = TF.resize(img_pil, (INPUT_SIZE, INPUT_SIZE))
    img_tensor = TF.to_tensor(img_tensor)
    img_tensor = TF.normalize(img_tensor, mean=[0.5], std=[0.5])
    img_tensor = img_tensor.unsqueeze(0).to(device) # Batch dimension

    mask_tensor = TF.resize(mask_pil, (INPUT_SIZE, INPUT_SIZE), interpolation=TF.InterpolationMode.NEAREST)
    mask_np = np.array(mask_tensor) > 0 # Ground Truth binary

    # 4. Inferență
    print("🚀 Rulare inferență...")
    with torch.no_grad():
        output = model(img_tensor)
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = pred_prob > 0.5

    # 5. Procesare Feature Maps
    features = extractor.activations
    
    # --- A. Procesare Backbone Features ---
    # Backbone-ul scoate (1, C, H/32, W/32). Facem media pe canale pentru a vedea "activarea globală"
    bb_feats = features['Backbone (Encoder)']
    # Media pe axa canalelor (dim 1) -> Harta de intensitate
    bb_heatmap = torch.mean(bb_feats, dim=1).squeeze() 
    bb_heatmap = normalize_heatmap(bb_heatmap)
    # Resize la dimensiunea originală pentru vizualizare
    bb_heatmap = cv2.resize(bb_heatmap, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    # --- B. Procesare Decoder Features ---
    # Decoderul scoate (1, 32, H, W). Facem media pe canale.
    dec_feats = features['Decoder (Final Layer)']
    dec_heatmap = torch.mean(dec_feats, dim=1).squeeze()
    dec_heatmap = normalize_heatmap(dec_heatmap)
    # Resize (de siguranță)
    dec_heatmap = cv2.resize(dec_heatmap, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    # 6. Vizualizare
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Rândul 1: Input și Activări
    img_display = np.array(img_pil.resize((INPUT_SIZE, INPUT_SIZE)))
    
    axes[0, 0].imshow(img_display, cmap='gray')
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')

    # Backbone Heatmap (folosim colormap 'jet' pentru a vedea punctele fierbinți)
    axes[0, 1].imshow(img_display, cmap='gray', alpha=0.5)
    axes[0, 1].imshow(bb_heatmap, cmap='jet', alpha=0.5) 
    axes[0, 1].set_title("Backbone Activation (Unde se uită Encoderul)")
    axes[0, 1].axis('off')

    # Decoder Heatmap
    axes[0, 2].imshow(img_display, cmap='gray', alpha=0.5)
    axes[0, 2].imshow(dec_heatmap, cmap='jet', alpha=0.5)
    axes[0, 2].set_title("Decoder Activation (Ce a reconstruit)")
    axes[0, 2].axis('off')

    # Rândul 2: Predicții și Erori
    axes[1, 0].imshow(mask_np, cmap='gray')
    axes[1, 0].set_title("Ground Truth (Ce trebuia să vadă)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_prob, cmap='magma')
    axes[1, 1].set_title("Model Prediction (Probabilities)")
    axes[1, 1].axis('off')

    # Harta Erorilor: Roșu = False Negative (Vase ratate), Albastru = False Positive (Zgomot)
    error_map = np.zeros((INPUT_SIZE, INPUT_SIZE, 3))
    # Vase ratate (FN) -> Roșu
    error_map[(mask_np == 1) & (pred_mask == 0)] = [1, 0, 0] 
    # Zgomot (FP) -> Albastru
    error_map[(mask_np == 0) & (pred_mask == 1)] = [0, 0.5, 1] 
    
    axes[1, 2].imshow(error_map)
    axes[1, 2].set_title("Erori: Roșu=Ratat, Albastru=Zgomot")
    axes[1, 2].axis('off')

    plt.tight_layout()
    save_path = f"analysis_result_img{IMAGE_INDEX}.png"
    plt.savefig(save_path)
    print(f"✅ Analiză salvată în: {save_path}")
    plt.show()

if __name__ == "__main__":
    analyze_model()