import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.cm as cm

# --- SETUP CĂI (PATHS) ---
# Adăugăm root-ul proiectului în path pentru a putea importa din 'zoo'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../')
sys.path.append(project_root)

from zoo.backbones import get_backbone

# --- CONFIGURARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Calea relativă către config (ajustează dacă e diferită)
CONFIG_PATH = os.path.join(project_root, "config/dino_config.yaml")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Eroare: Config file nu există la {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_model_and_transform():
    config = load_config()
    
    backbone_name = config["model"]["backbone"]
    in_channels = config["data"].get("in_channels", 3)
    experiment_name = config.get("experiment_name", f"dino_{backbone_name}")
    
    print(f"--> Loading Backbone: {backbone_name}")
    
    # 1. Inițializare Model (Backbone gol)
    model = get_backbone(model_name=backbone_name, in_channels=in_channels, pretrained=False)
    model.to(DEVICE)
    model.eval()
    
    # 2. Căutare Checkpoint (Best Backbone)
    # Căutăm în mai multe locuri posibile
    possible_paths = [
        os.path.join(project_root, "checkpoints/dino", experiment_name, "last_backbone.pth"),
        f"./checkpoints/dino/{experiment_name}/best_backbone.pth"
    ]
    
    ckpt_path = None
    for p in possible_paths:
        if os.path.exists(p):
            ckpt_path = p
            break
            
    if ckpt_path:
        print(f"--> Loading weights from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=DEVICE)
        
        # Curățare prefixe (student.backbone, encoder, etc.)
        clean_state = {}
        for k, v in state.items():
            k = k.replace("backbone.", "").replace("encoder.", "").replace("student_backbone.", "").replace("student.", "")
            clean_state[k] = v
            
        msg = model.load_state_dict(clean_state, strict=False)
        print(f"--> Weights loaded. Missing keys (head keys are normal): {len(msg.missing_keys)}")
    else:
        print(f"⚠ WARNING: Checkpoint not found! Using RANDOM weights. (Path cautat: {possible_paths[0]})")

    # 3. Transformare
    # Resize la 256x256 pentru viteză și consistență cu antrenamentul
    input_size = config["model"].get("input_size", 256)
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return model, transform, in_channels

# --- GLOBAL CACHE ---
# Încărcăm modelul o singură dată la start
MODEL, TRANSFORM, IN_CHANNELS = get_model_and_transform()
CURRENT_FEATURES = None # Aici stocăm feature-urile imaginii curente
CURRENT_SHAPE = None    # Dimensiunile originale ale imaginii

def extract_features(input_img_numpy):
    """
    Pasul 1: Preia imaginea, o trece prin backbone și salvează feature map-ul.
    """
    global CURRENT_FEATURES, CURRENT_SHAPE
    
    if input_img_numpy is None:
        return "Te rog încarcă o imagine."
    
    # input_img_numpy vine de la Gradio ca (H, W, 3) uint8
    pil_img = Image.fromarray(input_img_numpy)
    
    # Convertim la Grayscale sau RGB în funcție de model
    if IN_CHANNELS == 1:
        pil_img = pil_img.convert('L')
    else:
        pil_img = pil_img.convert('RGB')
        
    CURRENT_SHAPE = pil_img.size # (W, H)
    
    # Preprocesare
    img_tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE) # (1, C, H, W)
    
    with torch.no_grad():
        # Extragem features. Rezultat tipic: (1, 384, 16, 16) pentru ViT-Small
        features = MODEL(img_tensor)
        
        # Dacă output-ul e (B, N, D) (ViT standard), facem reshape la grid
        if len(features.shape) == 3:
            B, N, D = features.shape
            # Presupunem grid pătrat (ex: 256/16 = 16x16 patch-uri -> 256 tokens)
            H_grid = W_grid = int(np.sqrt(N)) 
            # (B, N, D) -> (B, D, H, W)
            features = features.transpose(1, 2).reshape(B, D, H_grid, W_grid)
            
    # Normalizare L2 (Crucial pentru Cosine Similarity)
    # După normalizare, Dot Product == Cosine Similarity
    features = F.normalize(features, dim=1, p=2)
    
    CURRENT_FEATURES = features
    return f"✅ Features extrase! Shape: {tuple(features.shape)}. Acum dă click pe imagine."

def query_point(evt: gr.SelectData, input_image):
    """
    Pasul 2: Când userul dă click, comparăm vectorul de sub cursor cu toți ceilalți.
    """
    global CURRENT_FEATURES
    
    if CURRENT_FEATURES is None:
        return input_image # Nu s-au extras features încă
    
    # Coordonatele click-ului (în pixeli pe imaginea originală)
    x, y = evt.index[0], evt.index[1]
    
    # Dimensiunile imaginii afișate
    H_img, W_img = input_image.shape[0], input_image.shape[1]
    
    # Dimensiunile Grid-ului de Features (ex: 16x16)
    B, Dim, H_feat, W_feat = CURRENT_FEATURES.shape
    
    # Mapăm coordonatele din Imagine -> Grid
    grid_x = int(x / W_img * W_feat)
    grid_y = int(y / H_img * H_feat)
    
    # Clamp pentru a nu ieși din matrice
    grid_x = min(max(grid_x, 0), W_feat - 1)
    grid_y = min(max(grid_y, 0), H_feat - 1)
    
    # --- PROBING MAGIC ---
    
    # 1. Luăm "Semnătura" punctului selectat (Vectorul Query)
    query_vector = CURRENT_FEATURES[:, :, grid_y, grid_x].unsqueeze(-1).unsqueeze(-1) # (1, D, 1, 1)
    
    # 2. Produs scalar cu toată harta (Cosine Similarity)
    # Rezultat: O hartă de căldură (1, H_feat, W_feat) cu valori între -1 și 1
    similarity_map = torch.sum(CURRENT_FEATURES * query_vector, dim=1) 
    
    # 3. Post-procesare pentru vizualizare clară
    # Tăiem valorile negative (ce nu seamănă deloc)
    heatmap = similarity_map.clamp(min=0)
    
    # Ridicăm la putere pentru a evidenția doar zonele FOARTE similare
    # DINO învață reprezentări foarte ascuțite, așa că ^3 sau ^4 arată bine
    heatmap = heatmap ** 3 
    
    # 4. Upscaling înapoi la rezoluția imaginii
    heatmap = F.interpolate(heatmap.unsqueeze(0), size=(H_img, W_img), mode='bicubic', align_corners=False)
    heatmap = heatmap.squeeze().cpu().numpy()
    
    # Normalizare 0-1 pentru plotare
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 5. Colorare (Jet: Albastru=0, Roșu=1)
    heatmap_colored = cm.jet(heatmap)[:, :, :3] # RGB
    
    # 6. Overlay
    if len(input_image.shape) == 2:
        input_image = np.stack([input_image]*3, axis=-1)
        
    img_float = input_image.astype(float) / 255.0
    
    # Mix: 30% Imagine Originală + 70% Heatmap
    overlay = 0.3 * img_float + 0.7 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    return (overlay * 255).astype(np.uint8)

# --- INTERFAȚA GRADIO ---
with gr.Blocks(title="DINO Anatomical Probing", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 DINO Dense Feature Probing
        **Cum funcționează:**
        1. Modelul transformă imaginea într-o grilă de vectori (semnături semantice).
        2. Când dai click, luăm vectorul de sub cursor.
        3. Colorăm cu **Roșu** zonele care au o semnătură similară.
        
        **Ce să cauți:** Dacă dai click pe un vas de sânge, ar trebui să se aprindă DOAR alte vase de sânge.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="1. Încarcă Angiografia", type="numpy")
            btn_extract = gr.Button("2. Extract Features (Rulează Backbone)", variant="primary")
            status_msg = gr.Textbox(label="Status", value="Aștept imagine...", interactive=False)
        
        with gr.Column():
            output_img = gr.Image(label="3. Rezultat Probing (Click pe imaginea din stânga!)", interactive=False)
            
    # Legături Evenimente
    btn_extract.click(extract_features, inputs=input_img, outputs=status_msg)
    input_img.select(query_point, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    print(f"Pornire server pe port 7860...")
    # share=True face un link public temporar (util dacă ești pe un server cloud)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)