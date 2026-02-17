import json
import os
import cv2
import numpy as np
import re

# --- GENERARE PALETĂ DINAMICĂ PENTRU N CLASE ---
def generate_palette(n_classes=30):
    """
    Generează o paletă de culori distincte.
    Index 0 este forțat la Negru (Background).
    Restul sunt generate folosind variații de Hue în spațiul HSV.
    """
    palette = np.zeros((n_classes, 1, 3), dtype=np.uint8)
    
    # Setăm culorile pentru clasele 1..N
    # Folosim un pas de aur (Golden Angle) sau o diviziune simplă pentru a separa nuanțele
    for i in range(1, n_classes):
        hue = int((i * 180) / n_classes)  # OpenCV Hue range is 0-179
        saturation = 255
        value = 255
        
        # Putem varia saturația pentru clasele pare/impare pentru extra contrast
        if i % 2 == 0:
            saturation = 200
            
        palette[i, 0, :] = [hue, saturation, value]
    
    # Convertim HSV -> BGR (pentru că OpenCV folosește BGR)
    bgr_palette = cv2.cvtColor(palette, cv2.COLOR_HSV2BGR)
    
    # Reshape la (N, 3) și forțăm index 0 la negru absolut
    bgr_palette = bgr_palette.reshape(-1, 3)
    bgr_palette[0] = [0, 0, 0] 
    
    return bgr_palette

# Generăm paleta globală (punem 30 ca să fim siguri că acoperim cele 25 de clase)
PALETTE_BGR = generate_palette(n_classes=30)

def natural_sort_key(s):
    """Sortare naturală (1, 2, 10, nu 1, 10, 2)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def save_colored_mask(grayscale_mask, output_path):
    """
    Aplică paleta generată peste masca cu indecși.
    """
    # Ne asigurăm că nu depășim numărul de culori generate
    max_idx = len(PALETTE_BGR) - 1
    mask_clamped = np.clip(grayscale_mask, 0, max_idx)
    
    # Aplicăm culorile (Broadcasting)
    colored_mask = PALETTE_BGR[mask_clamped]
    
    cv2.imwrite(output_path, colored_mask)

def process_annotations():
    # --- CĂI DE BAZĂ (Verifică-le să fie corecte pe mașina ta!) ---
    base_unprocessed = '/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/Unprocessed/arcade/syntax'
    output_mask_dir_main = '/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/syntax/label2'
    output_mask_dir_viz = os.path.join(output_mask_dir_main, 'visualization_check')
    dataset_json_path = '/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json'

    # Creare directoare
    os.makedirs(output_mask_dir_main, exist_ok=True)
    os.makedirs(output_mask_dir_viz, exist_ok=True)

    if not os.path.exists(dataset_json_path):
        print(f"❌ Critical: {dataset_json_path} lipsește!")
        return

    with open(dataset_json_path, 'r') as f:
        dataset_data = json.load(f)

    # Configurația split-urilor
    split_configs = [
        ('train', 'train', 'train'),
        ('val',   'val',   'validation'),
        ('test',  'test',  'test')
    ]

    total_updated = 0

    print(f"🎨 Paletă generată pentru {len(PALETTE_BGR)} clase (0=Negru).")

    for folder_name, filename_stem, target_key in split_configs:
        input_json_path = os.path.join(base_unprocessed, folder_name, 'annotations', f'{filename_stem}.json')
        
        if not os.path.exists(input_json_path):
            print(f"⚠️  Skipping {target_key}: Nu am găsit {input_json_path}")
            continue

        print(f"\n🔄 Processing {target_key}...")

        # 1. Target IDs (din dataset.json)
        if target_key not in dataset_data or 'syntax' not in dataset_data[target_key]:
            print(f"   ❌ Cheia '{target_key}' lipsă în dataset. Skip.")
            continue

        target_ids = list(dataset_data[target_key]['syntax'].keys())
        target_ids.sort(key=natural_sort_key)
        
        # 2. Source Images (din COCO json)
        with open(input_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Sortăm sursele ca să se alinieze cu target-ul
        source_images = sorted(coco_data['images'], key=lambda x: natural_sort_key(x['file_name']))
        
        # Mapare rapidă adnotări
        img_to_anns = {img['id']: [] for img in coco_data['images']}
        if 'annotations' in coco_data:
            for ann in coco_data['annotations']:
                img_to_anns[ann['image_id']].append(ann)

        count = 0
        # 3. Procesare sincronizată
        for source_img, target_id_key in zip(source_images, target_ids):
            
            # --- Generare Mască Grayscale (0, 1, ..., 25) ---
            mask_gray = np.zeros((source_img['height'], source_img['width']), dtype=np.uint8)
            anns = img_to_anns.get(source_img['id'], [])
            
            found_classes = set()
            for ann in anns:
                category_id = int(ann['category_id'])
                found_classes.add(category_id)
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(mask_gray, [poly], color=category_id)

            # A. Salvare pentru Antrenare (Index png)
            mask_filename = f"{target_id_key}.png"
            output_path_gray = os.path.join(output_mask_dir_main, mask_filename)
            cv2.imwrite(output_path_gray, mask_gray)

            # B. Salvare pentru Vizualizare (Color RGB)
            # Salvăm doar dacă există clase (ca să nu umplem folderul cu imagini negre degeaba) sau salvăm tot pt verificare
            viz_filename = f"{target_id_key}_viz.png"
            output_path_viz = os.path.join(output_mask_dir_viz, viz_filename)
            save_colored_mask(mask_gray, output_path_viz)

            # Update JSON
            label2_path = f"data/ARCADE/processed/syntax/label2/{mask_filename}"
            dataset_data[target_key]['syntax'][target_id_key]["label2"] = label2_path
            
            count += 1
            total_updated += 1

        print(f"   ✅ Gata {target_key}: {count} imagini.")

    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_data, f, indent=4)

    print(f"\n🎉 SUCCESS! Total: {total_updated}")
    print(f"👀 Vezi folderul colorat aici: {output_mask_dir_viz}")

if __name__ == "__main__":
    process_annotations()