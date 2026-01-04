import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.filters import frangi, gaussian, threshold_otsu
from skimage import exposure, morphology, measure
from skimage.measure import label, regionprops
from skimage.morphology import disk, reconstruction, white_tophat, remove_small_objects, binary_erosion
from scipy import ndimage as ndi
from skimage.io import imread, imsave
import sys
import os
import glob

# --- IMPORTS ---
sys.path.append(os.getcwd()) 
from data.ARCADE.dataloader import ARCADEDataset

def _remove_border_crop(img, threshold=0.08):
    """Pasul 1: Crop brut al benzilor negre exterioare."""
    row_mean = img.mean(axis=1)
    col_mean = img.mean(axis=0)
    
    valid_rows = np.where(row_mean > threshold)[0]
    valid_cols = np.where(col_mean > threshold)[0]
    
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return img, (0, img.shape[0], 0, img.shape[1])
    
    y_min, y_max = valid_rows[0], valid_rows[-1] + 1
    x_min, x_max = valid_cols[0], valid_cols[-1] + 1
    
    # Marjă de siguranță
    pad = 5
    y_min = max(0, y_min + pad); y_max = min(img.shape[0], y_max - pad)
    x_min = max(0, x_min + pad); x_max = min(img.shape[1], x_max - pad)
    
    return img[y_min:y_max, x_min:x_max], (y_min, y_max, x_min, x_max)

def _create_fov_mask(img, erosion=20):
    """
    Pasul 2: Mască FOV (Field of View).
    Detectează zona activă și o 'erodează' pentru a evita artefactele de la marginea colimatorului.
    """
    # Tot ce nu e negru absolut
    mask = img > 0.05 
    # Umplem găurile (inima)
    mask = ndi.binary_fill_holes(mask)
    # Erodăm marginile agresiv (20px) pentru a tăia "linia roșie"
    mask = binary_erosion(mask, disk(erosion))
    return mask

def _keep_largest_vessel_tree(mask):
    """Păstrează doar structura vasculară principală conectată."""
    mask = remove_small_objects(mask, min_size=50)
    labels = label(mask)
    if labels.max() == 0: return mask
    
    regions = regionprops(labels)
    regions.sort(key=lambda x: x.area, reverse=True)
    
    output = np.zeros_like(mask, dtype=bool)
    if len(regions) > 0:
        output[labels == regions[0].label] = True # Componenta principală
        
    # Păstrăm și a 2-a componentă dacă e mare (vas rupt)
    for i in range(1, min(len(regions), 3)):
        if regions[i].area > regions[0].area * 0.15:
            output[labels == regions[i].label] = True
            
    return output

def _filter_linear_artifacts(mask):
    """Elimină cateterele (linii drepte lungi)."""
    labels = label(mask)
    for region in regionprops(labels):
        if region.eccentricity > 0.990 and region.area > 100:
            if region.minor_axis_length < 6:
                mask[labels == region.label] = False
    return mask

def _remove_border_artifacts(mask, border_thickness=10, area_threshold=500):
    """
    Elimină componentele detectate greșit pe marginile imaginii.
    
    Args:
        mask: Masca binară de segmentare
        border_thickness: Grosimea zonei de margine (în pixeli)
        area_threshold: Componente mai mici de această valoare din zona de margine sunt eliminate
    
    Returns:
        Masca curățată
    """
    if mask.sum() == 0:
        return mask
    
    h, w = mask.shape
    labels = label(mask)
    if labels.max() == 0:
        return mask
        
    regions = regionprops(labels)
    output_mask = mask.copy()
    
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        
        # Verifică dacă componenta atinge marginea
        touches_border = (minr < border_thickness or 
                         minc < border_thickness or 
                         maxr > h - border_thickness or 
                         maxc > w - border_thickness)
        
        if touches_border:
            # Calculează procentul din componentă care este în zona de margine
            border_mask = np.zeros_like(mask, dtype=bool)
            border_mask[:border_thickness, :] = True
            border_mask[-border_thickness:, :] = True
            border_mask[:, :border_thickness] = True
            border_mask[:, -border_thickness:] = True
            
            component_mask = (labels == region.label)
            border_pixels = (component_mask & border_mask).sum()
            total_pixels = region.area
            border_ratio = border_pixels / total_pixels if total_pixels > 0 else 0
            
            # Elimină componenta dacă:
            # 1. Este mică și peste 50% din ea este pe margine
            # 2. Este pe margine și foarte alungită (probabil border artifact)
            if (region.area < area_threshold and border_ratio > 0.5) or \
               (border_ratio > 0.7 and region.eccentricity > 0.95):
                output_mask[labels == region.label] = False
    
    return output_mask

def get_binary_segmentation(img):
    """
    Returnează:
      - final_mask: Masca binară curată (True/False)
      - img_cropped: Imaginea originală tăiată (pentru vizualizare/salvare)
    """
    # 0. Asigurare 2D
    if img.ndim == 3: img = img.squeeze()
    
    # 1. Crop margini negre
    img_cropped, coords = _remove_border_crop(img)
    
    # 2. Generare Mască Siguranță (FOV)
    # Asta previne detectarea bordurii circulare
    fov_mask = _create_fov_mask(img_cropped, erosion=20)

    # 3. Pre-procesare (Top-Hat)
    img_inv = 1.0 - img_cropped
    img_tophat = white_tophat(img_inv, disk(25))
    img_clahe = exposure.equalize_adapthist(img_tophat, clip_limit=0.02, kernel_size=(32, 32))
    img_smooth = gaussian(img_clahe, sigma=1.0)

    # 4. Frangi & Segmentare Iterativă
    scales_config = [
        {'sigma': 3.5, 'thresh_factor': 0.50}, # Trunchi
        {'sigma': 1.0, 'thresh_factor': 0.30}  # Ramuri
    ]
    
    final_mask = np.zeros_like(img_cropped, dtype=bool)
    
    for i, config in enumerate(scales_config):
        # Calcul Frangi
        f_map = frangi(img_smooth, sigmas=[config['sigma']], black_ridges=False, beta=0.5, gamma=15)
        
        # --- FIX CRITIC: Aplicăm FOV Mask imediat ---
        f_map = f_map * fov_mask 
        
        # Normalizare & Threshold
        if f_map.max() > 0: f_map /= f_map.max()
        thresh = threshold_otsu(f_map) if f_map.max() > 0 else 0.1
        candidates = f_map > (thresh * config['thresh_factor'])
        
        # Conectivitate
        layer_mask = np.zeros_like(candidates)
        if i == 0: # Nivel 1: Orice obiect mare
            lbls = label(candidates)
            if lbls.max() > 0:
                regions = regionprops(lbls)
                regions.sort(key=lambda x: x.area, reverse=True)
                for r in regions[:3]: # Top 3 obiecte
                    if r.area > 50: layer_mask[lbls == r.label] = 1
        else: # Nivel 2: Doar ce atinge Nivelul 1
            seed = candidates & morphology.dilation(final_mask, disk(4))
            if seed.sum() > 0:
                layer_mask = reconstruction(seed, candidates, method='dilation').astype(bool)
        
        final_mask = final_mask | layer_mask.astype(bool)

    # 5. Curățare Finală
    final_mask = _filter_linear_artifacts(final_mask)
    final_mask = _remove_border_artifacts(final_mask)
    final_mask = _keep_largest_vessel_tree(final_mask)
    
    return final_mask, img_cropped

# ==========================================
# FUNCȚIE PENTRU CURĂȚAREA MĂȘTILOR EXISTENTE
# ==========================================
def clean_existing_masks(input_dir, output_dir, 
                         border_thickness=10, 
                         area_threshold=500,
                         min_tree_ratio=0.15,
                         visualize=True):
    """
    Procesează măștile binare existente și le curăță de artefacte de margine.
    
    Args:
        input_dir: Directorul cu măștile originale (PNG/NPY)
        output_dir: Directorul unde se salvează măștile curățate
        border_thickness: Grosimea zonei de margine (pixeli)
        area_threshold: Prag pentru componente mici pe margine
        min_tree_ratio: Raportul minim pentru componente secundare (față de principala)
        visualize: Dacă True, salvează și comparația vizuală
    
    Returns:
        Numărul de măști procesate
    """
    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
    
    # Găsește toate fișierele de mască
    mask_files = []
    for ext in ['*.png', '*.npy', '*.jpg', '*.tif']:
        mask_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if len(mask_files) == 0:
        print(f"⚠️  Nu s-au găsit măști în {input_dir}")
        return 0
    
    print(f"🔍 Găsite {len(mask_files)} măști în {input_dir}")
    print(f"📊 Parametri: border={border_thickness}px, area_thresh={area_threshold}, tree_ratio={min_tree_ratio}")
    print("-" * 60)
    
    processed_count = 0
    
    for i, mask_path in enumerate(mask_files):
        filename = os.path.basename(mask_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Încarcă masca
        if mask_path.endswith('.npy'):
            mask_original = np.load(mask_path)
        else:
            mask_original = imread(mask_path)
            # Convertește la binar dacă e necesar
            if mask_original.ndim == 3:
                mask_original = mask_original[:, :, 0]
            if mask_original.dtype == np.uint8:
                mask_original = mask_original > 127
            else:
                mask_original = mask_original > 0.5
        
        # Asigură-te că e boolean
        mask_original = mask_original.astype(bool)
        
        pixels_before = mask_original.sum()
        
        # Aplică filtrele de curățare
        mask_clean = mask_original.copy()
        
        # 1. Elimină artefacte liniare
        mask_clean = _filter_linear_artifacts(mask_clean)
        
        # 2. Elimină componentele de pe margini
        mask_clean_temp = _remove_border_artifacts_custom(
            mask_clean, 
            border_thickness=border_thickness,
            area_threshold=area_threshold
        )
        
        # 3. Păstrează arborele principal
        mask_clean_final = _keep_largest_vessel_tree_custom(
            mask_clean_temp,
            min_ratio=min_tree_ratio
        )
        
        pixels_after = mask_clean_final.sum()
        pixels_removed = pixels_before - pixels_after
        removal_percent = (pixels_removed / pixels_before * 100) if pixels_before > 0 else 0
        
        # Salvează masca curățată
        output_path = os.path.join(output_dir, f"{name_without_ext}_clean.png")
        imsave(output_path, (mask_clean_final * 255).astype(np.uint8))
        
        # Vizualizare comparație
        if visualize:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(mask_original, cmap='gray')
            axes[0].set_title(f"Original\n{pixels_before} pixeli")
            axes[0].axis('off')
            
            axes[1].imshow(mask_clean_final, cmap='gray')
            axes[1].set_title(f"Curățat\n{pixels_after} pixeli")
            axes[1].axis('off')
            
            # Diferență (ce s-a eliminat)
            diff = mask_original.astype(int) - mask_clean_final.astype(int)
            axes[2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
            axes[2].set_title(f"Eliminat\n{pixels_removed} px ({removal_percent:.1f}%)")
            axes[2].axis('off')
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, "comparisons", f"{name_without_ext}_comparison.png")
            plt.savefig(comparison_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
        
        processed_count += 1
        print(f"✅ [{i+1}/{len(mask_files)}] {filename}: "
              f"{pixels_before} → {pixels_after} pixeli (-{removal_percent:.1f}%)")
    
    print("-" * 60)
    print(f"✨ Procesare completă! {processed_count} măști curățate în {output_dir}")
    return processed_count

def _remove_border_artifacts_custom(mask, border_thickness=10, area_threshold=500):
    """Versiune customizabilă pentru curățarea batch."""
    if mask.sum() == 0:
        return mask
    
    h, w = mask.shape
    labels = label(mask)
    if labels.max() == 0:
        return mask
        
    regions = regionprops(labels)
    output_mask = mask.copy()
    
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        
        touches_border = (minr < border_thickness or 
                         minc < border_thickness or 
                         maxr > h - border_thickness or 
                         maxc > w - border_thickness)
        
        if touches_border:
            border_mask = np.zeros_like(mask, dtype=bool)
            border_mask[:border_thickness, :] = True
            border_mask[-border_thickness:, :] = True
            border_mask[:, :border_thickness] = True
            border_mask[:, -border_thickness:] = True
            
            component_mask = (labels == region.label)
            border_pixels = (component_mask & border_mask).sum()
            total_pixels = region.area
            border_ratio = border_pixels / total_pixels if total_pixels > 0 else 0
            
            if (region.area < area_threshold and border_ratio > 0.5) or \
               (border_ratio > 0.7 and region.eccentricity > 0.95):
                output_mask[labels == region.label] = False
    
    return output_mask

def _keep_largest_vessel_tree_custom(mask, min_ratio=0.15):
    """Versiune customizabilă pentru păstrarea arborelui principal."""
    mask_clean = remove_small_objects(mask, min_size=50)
    labels = label(mask_clean)
    
    if labels.max() == 0:
        return mask_clean
        
    regions = regionprops(labels)
    regions.sort(key=lambda x: x.area, reverse=True)
    
    output = np.zeros_like(mask, dtype=bool)
    
    if len(regions) > 0:
        output[labels == regions[0].label] = True
        
        for i in range(1, min(len(regions), 4)):
            if regions[i].area > regions[0].area * min_ratio:
                output[labels == regions[i].label] = True
            
    return output

# ==========================================
# RULARE & CALCUL VASOMIM PROBABILITY
# ==========================================
if __name__ == "__main__":
    from skimage.measure import block_reduce
    
    # Configurare
    NUM_PATIENTS = 10
    dataset = ARCADEDataset(json_path='data/ARCADE/processed/dataset.json', split='train', task='Unsupervised')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = "binary_masks_output"
    os.makedirs(output_dir, exist_ok=True)

    print("Generare Măști Binare...")

    for i, batch in enumerate(dataloader):
        if i >= NUM_PATIENTS: break
        
        img = batch[0][0].numpy() if isinstance(batch, (list, tuple)) else batch[0].numpy()
        
        # --- 1. OBȚINEM SEGMENTAREA BINARĂ ---
        binary_mask, img_crop = get_binary_segmentation(img)
        
        # --- 2. CALCULĂM HARTA VASOMIM (din binar) ---
        # Formula: Densitatea pixelilor de vas (1) într-un patch de 16x16 (pt ViT)
        # Patch size 16 este standard pentru ViT-B/16
        PATCH_SIZE = 16 
        
        # Convertim boolean în float (0.0 sau 1.0)
        mask_float = binary_mask.astype(float)
        
        # Calculăm media pe patch-uri (Average Pooling)
        # Rezultatul va fi o matrice mică (ex: 14x14) cu valori între 0 și 1
        vasomim_prob_map = block_reduce(mask_float, block_size=(PATCH_SIZE, PATCH_SIZE), func=np.mean)
        
        # Normalizare sumă = 1 (pentru sampling probabilistic)
        if vasomim_prob_map.sum() > 0:
            vasomim_prob_map = vasomim_prob_map / vasomim_prob_map.sum()
        
        # --- VIZUALIZARE ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_crop, cmap='gray')
        axes[0].set_title("Input (Clipped)")
        axes[0].axis('off')
        
        axes[1].imshow(binary_mask, cmap='gray')
        axes[1].set_title("Segmentare Binară")
        axes[1].axis('off')
        
        # Vizualizam probabilitatea (upscaled pt a se vedea peste imagine)
        axes[2].imshow(vasomim_prob_map, cmap='hot', interpolation='nearest')
        axes[2].set_title(f"VasoMIM Probability\n(Density of Binary Mask)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/patient_{i:03d}.png")
        plt.close()
        print(f"Processed patient {i}")