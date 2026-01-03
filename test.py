import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from skimage.filters import frangi, threshold_otsu
from skimage import morphology
from skimage.measure import label, regionprops, block_reduce
from skimage.morphology import disk, reconstruction
from skimage.color import label2rgb

# --- IMPORTS ---
import sys
import os
sys.path.append(os.getcwd()) 
from data.ARCADE.dataloader import ARCADEDataset


def _keep_center_connected_component(mask: np.ndarray, *, margin_px: int = 15) -> np.ndarray:
    """Păstrează doar componenta conexă care conține centrul (și elimină o margine fixă)."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    h, w = mask.shape

    # Safety margin: eliminăm o bandă de pe toate marginile
    if margin_px > 0:
        edge_mask = np.ones_like(mask, dtype=bool)
        edge_mask[:margin_px, :] = False
        edge_mask[-margin_px:, :] = False
        edge_mask[:, :margin_px] = False
        edge_mask[:, -margin_px:] = False
        mask = mask & edge_mask

    if not mask.any():
        return mask

    cy, cx = h // 2, w // 2
    labels_cc = label(mask)

    if labels_cc[cy, cx] != 0:
        keep_label = labels_cc[cy, cx]
        return labels_cc == keep_label

    # Dacă centrul nu e în ROI, păstrăm cea mai mare componentă
    regions = regionprops(labels_cc)
    if not regions:
        return np.zeros_like(mask, dtype=bool)
    regions.sort(key=lambda r: r.area, reverse=True)
    return labels_cc == regions[0].label


def _frangi_response(img_norm: np.ndarray, sigmas: list[float], roi_mask: np.ndarray, *, gamma: float = 15.0) -> np.ndarray:
    """Răspuns Frangi normalizat (0..1) pentru un set de scale."""
    resp = frangi(
        img_norm,
        sigmas=sigmas,
        black_ridges=True,
        alpha=0.5,
        beta=0.5,
        gamma=gamma,
    )
    resp = resp * roi_mask
    m = float(resp.max())
    if m > 0:
        resp = resp / m
    return resp


def get_clean_vasomim_map(
    img,
    debug: bool = False,
    *,
    thresh_thick: float = 0.25,
    thresh_medium: float = 0.20,
    thresh_thin: float = 0.15,
    roi_thresh: float = 0.05,
    border_margin_px: int = 15,
    gamma: float = 15.0,
):
    """
    Segmentare multi-clasă DOAR cu Frangi (fără CLAHE/bilateral/etc.).

    Clase:
      0 = fundal
      1 = vase groase
      2 = vase medii
      3 = vase fine
    """
    
    # ==========================================
    # 1. PRE-PROCESARE MINIMĂ
    # ==========================================
    # Doar o normalizare simplă 0-1
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Mască ROI simplă + omiterea borderului:
    # 1) prag pe intensitate
    # 2) păstrăm doar componenta conectată din centru
    mask_roi = img_norm > roi_thresh
    mask_roi = morphology.binary_erosion(mask_roi, disk(5))
    mask_roi = _keep_center_connected_component(mask_roi, margin_px=border_margin_px)
    mask_roi = morphology.binary_dilation(mask_roi, disk(3))
    
    # ==========================================
    # 2. APLICARE FRANGI (PUR)
    # ==========================================
    # Scale grupate pe grosime (poți ajusta listele)
    sigmas_thick = [4.5, 6.0]
    sigmas_medium = [2.0, 4.4]
    sigmas_thin = [0.5, 1.9]

    resp_thick = _frangi_response(img_norm, sigmas_thick, mask_roi, gamma=gamma)
    resp_medium = _frangi_response(img_norm, sigmas_medium, mask_roi, gamma=gamma)
    resp_thin = _frangi_response(img_norm, sigmas_thin, mask_roi, gamma=gamma)

    # Combinat (pentru probabilități)
    frangi_response = np.maximum.reduce([resp_thick, resp_medium, resp_thin])
    
    # ==========================================
    # 3. BINARIZARE SIMPLĂ
    # ==========================================
    # Praguri custom pe clase
    mask_thick = resp_thick > thresh_thick
    mask_medium = resp_medium > thresh_medium
    mask_thin = resp_thin > thresh_thin

    # Rezolvăm overlap-urile cu prioritate: gros > mediu > fin
    vessel_class = np.zeros_like(img_norm, dtype=np.uint8)
    vessel_class[mask_thick] = 1
    vessel_class[(vessel_class == 0) & mask_medium] = 2
    vessel_class[(vessel_class == 0) & mask_thin] = 3

    # Curățare minimă per-clasă (ajustează dacă vrei)
    for cls, min_size in [(1, 80), (2, 50), (3, 20)]:
        m = vessel_class == cls
        m = morphology.remove_small_objects(m, min_size=min_size)
        m = morphology.remove_small_holes(m, area_threshold=80)
        vessel_class[vessel_class == cls] = 0
        vessel_class[m] = cls
    
    # ==========================================
    # 4. GENERARE MAPĂ VasoMIM
    # ==========================================
    PATCH_SIZE = 8
    
    vessel_any = vessel_class > 0
    final_prob_dilated = morphology.dilation(frangi_response * vessel_any, disk(2))
    patch_density = block_reduce(final_prob_dilated, block_size=(PATCH_SIZE, PATCH_SIZE), func=np.mean)
    
    # Center Bias
    h_p, w_p = patch_density.shape
    y, x = np.ogrid[:h_p, :w_p]
    center_bias = np.exp(-((x - w_p/2)**2 + (y - h_p/2)**2) / (2 * (w_p/2.5)**2))
    
    vasomim_probs = patch_density * center_bias
    vasomim_probs = (vasomim_probs - vasomim_probs.min()) / (vasomim_probs.max() - vasomim_probs.min() + 1e-6)
    
    vasomim_probs = vasomim_probs + 0.02
    vasomim_probs = vasomim_probs / vasomim_probs.sum()

    debug_payload = {
        "mask_roi": mask_roi,
        "resp_thick": resp_thick,
        "resp_medium": resp_medium,
        "resp_thin": resp_thin,
    }

    return vessel_class, vasomim_probs, debug_payload, img_norm


# ==========================================
# TESTARE
# ==========================================
if __name__ == "__main__":
    NUM_PATIENTS = 30

    pathJson = 'data/ARCADE/processed/dataset.json'
    dataset = ARCADEDataset(json_path=pathJson, split='train', task='Unsupervised')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = "vasomim_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Start processing {len(dataset)} patients...")

    for i, batch in enumerate(dataloader):
        if NUM_PATIENTS != "all" and i >= NUM_PATIENTS:
            print(f"Limita de {NUM_PATIENTS} pacienti atinsa.")
            break

        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
            
        img = images[0, 0].numpy()

        vessel_class, mim_probs, dbg, img_processed = get_clean_vasomim_map(
            img,
            debug=True,
            thresh_thick=0.01,
            thresh_medium=0.01,
            thresh_thin=0.01,
            roi_thresh=0.05,
            border_margin_px=15,
            gamma=20.0,
        )

        overlay = label2rgb(
            vessel_class,
            image=img,
            bg_label=0,
            colors=['red', 'orange', 'cyan'],
            alpha=0.35,
        )

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        axes[0].imshow(img_processed, cmap='gray')
        axes[0].set_title("Processed Input (Simple Normalization)")
        axes[0].axis('off')

        im_cls = axes[1].imshow(vessel_class, cmap='tab10', vmin=0, vmax=3)
        axes[1].set_title("Vessel Segmentation (Multi-class: 1=Thick,2=Med,3=Thin)")
        axes[1].axis('off')

        # colorbar compact pentru clase
        cbar = fig.colorbar(im_cls, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['bg', 'thick', 'med', 'thin'])

        axes[2].imshow(overlay)
        axes[2].set_title("Overlapped Segmentation")
        axes[2].axis('off')

        axes[3].imshow(mim_probs, cmap='jet')
        axes[3].set_title("VasoMIM Probability (Patch 8)")
        axes[3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"patient_{i:03d}.png")
        plt.savefig(save_path)
        plt.close(fig)
        
        if i % 10 == 0:
            print(f"Saved {save_path}")
            
    print("Done.")
