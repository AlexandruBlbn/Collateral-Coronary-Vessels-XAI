# VasoJEPA v2 — Robust Vessel Prior & Self-Correcting Latent Denoising

> **Addresses**: Systematic failure modes of traditional Hessian/Frangi/Sato filters on coronary angiograms:
> 1. **Catheters (False Positives)**: High-contrast, thick, linear structures.
> 2. **Ribs (False Positives)**: Wide, diffuse bone shadows.
> 3. **Faint Vessels (False Negatives)**: Narrow, low-contrast, or overlapping vessels (including collaterals).

---

## 1. Diagnostic of Frangi/Sato Failure Modes

On coronary X-ray angiograms, traditional vesselness filters (like Frangi or Sato) suffer from three systematic failures that contaminate self-supervised learning (SSL) guidance if used naively:

```
                  ┌────────────────────────────────────────┐
                  │       Angiogram Input Image            │
                  └──────────────────┬─────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
  [Catheter (Dark/Thick)]     [Ribs (Diffuse/Wide)]      [Collaterals (Faint/Thin)]
         │                           │                           │
  Sato response: STRONG       Sato response: MEDIUM      Sato response: WEAK/NONE
  (False Positive)            (False Positive)           (False Negative)
         │                           │                           │
         ▼                           ▼                           ▼
  ──────────────────────► SOLUTIONS IMPLEMENTED ◄─────────────────────────
         │                           │                           │
  1. Scale Restriction        1. Local Ridge Modulation  1. Self-Correcting LDS
  2. Intensity Thresholding      (Suppresses diffuse      2. Guidance Annealing
  3. Border Distance Weighting   broad structures)          (Allows denoiser to
                                                            discover faint anatomy)
```

1. **Catheters**: Because catheters are highly radio-opaque tube-like objects, they yield the highest vesselness response. If the network is guided by raw vesselness, it focuses its highest-capacity representations on the catheter rather than coronary anatomy.
2. **Ribs**: Ribs are stationary bone structures that cross the heart. Their boundaries form long, low-frequency linear edges. Hessian-based filters with broad scale ranges ($\sigma \ge 4$) respond to these edges, creating large false-positive vessel regions.
3. **Faint/Collateral Vessels**: These are often 1-2 pixels wide with very low local contrast. If we set a threshold to filter out ribs and background noise, these faint vessels are the first to be discarded, leading to false negatives.

---

## 2. Robust Consensus Prior (RCP) Formulation

To fix these issues, we construct a **Robust Consensus Prior (RCP)** that combines multi-scale geometry, local intensity contrast, and intensity masking to suppress false positives.

### 2.1 Scale-Restricted Sato Filter ($S_{\text{restricted}}$)
We restrict the Sato filter's scales (sigmas) to fine structures:
$$\sigma \in \{1.0, 1.5, 2.0, 2.5\}$$
By excluding larger scales ($\sigma \ge 4.0$), we drastically reduce the response on thick catheters and broad ribs.

### 2.2 Local Dark Ridge Modulation ($R_{\text{local}}$)
Coronary vessels are thin, dark valleys in a brighter background. Ribs are diffuse and lack sharp local contrast. We compute a local dark ridge response:
$$R_{\text{local}}(x) = \frac{\text{LocalMean}_k(x) - x}{\max(\text{LocalMean}_k(x) - x) + \epsilon}$$
where local mean is computed using a window of size $k = 9$ (matching the vessel width). We then modulate the Sato response:
$$V_{\text{modulated}} = S_{\text{restricted}} \cdot R_{\text{local}}$$
Because ribs have broad gradients, $\text{LocalMean}_k(x) - x \approx 0$, which suppresses rib boundaries.

### 2.3 Catheter & Border Suppression ($M_{\text{catheter}}$ & $W_{\text{border}}$)
1. **Intensity Masking**: Catheters are extremely dark (often near black). We mask out pixels below a low-intensity threshold (e.g., $I < 20$ on a $[0, 255]$ scale) to suppress catheter shafts and marker bands:
   $$M_{\text{catheter}}(x) = \mathbb{I}(x > 20)$$
2. **Border Distance Weighting**: Catheters always enter the frame from the image boundaries. We apply a soft border decay function $W_{\text{border}}$ that damps the prior near the borders where catheter introduction occurs:
   $$W_{\text{border}}(i, j) = \text{sigmoid}\left(\frac{\min(i, H-i, j, W-j) - d_{\text{border}}}{s_{\text{border}}}\right)$$
   where $d_{\text{border}} = 16$ (one patch width) and $s_{\text{border}} = 4.0$.

### 2.4 Final RCP Score
Combining all components, the pixel-level Robust Consensus Prior map is:
$$\text{RCP}(x) = V_{\text{modulated}}(x) \cdot M_{\text{catheter}}(x) \cdot W_{\text{border}}(x)$$
We then average-pool this map to the $14 \times 14$ patch grid (matching the ViT patches) to obtain the soft patch guidance targets:
$$y_i = \text{AvgPool}_{16 \times 16}(\text{RCP})^{(i)}$$

---

## 3. Self-Correcting Latent Denoising (SCLD)

While the RCP is significantly cleaner than the raw Frangi filter, some noise will remain. We use **Self-Correcting Latent Denoising (SCLD)** in the LDS branch to allow the network to self-correct during training.

### 3.1 Denoising as a Manifold Learner
The Latent Denoising Score (LDS) branch is trained with two objectives:
$$\mathcal{L}_{\text{lds}} = \mathcal{L}_{\text{denoise}} + \lambda_{\text{guide}}(t) \cdot \mathcal{L}_{\text{guide}}$$

1. **Self-Supervised Denoising ($\mathcal{L}_{\text{denoise}}$)**: 
   $$\mathcal{L}_{\text{denoise}} = \mathbb{E}_{z, \epsilon, t} \left[\|\epsilon - D_\phi(z_t, t)\|_2^2\right]$$
   This objective is completely unsupervised. It forces the MLP denoiser $D_\phi$ to learn the natural data manifold of coronary angiograms. Since vessels are dynamic and variable across images, while ribs are static/background and catheters are out-of-distribution artifacts, the denoiser naturally learns to represent genuine coronary vessels.
2. **Soft Guidance Classification ($\mathcal{L}_{\text{guide}}$)**:
   $$\mathcal{L}_{\text{guide}} = \text{BCE}(V_\phi(\hat{f}_{12}), y_{\text{consensus}})$$
   This connects the estimated clean embeddings $\hat{f}_{12}$ to our robust prior $y_{\text{consensus}}$ via a vessel read-out head $V_\phi$.

### 3.2 Dynamic Guidance Annealing
To prevent the noisy prior from permanently biasing the model, we **anneal the guidance weight** $\lambda_{\text{guide}}$ over the pretraining epochs:
$$\lambda_{\text{guide}}(epoch) = \lambda_{\text{min}} + (\lambda_{\text{max}} - \lambda_{\text{min}}) \cdot \cos\left(\frac{\pi \cdot epoch}{2 \cdot \text{TotalEpochs}}\right)$$
*   **Early training (Epochs 0–50)**: $\lambda_{\text{guide}} \approx 1.0$. The model uses the RCP to bootstrap and locate general vessel structures.
*   **Late training (Epochs 150+ )**: $\lambda_{\text{guide}} \approx 0.05$. The guidance is turned off, and the self-supervised denoising loss takes over. The model refines the vessel representations on its own, naturally restoring faint vessels and ignoring catheter/rib artifacts that do not match the learned coronary manifold.

### 3.3 Bootstrapped Target Update (Reed's Formulation)
Rather than fitting static $y_i$ directly, we update the targets dynamically using the EMA teacher's consensus prediction:
$$y_{\text{consensus}}^{(i)} = \gamma \cdot y_i + (1 - \gamma) \cdot \text{sg}\left(V_{\phi,\text{EMA}}(\hat{f}_{12,\text{EMA}}^{(i)})\right)$$
where $\gamma = 0.7$ controls the trust in the prior vs. the model's own predictions. This allows the model to "fill in" faint vessels that it detects but which are missing from the prior.

---

## 4. Implementation Code

Below is the implementation of the robust consensus prior generation. This code will replace the naive region-growing pipeline and provide clean, soft $14 \times 14$ targets for VasoJEPA pretraining.

Save this as [robust_prior.py](file:///D:/Collateral%20Coronary%20Vessels%20XAI/XA-SSL-REPO/robust_prior.py).

```python
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.filters import sato

def compute_robust_prior(image: np.ndarray, patch_size: int = 16, border_pad: int = 16) -> np.ndarray:
    """
    Computes a Robust Consensus Prior (RCP) map for coronary vessels.
    Suppresses:
      - Catheters (via intensity thresholding and border decay)
      - Ribs (via local dark ridge contrast modulation and restricted scale Sato)
    Recovers:
      - Faint vessels (retained as soft probabilities rather than binary masks)
    
    Args:
        image: Grayscale image (H, W) in range [0, 255].
        patch_size: Resolution of the target grid (typically 16 for ViT-S/16).
        border_pad: Distance from border to start decaying prior.
        
    Returns:
        patch_prior: Soft prior scores of shape (H/patch_size, W/patch_size).
    """
    H, W = image.shape
    img_float = image.astype(np.float32)

    # 1. Scale-Restricted Sato Filter (forces focus on thin/medium vessels, not wide ribs)
    # Sigmas are restricted to 1.0, 1.5, 2.0, 2.5
    sato_response = sato(image, sigmas=[1.0, 1.5, 2.0, 2.5], black_ridges=True, mode="reflect")
    # Normalize Sato response to [0, 1]
    sato_norm = sato_response / (sato_response.max() + 1e-6)

    # 2. Local Dark Ridge Modulation (suppresses diffuse rib shadows)
    # Compute local mean in 9x9 neighborhood
    local_mean = cv2.blur(img_float, (9, 9))
    dark_ridge = (local_mean - img_float).clip(min=0.0)
    dark_ridge_norm = dark_ridge / (dark_ridge.max() + 1e-6)

    # Modulate Sato response with local dark ridge
    vessel_prob = sato_norm * dark_ridge_norm

    # 3. Catheter Suppression (extremely dark metallic objects)
    # Mask out pixels that are very close to black (catheters, marker bands, background boundaries)
    catheter_mask = np.ones_like(img_float)
    catheter_mask[image < 20] = 0.0

    # 4. Border Distance Decay (suppresses catheter entry points near image edges)
    # Distance transform from boundaries
    border_dist = np.zeros_like(img_float)
    for i in range(H):
        for j in range(W):
            border_dist[i, j] = min(i, H - 1 - i, j, W - 1 - j)
    
    # Soft sigmoid decay: 0 at border, 1 at border_pad
    # sigmoid((d - border_pad) / scale)
    border_weight = 1.0 / (1.0 + np.exp(-(border_dist - border_pad) / 4.0))

    # 5. Combined Robust Consensus Prior
    rcp_map = vessel_prob * catheter_mask * border_weight
    # Normalize to [0, 1] range globally
    rcp_map = rcp_map / (rcp_map.max() + 1e-6)

    # 6. Average Pool to Patch Grid (14x14 for 224x224 input with patch_size=16)
    grid_h, grid_w = H // patch_size, W // patch_size
    patch_prior = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    for i in range(grid_h):
        for j in range(grid_w):
            patch = rcp_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patch_prior[i, j] = patch.mean()
            
    # Apply soft scaling to map background closer to 0 and enhance intermediate features
    patch_prior = np.clip(patch_prior, 0, 1)
    
    return patch_prior
```
