"""
DINOv3 Multi-Crop Augmentation for Coronary Angiography Pretraining.

Features:
    - Apply transforms once on the source image, then crop from the transformed image
    - 4-channel preprocessing: CLAHE | white-hat | black-hat | sharpness (trainv4 recipe)
    - 2 global crops (512x512) + 8 local crops (192x192)

Performance notes (why this is fast):
    - Morphology kernel & CLAHE object created once in __init__, reused every call.
    - Source image is capped at `max_source_size` before heavy CV ops; crops are
      then resized up to their target resolution.  Morphology on a 768px image is
      ~3× faster than on a 1800px original.
    - ElasticTransform (dense warp; very slow) replaced by a lightweight random
      affine shear/scale warp implemented directly with cv2.warpAffine.
    - antialias=False for local crops (192 px target, no visible benefit from AA).
    - No per-call kernel / CLAHE construction.
"""

import random
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# 4-channel preprocessing (identical to trainv4.py)
# ---------------------------------------------------------------------------

def build_4channel_tensor(
    crop_np: np.ndarray,
    kernel:  np.ndarray,
    clahe:   cv2.CLAHE,
) -> torch.Tensor:
    """
    Convert a uint8 grayscale image [H, W] → 4-channel float32 tensor [4, H, W] in [0, 1].
    Channels: CLAHE | white-hat (tophat) | black-hat | high-pass sharpness.

    `kernel` and `clahe` are pre-built objects passed in from DINOv3Augmentation
    so they are NOT re-created on every call.
    """
    c1 = clahe.apply(crop_np)
    c2 = cv2.morphologyEx(crop_np, cv2.MORPH_TOPHAT,   kernel)
    c3 = cv2.morphologyEx(crop_np, cv2.MORPH_BLACKHAT, kernel)
    blurred = cv2.GaussianBlur(crop_np, (0, 0), sigmaX=10)
    c4 = cv2.addWeighted(crop_np, 4.0, blurred, -4.0, 128)

    arr = np.stack([c1, c2, c3, c4], axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Fast random affine warp (replaces slow ElasticTransform)
# ---------------------------------------------------------------------------

def _random_affine_warp(img_np: np.ndarray,
                        shear_range: float = 0.06,
                        scale_range: tuple = (0.95, 1.05)) -> np.ndarray:
    """
    Apply a random affine transform (shear + scale) in-place via cv2.warpAffine.
    Much faster than torchvision ElasticTransform (no dense displacement field).
    """
    h, w = img_np.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    scale = random.uniform(*scale_range)
    shear_x = random.uniform(-shear_range, shear_range)
    shear_y = random.uniform(-shear_range, shear_range)

    M = np.array([
        [scale,          scale * shear_x, cx * (1 - scale) - cy * scale * shear_x],
        [scale * shear_y, scale,          cy * (1 - scale) - cx * scale * shear_y],
    ], dtype=np.float32)

    return cv2.warpAffine(img_np, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


# ---------------------------------------------------------------------------
# Main augmentation class
# ---------------------------------------------------------------------------

class DINOv3Augmentation:
    """
    Multi-crop strategy for DINOv3 pretraining on coronary angiographies.

    Returns per image:
        List of (4, H, W) float32 tensors:
        [ global_0, global_1, local_0, ..., local_{n_local-1} ]

    Parameters
    ----------
    max_source_size : int
        Cap the source image at this resolution before running the expensive
        4-channel CV ops.  The crops are upsampled back to their target size.
        Smaller → faster morphology / CLAHE.  768 is a good balance for
        512 px global crops.  Set to 0 to disable the cap.
    """

    def __init__(
        self,
        global_crop_size:          int   = 512,
        local_crop_size:           int   = 192,
        n_global_crops:            int   = 2,
        n_local_crops:             int   = 8,
        global_scale:              tuple = (0.4, 1.0),
        local_scale:               tuple = (0.05, 0.4),
        max_source_size:           int   = 768,
        morph_ksize:               int   = 15,
    ):
        self.global_crop_size  = global_crop_size
        self.local_crop_size   = local_crop_size
        self.n_global_crops    = n_global_crops
        self.n_local_crops     = n_local_crops
        self.global_scale      = global_scale
        self.local_scale       = local_scale
        self.max_source_size   = max_source_size
        self.crop_ratio        = (3.0 / 4.0, 4.0 / 3.0)

        # ── Pre-build expensive objects ONCE per worker ──────────────────────
        self.clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_np(pil_img: Image.Image) -> np.ndarray:
        return np.array(pil_img, dtype=np.uint8)

    def _cap_size(self, img_np: np.ndarray) -> np.ndarray:
        """Downscale the image so its longest side == max_source_size."""
        if self.max_source_size <= 0:
            return img_np
        h, w = img_np.shape[:2]
        longest = max(h, w)
        if longest <= self.max_source_size:
            return img_np
        scale = self.max_source_size / longest
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _augment_source(self, img_np: np.ndarray) -> np.ndarray:
        """
        Stochastic photometric / geometric augmentations applied to the
        source uint8 numpy image (grayscale, 2-D).
        All ops work on numpy arrays directly — no PIL round-trips.
        """
        # Flips
        if random.random() < 0.5:
            img_np = np.fliplr(img_np)
        if random.random() < 0.5:
            img_np = np.flipud(img_np)

        # Fast affine warp replaces slow ElasticTransform
        if random.random() < 0.3:
            img_np = _random_affine_warp(img_np)

        # Brightness (additive noise on uint8)
        if random.random() < 0.6:
            delta = random.uniform(-0.25, 0.25)  # fraction of 255
            img_np = np.clip(img_np.astype(np.int16) + int(delta * 255),
                             0, 255).astype(np.uint8)

        # Contrast (multiply around mean)
        if random.random() < 0.4:
            factor = random.uniform(0.8, 1.2)
            mean   = int(img_np.mean())
            img_np = np.clip(
                mean + factor * (img_np.astype(np.float32) - mean),
                0, 255).astype(np.uint8)

        # Gaussian blur
        if random.random() < 0.4:
            sigma = random.uniform(0.1, 2.0)
            # kernel size must be odd; clamp to (3,3) – (9,9)
            ks = max(3, min(9, int(sigma * 3) * 2 + 1)) | 1
            img_np = cv2.GaussianBlur(img_np, (ks, ks), sigma)

        return np.ascontiguousarray(img_np)

    def _sample_crop(
        self,
        source_pil:    Image.Image,
        source_tensor: torch.Tensor,
        target_size:   int,
        scale:         tuple,
        antialias:     bool = True,
    ) -> torch.Tensor:
        i, j, h, w = T.RandomResizedCrop.get_params(
            source_pil, scale=scale, ratio=self.crop_ratio)
        return TF.resized_crop(
            source_tensor, i, j, h, w,
            size=[target_size, target_size],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=antialias,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, pil_img: Image.Image) -> list:
        """
        Args:
            pil_img: PIL Image (any mode; converted to L internally)
        Returns:
            List[Tensor(4, H, W)]: 2 global + n_local crops
        """
        # 1. Convert to uint8 numpy (grayscale)
        img_np = self._to_np(pil_img.convert('L'))

        # 2. Stochastic source-level augmentation (fast numpy / cv2 ops)
        img_np = self._augment_source(img_np)

        # 3. Cap resolution before morphology (big win on high-res originals)
        img_np_small = self._cap_size(img_np)

        # 4. Build 4-channel tensor from the (possibly downscaled) source
        source_tensor = build_4channel_tensor(img_np_small, self.kernel, self.clahe)

        # 5. PIL proxy for RandomResizedCrop.get_params (needs H/W only, no data copy)
        h_s, w_s = img_np_small.shape[:2]
        source_pil = Image.new('L', (w_s, h_s))   # empty PIL — only shape matters

        crops = []

        # Global crops — antialias ON (quality matters, 2 crops only)
        for _ in range(self.n_global_crops):
            crops.append(self._sample_crop(
                source_pil, source_tensor,
                target_size=self.global_crop_size,
                scale=self.global_scale,
                antialias=True,
            ))

        # Local crops — antialias OFF (192 px target; AA is negligible / slow)
        for _ in range(self.n_local_crops):
            crops.append(self._sample_crop(
                source_pil, source_tensor,
                target_size=self.local_crop_size,
                scale=self.local_scale,
                antialias=False,
            ))

        return crops