"""Persistent Frangi vesselness cache for coronary X-ray angiography images.

This module computes Frangi vesselness responses once, stores them as torch
`.pt` tensors on disk, and reloads them across epochs to avoid repeated
expensive preprocessing.
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.filters import frangi
from tqdm import tqdm


class FrangiCache:
    """Disk-backed cache for Frangi vesselness maps.

    Args:
        cache_dir: Directory where `.pt` cache files are stored.
        sigmas: Sigma values passed to `skimage.filters.frangi`.
        image_size: Square output size for the cached Frangi response.
    """

    def __init__(self, cache_dir: str, sigmas=None, image_size: int = 256):
        self.cache_dir = cache_dir
        self.sigmas = list(range(1, 16, 2)) if sigmas is None else list(sigmas)
        self.image_size = image_size
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_key(self, img_path: str) -> str:
        """Convert an image path to a cache filename."""
        base = os.path.splitext(img_path)[0]
        return base.replace(os.sep, "_") + ".pt"

    def _compute_frangi(self, img_path: str) -> np.ndarray:
        """Compute the Frangi vesselness response for one image.

        Returns:
            A float32 NumPy array with shape [image_size, image_size].
        """
        img = Image.open(img_path).convert("L")
        img_np = np.array(img)
        img_np = cv2.GaussianBlur(img_np, (7, 7), sigmaX=3)
        img_np = img_np[16:-16, 16:-16]
        img_np = cv2.resize(img_np, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        img_np = img_np.astype(np.float32)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        vesselness = frangi(
            img_np,
            sigmas=self.sigmas,
            alpha=0.5,
            beta=1,
            gamma=10,
            mode="reflect",
            black_ridges=True,
        )
        p_low, p_high = np.percentile(vesselness, (1, 99.9))
        vesselness = np.clip(vesselness, p_low, p_high)
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
        vesselness = cv2.GaussianBlur(vesselness.astype(np.float32), (3, 3), 0)
        return vesselness.astype(np.float32)

    def get(self, img_path: str) -> torch.Tensor:
        """Return the cached Frangi response for an image path.

        If the cache entry does not exist, the response is computed, stored, and
        returned.
        """
        cache_key = self._cache_key(img_path)
        cache_path = os.path.join(self.cache_dir, cache_key)

        if os.path.exists(cache_path):
            try:
                cached = torch.load(cache_path, weights_only=True)
                if isinstance(cached, torch.Tensor):
                    return cached.to(dtype=torch.float32)
                return torch.as_tensor(cached, dtype=torch.float32)
            except Exception as exc:
                print(f"[FrangiCache] Warning: failed to load {cache_key}: {exc}")

        try:
            result = self._compute_frangi(img_path)
        except Exception as exc:
            print(f"[FrangiCache] Warning: failed to compute {img_path}: {exc}")
            result = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        tensor = torch.from_numpy(result).float()
        try:
            torch.save(tensor, cache_path)
            print(f"[FrangiCache] Cached: {cache_key}")
        except Exception as exc:
            print(f"[FrangiCache] Warning: failed to save {cache_key}: {exc}")
        return tensor


def precompute_all(dataset_samples: list, cache: FrangiCache) -> None:
    """Precompute Frangi responses for every sample in a dataset."""
    for sample in tqdm(dataset_samples, desc="Precomputing Frangi cache"):
        cache.get(sample["path"])
