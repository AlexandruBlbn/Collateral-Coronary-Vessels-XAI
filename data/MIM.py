import math
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def resolve_mask_block_count(
    input_size: int,
    mask_block_size: int,
    target_mask_ratio: float,
    fixed_mask_blocks: Optional[int] = None,
) -> Tuple[int, int, float]:
    if input_size % mask_block_size != 0:
        raise ValueError(
            f"input_size ({input_size}) must be divisible by mask_block_size ({mask_block_size})."
        )

    grid_size = input_size // mask_block_size
    token_count = grid_size * grid_size

    if fixed_mask_blocks is None:
        mask_count = int(math.ceil(token_count * float(target_mask_ratio)))
    else:
        mask_count = int(fixed_mask_blocks)

    mask_count = max(1, min(mask_count, token_count))
    effective_ratio = float(mask_count) / float(token_count)
    return mask_count, token_count, effective_ratio


class MaskGenerator:
    """
    Generates random non-overlapping contiguous block masks.

    For 256x256 with block_size=32, the coarse mask is an 8x8 grid.
    """

    def __init__(
        self,
        input_size: int = 256,
        mask_block_size: int = 32,
        mask_ratio: float = 0.65,
        fixed_mask_blocks: Optional[int] = None,
        seed: int = 42,
    ):
        self.input_size = int(input_size)
        self.mask_block_size = int(mask_block_size)
        self.mask_ratio = float(mask_ratio)
        self.base_seed = int(seed)

        self.mask_count, self.token_count, self.effective_mask_ratio = resolve_mask_block_count(
            input_size=self.input_size,
            mask_block_size=self.mask_block_size,
            target_mask_ratio=self.mask_ratio,
            fixed_mask_blocks=fixed_mask_blocks,
        )

        self.grid_size = self.input_size // self.mask_block_size
        self._epoch = 0
        self._rank = 0
        self._call_index = 0

    def set_epoch(self, epoch: int, rank: int = 0) -> None:
        self._epoch = int(epoch)
        self._rank = int(rank)
        self._call_index = 0

    def _sample_one(self, seed: int, dtype: torch.dtype) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        mask_ids = torch.randperm(self.token_count, generator=gen)[: self.mask_count]
        coarse_mask = torch.zeros(self.token_count, dtype=dtype)
        coarse_mask[mask_ids] = 1.0
        coarse_mask = coarse_mask.view(self.grid_size, self.grid_size)

        return coarse_mask.repeat_interleave(self.mask_block_size, dim=0).repeat_interleave(
            self.mask_block_size, dim=1
        )

    def __call__(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        # Offset seeds by epoch/rank/call index to avoid repeating identical masks across workers.
        base = self.base_seed + self._epoch * 100003 + self._rank * 1009 + self._call_index * 97
        masks = [self._sample_one(base + i, dtype=dtype) for i in range(batch_size)]
        self._call_index += batch_size

        stacked = torch.stack(masks, dim=0)
        if device is not None:
            stacked = stacked.to(device=device, non_blocking=True)
        return stacked


class XA170KImageDataset(Dataset):
    """
    Image-only XA-170K dataset for SimMIM pretraining.
    Expects a folder with source subfolders such as:
      coronarydominance/, xcad/, cadica/, syntax/
    """

    def __init__(
        self,
        base_path: str,
        input_size: int = 256,
        sources: Optional[Sequence[str]] = None,
        require_all_sources: bool = True,
        extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        self.base_path = Path(base_path)
        if sources is None:
            sources = ("coronarydominance", "xcad", "cadica", "syntax")
        self.sources = tuple(sources)
        self.require_all_sources = bool(require_all_sources)
        self.extensions = {e.lower() for e in extensions}

        if not self.base_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.base_path}")

        self.image_paths = []
        missing_sources = []

        for source in self.sources:
            source_dir = self.base_path / source
            if not source_dir.exists():
                missing_sources.append(source)
                continue

            source_paths = []
            for root, _, files in os.walk(source_dir):
                for name in files:
                    suffix = os.path.splitext(name)[1].lower()
                    if suffix in self.extensions:
                        source_paths.append(str(Path(root) / name))

            source_paths.sort()
            self.image_paths.extend(source_paths)

        if missing_sources and self.require_all_sources:
            raise FileNotFoundError(
                "Missing expected source folders under "
                f"{self.base_path}: {', '.join(missing_sources)}"
            )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.base_path} for sources {self.sources}")

        self.transform = T.Compose(
            [
                T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        return image


class ArcadeDatasetMIM(Dataset):
    """
    Backward-compatible wrapper that adds a runtime-generated mask to an arbitrary image dataset.
    """

    def __init__(self, arcade_dataset: Dataset, mask_generator: MaskGenerator):
        self.arcade_dataset = arcade_dataset
        self.mask_generator = mask_generator

    def __len__(self) -> int:
        return len(self.arcade_dataset)

    def __getitem__(self, idx: int):
        sample = self.arcade_dataset[idx]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        mask = self.mask_generator(batch_size=1).squeeze(0)
        return image, mask


if __name__ == "__main__":
    generator = MaskGenerator(input_size=256, mask_block_size=32, mask_ratio=0.65, fixed_mask_blocks=42)
    print(
        "Mask stats:",
        {
            "grid_size": generator.grid_size,
            "token_count": generator.token_count,
            "mask_count": generator.mask_count,
            "effective_mask_ratio": generator.effective_mask_ratio,
        },
    )
