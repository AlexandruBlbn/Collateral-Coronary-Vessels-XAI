import os
import json
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import cv2
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class LeJepaDenseDataset(Dataset):
    def __init__(self, base_dataset_json, crops_json_path, root_dir='.', 
                 num_global=2, num_local=4, global_size=224, local_size=96, max_jitter=4,
                 num_vessel_classes=26):
        """
        Dataloader engineered for Dense LeJEPA using pre-computed exact coordinate pools.
        """
        self.root_dir = root_dir
        self.num_global = num_global
        self.num_local = num_local
        self.global_size = global_size
        self.local_size = local_size
        self.max_jitter = max_jitter
        self.num_vessel_classes = num_vessel_classes

        # We load base to know the splits, but we only really care about train pretraining
        with open(base_dataset_json, 'r') as f:
            base_data = json.load(f)
            
        with open(crops_json_path, 'r') as f:
            self.crops_meta = json.load(f)
            
        self.samples = []
        train_split = base_data.get('train', {})
        for source, items in train_split.items():
            for s_id, s_info in items.items():
                img_path = s_info.get('data')
                if img_path and img_path in self.crops_meta:
                    meta = self.crops_meta[img_path]
                    # Only include if it has enough valid bounding boxes
                    if len(meta.get('global_crops', [])) > 0 and len(meta.get('local_crops', [])) > 0:
                        self.samples.append({
                            'path': img_path,
                            'source': source,
                            'meta': meta
                        })
                        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        print(f"Loaded {len(self.samples)} valid samples for Dense SSL.")

    def __len__(self):
        return len(self.samples)

    def _apply_jitter(self, coord, h_img, w_img):
        y, x, h, w = coord['y'], coord['x'], coord['h'], coord['w']
        
        # apply jitter
        jy = random.randint(-self.max_jitter, self.max_jitter)
        jx = random.randint(-self.max_jitter, self.max_jitter)
        
        y = max(0, min(y + jy, h_img - h))
        x = max(0, min(x + jx, w_img - w))
        
        return int(y), int(x), int(h), int(w)
        
    def _stochastic_aug(self, crop_t):
        # We NO LONGER augment the individual patches independently because Dense SSL 
        # requires strict topological alignment. If we flip the Target patch independently, 
        # the Context Predictor still predicts the unflipped vector sequence, causing spatial collapse.
        # We will instead apply transformations at the FULL IMAGE level below.
        return crop_t

    @staticmethod
    def _center_in_abs_box(cy, cx, box):
        y, x, h, w = box
        return (cy >= y) and (cy <= y + h) and (cx >= x) and (cx <= x + w)

    def __getitem__(self, idx):
        item = self.samples[idx]
        abs_path = os.path.join(self.root_dir, item['path'])
        meta = item['meta']
        
        # Load and CLAHE
        img = np.array(Image.open(abs_path).convert('L'))
        img_h, img_w = img.shape
        img = self.clahe.apply(img)
        
        # Normalize to [-1, 1] tensor for backbone
        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        img_t = img_t * 2.0 - 1.0 # [-1, 1] normalization
        
        g_candidates = [dict(c) for c in meta['global_crops']]
        l_candidates = [dict(c) for c in meta['local_crops']]
        
        # Whole-Canvas Topolocial Augmentations: Apply flips to the source image 
        # AND correctly mathematically invert the bounding box tracking coordinates identically.
        if random.random() < 0.5:
            img_t = TF.hflip(img_t)
            for c in g_candidates + l_candidates:
                c['x'] = float(img_w - (c['x'] + c['w']))
                
        if random.random() < 0.5:
            img_t = TF.vflip(img_t)
            for c in g_candidates + l_candidates:
                c['y'] = float(img_h - (c['y'] + c['h']))
                
        # Simulating X-Ray Noise / Variance 
        if random.random() < 0.8:
            noise_sigma = random.uniform(0.01, 0.05)
            img_t = img_t + torch.randn_like(img_t) * noise_sigma
            img_t = torch.clamp(img_t, -1.0, 1.0)
        
        global_crops_t = []
        global_boxes = []
        local_crops_t = []
        local_boxes = []
        
        # Random pick global coords; prefer unique contexts for better local coverage.
        if len(g_candidates) >= self.num_global:
            g_coords = random.sample(g_candidates, k=self.num_global)
        else:
            g_coords = random.choices(g_candidates, k=self.num_global)

        selected_global_abs = []
        
        for c in g_coords:
            y, x, h, w = self._apply_jitter(c, img_h, img_w)
            crop = TF.crop(img_t, y, x, h, w)
            global_crops_t.append(crop)
            global_boxes.append([y / img_h, x / img_w, h / img_h, w / img_w])
            selected_global_abs.append((y, x, h, w))
            
        # Prefer local crops whose centers are inside at least one selected global crop.
        local_pool = []
        for c in l_candidates:
            cy = float(c['y']) + 0.5 * float(c['h'])
            cx = float(c['x']) + 0.5 * float(c['w'])
            if any(self._center_in_abs_box(cy, cx, g) for g in selected_global_abs):
                local_pool.append(c)

        if len(local_pool) == 0:
            local_pool = l_candidates

        if len(local_pool) >= self.num_local:
            l_coords = random.sample(local_pool, k=self.num_local)
        else:
            l_coords = random.choices(local_pool, k=self.num_local)
        
        for c in l_coords:
            y, x, h, w = self._apply_jitter(c, img_h, img_w)
            # Keep local-center alignment with selected globals after jitter when possible.
            valid = False
            for _ in range(3):
                cy = float(y) + 0.5 * float(h)
                cx = float(x) + 0.5 * float(w)
                if any(self._center_in_abs_box(cy, cx, g) for g in selected_global_abs):
                    valid = True
                    break
                y, x, h, w = self._apply_jitter(c, img_h, img_w)

            if not valid:
                y = int(max(0, min(float(c['y']), img_h - float(c['h']))))
                x = int(max(0, min(float(c['x']), img_w - float(c['w']))))
                h = int(float(c['h']))
                w = int(float(c['w']))

            crop = TF.crop(img_t, y, x, h, w)
            local_crops_t.append(crop)
            local_boxes.append([y / img_h, x / img_w, h / img_h, w / img_w])
            
        # Classification probe metadata
        syntax_classes = meta.get('syntax_classes', [])
        is_syntax = (item['source'] == 'syntax')
        
        cls_target = torch.zeros(self.num_vessel_classes, dtype=torch.float32)
        if is_syntax:
            for cls_id in syntax_classes:
                if 1 <= cls_id <= self.num_vessel_classes:
                    cls_target[cls_id - 1] = 1.0
                    
        return {
            'global_crops': torch.stack(global_crops_t),
            'global_boxes': torch.tensor(global_boxes, dtype=torch.float32),
            'local_crops': torch.stack(local_crops_t),
            'local_boxes': torch.tensor(local_boxes, dtype=torch.float32),
            'is_syntax': torch.tensor(is_syntax),
            'cls_target': cls_target
        }
def _natural_sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _resolve_path(base_dir: Optional[Union[str, Path]], relative_path: Union[str, Path]) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or base_dir in (None, ""):
        return path
    return Path(base_dir) / path


def _load_grayscale_image(path: Union[str, Path]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


def _to_tensor_label(label: Image.Image) -> torch.Tensor:
    label_tensor = TF.pil_to_tensor(label).float()
    if label_tensor.numel() > 0 and label_tensor.max().item() > 1.0:
        label_tensor = label_tensor / 255.0
    return label_tensor


class ArcadeDataset(Dataset):
    """ARCADE loader with explicit mode handling.

    Modes:
    - pretrain: combined sources, optional subset via ``pretrain_limit``.
    - syntax: ARCADE syntax images and labels.
    - stenosis: ARCADE stenoza images and labels.

    The dataset returns PIL images. For pretrain it returns the image only by default.
    For syntax and stenosis it returns ``(image, label)``.
    """

    PRETRAIN_SOURCES = ("syntax", "stenoza", "cadica", "extra", "coronarydominance")

    def __init__(
        self,
        json_path,
        split="train",
        transform=None,
        mode="syntax",
        root_dir=None,
        pretrain_limit="all",
        label_key="label",
        sources=None,
    ):
        self.json_path = json_path
        self.split = split
        self.transform = transform
        self.mode = self._normalize_mode(mode)
        self.root_dir = root_dir
        self.pretrain_limit = pretrain_limit
        self.label_key = label_key
        self.sources = tuple(sources) if sources is not None else self.PRETRAIN_SOURCES

        with open(json_path, "r") as file_handle:
            self.data = json.load(file_handle)

        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in dataset.")

        self.samples = self._prepare_data()

    @staticmethod
    def _normalize_mode(mode):
        normalized = mode.lower() if mode else "syntax"
        if normalized == "stenosis":
            return "stenoza"
        return normalized

    def _prepare_data(self):
        split_data = self.data[self.split]
        if not isinstance(split_data, dict):
            raise ValueError(f"Split '{self.split}' must contain a mapping of sources.")

        samples = []

        if self.mode == "pretrain":
            for source_name in self.sources:
                source_data = split_data.get(source_name)
                if not isinstance(source_data, dict):
                    continue
                for sample_id in sorted(source_data.keys(), key=_natural_sort_key):
                    sample_info = source_data[sample_id]
                    if not isinstance(sample_info, dict):
                        continue
                    image_path = sample_info.get("data")
                    if not image_path:
                        continue
                    samples.append(
                        {
                            "image_path": image_path,
                            "label_path": sample_info.get(self.label_key),
                            "source": source_name,
                            "id": sample_id,
                        }
                    )

            if isinstance(self.pretrain_limit, int):
                if self.pretrain_limit < 1:
                    raise ValueError("pretrain_limit must be a natural number or 'all'.")
                samples = samples[: self.pretrain_limit]
            elif isinstance(self.pretrain_limit, str):
                if self.pretrain_limit.lower() != "all":
                    raise ValueError("pretrain_limit must be a natural number or 'all'.")
            elif self.pretrain_limit is not None:
                raise TypeError("pretrain_limit must be a natural number, 'all', or None.")
            return samples

        source_name = self.mode
        source_data = split_data.get(source_name)
        if not isinstance(source_data, dict):
            raise ValueError(f"Source '{source_name}' not found in split '{self.split}'")

        for sample_id in sorted(source_data.keys(), key=_natural_sort_key):
            sample_info = source_data[sample_id]
            if not isinstance(sample_info, dict):
                continue
            image_path = sample_info.get("data")
            if not image_path:
                continue
            samples.append(
                {
                    "image_path": image_path,
                    "label_path": sample_info.get(self.label_key, sample_info.get("label")),
                    "source": source_name,
                    "id": sample_id,
                }
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = _resolve_path(self.root_dir, item["image_path"])
        image = _load_grayscale_image(image_path)

        if self.mode == "pretrain":
            if self.transform is not None:
                return self.transform(image)
            return image

        label_path = item.get("label_path")
        if isinstance(label_path, str) and label_path:
            label = _load_grayscale_image(_resolve_path(self.root_dir, label_path))
        else:
            label = Image.new("L", image.size, 0)

        if self.transform is not None:
            return self.transform(image, label)

        return image, label


class XCADTestDataset(Dataset):
    """Paired XCAD test dataset returning PIL image/mask pairs."""

    def __init__(
        self,
        base_dir="data/XCAD/XCAD/test",
        transform=None,
        image_dir="images",
        mask_dir="masks",
        extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.image_dir = self.base_dir / image_dir
        self.mask_dir = self.base_dir / mask_dir
        self.extensions = {extension.lower() for extension in extensions}

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.samples = self._collect_pairs()
        if len(self.samples) == 0:
            raise RuntimeError(f"No paired image/mask files found under {self.base_dir}")

    def _collect_pairs(self):
        samples = []
        image_paths = sorted(
            [path for path in self.image_dir.iterdir() if path.is_file() and path.suffix.lower() in self.extensions],
            key=lambda path: path.name,
        )

        for image_path in image_paths:
            mask_path = self.mask_dir / image_path.name
            if not mask_path.exists():
                continue
            samples.append({"image_path": image_path, "mask_path": mask_path})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = _load_grayscale_image(item["image_path"])
        mask = _load_grayscale_image(item["mask_path"])

        if self.transform is not None:
            return self.transform(image, mask)

        return image, mask


class Transforms:
    """Paired torchvision-style transforms for PIL images and masks.

    Geometry is sampled once and applied to both image and label.
    Intensity transforms are applied to the image only.
    """

    def __init__(
        self,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        training: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        rotation_degrees: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        blur_prob: float = 0.0,
        blur_kernel_size: int = 5,
        normalize: bool = True,
        mean: Sequence[float] = (0.5,),
        std: Sequence[float] = (0.5,),
    ):
        if image_size is None:
            self.image_size = None
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)

        self.training = bool(training)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotation_degrees = float(rotation_degrees)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.blur_prob = float(blur_prob)
        self.blur_kernel_size = int(blur_kernel_size)
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1
        self.normalize = bool(normalize)
        self.mean = tuple(mean)
        self.std = tuple(std)

    def _resize_pair(self, image, label):
        if self.image_size is None:
            return image, label

        image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        if isinstance(label, Image.Image):
            label = TF.resize(label, self.image_size, interpolation=InterpolationMode.NEAREST)
        return image, label

    @staticmethod
    def _apply_if_pil(label, transform_fn):
        if isinstance(label, Image.Image):
            return transform_fn(label)
        return label

    def __call__(self, image, label=None):
        if not isinstance(image, Image.Image):
            raise TypeError("Transforms expects PIL images as input.")

        image, label = self._resize_pair(image, label)

        if self.training:
            if torch.rand(1).item() < self.hflip_prob:
                image = TF.hflip(image)
                label = self._apply_if_pil(label, TF.hflip)

            if torch.rand(1).item() < self.vflip_prob:
                image = TF.vflip(image)
                label = self._apply_if_pil(label, TF.vflip)

            if self.rotation_degrees > 0:
                angle = float(torch.empty(1).uniform_(-self.rotation_degrees, self.rotation_degrees).item())
                if angle != 0.0:
                    image = TF.rotate(
                        image,
                        angle,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    )
                    label = self._apply_if_pil(
                        label,
                        lambda pil_label: TF.rotate(
                            pil_label,
                            angle,
                            interpolation=InterpolationMode.NEAREST,
                            fill=0,
                        ),
                    )

            if self.blur_prob > 0 and torch.rand(1).item() < self.blur_prob:
                sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
                image = TF.gaussian_blur(
                    image,
                    kernel_size=[self.blur_kernel_size, self.blur_kernel_size],
                    sigma=[sigma, sigma],
                )

            if self.brightness > 0:
                factor = float(torch.empty(1).uniform_(1.0 - self.brightness, 1.0 + self.brightness).item())
                image = TF.adjust_brightness(image, factor)

            if self.contrast > 0:
                factor = float(torch.empty(1).uniform_(1.0 - self.contrast, 1.0 + self.contrast).item())
                image = TF.adjust_contrast(image, factor)

        image_tensor = TF.to_tensor(image)
        if self.normalize:
            image_tensor = TF.normalize(image_tensor, self.mean, self.std)

        if label is None:
            return image_tensor

        if isinstance(label, Image.Image):
            label_tensor = _to_tensor_label(label)
        elif torch.is_tensor(label):
            label_tensor = label
        else:
            label_tensor = torch.as_tensor(label)

        return image_tensor, label_tensor


def _sample_to_image_and_label(sample):
    if isinstance(sample, dict):
        image = sample.get("image")
        label = sample.get("label")
        return image, label

    if isinstance(sample, (tuple, list)):
        if len(sample) == 0:
            raise ValueError("Empty sample received for plotting.")
        if len(sample) == 1:
            return sample[0], None
        return sample[0], sample[1]

    return sample, None


def _image_to_numpy(image, mean=None, std=None):
    if isinstance(image, Image.Image):
        array = np.asarray(image)
    elif torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            if mean is not None and std is not None:
                mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
                std_tensor = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
                if mean_tensor.numel() == tensor.shape[0] and std_tensor.numel() == tensor.shape[0]:
                    tensor = tensor * std_tensor + mean_tensor
            elif tensor.min().item() < 0.0 or tensor.max().item() > 1.0:
                if tensor.min().item() >= -1.1 and tensor.max().item() <= 1.1:
                    tensor = (tensor + 1.0) / 2.0

            tensor = tensor.clamp(0.0, 1.0)
            if tensor.shape[0] == 1:
                array = tensor.squeeze(0).numpy()
            else:
                array = tensor.permute(1, 2, 0).numpy()
        elif tensor.ndim == 2:
            array = tensor.numpy()
        else:
            array = tensor.squeeze().numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array.squeeze(-1)
    return array


def plot_dataset_sample(
    dataset,
    index=0,
    transform=None,
    overlay_label=False,
    show_label=True,
    image_cmap="gray",
    label_cmap="magma",
    alpha=0.35,
    title=None,
    figsize=None,
    save_path=None,
    show=True,
    mean=None,
    std=None,
):
    """Plot a single sample from ``ArcadeDataset`` or ``XCADTestDataset``.

    The function works with unlabeled pretrain samples as well as paired image/mask samples.
    If ``transform`` is supplied, it is applied to the sample before plotting.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plot_dataset_sample().") from exc

    sample = dataset[index]
    if transform is not None:
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            sample = transform(sample[0], sample[1])
        else:
            sample = transform(sample)

    image, label = _sample_to_image_and_label(sample)
    image_array = _image_to_numpy(image, mean=mean, std=std)

    has_label = label is not None and show_label
    if figsize is None:
        figsize = (10, 5) if has_label and not overlay_label else (6, 6)

    if has_label and not overlay_label:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = np.atleast_1d(axes)
        axes[0].imshow(image_array, cmap=image_cmap)
        axes[0].set_title("Image")
        axes[0].axis("off")

        label_array = _image_to_numpy(label)
        axes[1].imshow(label_array, cmap=label_cmap)
        axes[1].set_title("Label")
        axes[1].axis("off")
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(image_array, cmap=image_cmap)

        if has_label and overlay_label:
            label_array = _image_to_numpy(label)
            ax.imshow(label_array, cmap=label_cmap, alpha=alpha)
            ax.set_title("Image + label")
        else:
            ax.set_title("Image")

        ax.axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    return fig


def help():
    print("Example for class istance creation:")
    print("    dataset = ArcadeDataset(root_dir, mode='pretrain')")
    print("    dataset = XCADTestDataset(root_dir)")
    print("\nExample for plotting a sample:"
          "    plot_dataset_sample(dataset, index=0, transform=Transforms(training=True))")