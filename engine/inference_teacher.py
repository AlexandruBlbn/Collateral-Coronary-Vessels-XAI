import os
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def get_inference_transform(size=512):
    return albu.Compose([
        albu.Resize(size, size),
        albu.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])


@torch.no_grad()
def predict_image(model, image_path, transform, device, tta=True):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    if tta:
        preds = []
        for aug, deaug in [
            (lambda x: x, lambda x: x),
            (lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1])),
            (lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2])),
        ]:
            aug_input = aug(input_tensor)
            logits = model(aug_input)
            logits = deaug(logits)
            preds.append(torch.sigmoid(logits))

        pred = torch.stack(preds).mean(dim=0)
    else:
        logits = model(input_tensor)
        pred = torch.sigmoid(logits)

    pred = pred.squeeze().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model .pth file")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of input images")
    parser.add_argument("--output_dir", type=str, default="pseudolabels",
                        help="Output directory for pseudolabels")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tta", action="store_true", default=True,
                        help="Use test-time augmentation")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    transform = get_inference_transform(args.size)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([
        str(p) for p in Path(args.input_dir).rglob("*")
        if p.suffix.lower() in exts
    ])
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} images")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in tqdm(image_paths, desc="Generating pseudolabels"):
        pred = predict_image(model, path, transform, device, tta=args.tta)
        rel = Path(path).relative_to(args.input_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst.with_suffix(".png")), pred)

    print(f"Done. Pseudolabels saved to {args.output_dir}")


if __name__ == "__main__":
    main()
