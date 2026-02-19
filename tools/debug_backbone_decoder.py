#!/usr/bin/env python3
import argparse
import math
import traceback
import torch
import timm


def load_model(weights_path, device):
    weights = torch.load(weights_path, map_location='cpu')
    if 'model_state_dict' in weights:
        weights = weights['model_state_dict']

    new_weights = {}
    for k, v in weights.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
        if k.startswith('backbone.'):
            k = k.replace('backbone.', '')
        new_weights[k] = v

    model = timm.create_model('coatnet_1_rw_224', pretrained=False, in_chans=1, num_classes=0, global_pool='')
    model.load_state_dict(new_weights, strict=False)
    return model.to(device)


def print_tensor_info(x, prefix=''):
    print(f"{prefix} type={type(x)}")
    if isinstance(x, torch.Tensor):
        print(f"{prefix} tensor dim={x.dim()} shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
        try:
            print(f"{prefix} stats mean={float(x.mean()):.6f} std={float(x.std()):.6f} min={float(x.min()):.6f} max={float(x.max()):.6f} nan={torch.isnan(x).any().item()}")
        except Exception:
            print(f"{prefix} stats: could not compute (maybe non-numeric)")
    elif isinstance(x, (list, tuple)):
        print(f"{prefix} list/tuple len={len(x)}")
        for i, el in enumerate(x):
            print_tensor_info(el, prefix=f"{prefix}[{i}]")
    else:
        print(f"{prefix} repr={repr(x)}")


def try_decoder_reshape(x, in_channels):
    print("Simulating decoder input handling...")
    while isinstance(x, (list, tuple)):
        print(" was list/tuple -> taking first element")
        x = x[0]

    if not isinstance(x, torch.Tensor):
        print(" after unwrapping not a tensor:", type(x))
        return

    print_tensor_info(x, prefix=" after unwrap:")

    if x.dim() == 3:
        B, D1, D2 = x.shape
        C_expected = in_channels

        if D1 == C_expected:
            print("Detected (B,C,L) format")
            B, C, L = x.shape
            H = W = int(math.sqrt(L))
            y = x.reshape(B, C, H, W)
        elif D2 == C_expected:
            print("Detected (B,L,C) format")
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            y = x.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            print("Neither dim matches expected channels; reporting dims")
            print(f"D1={D1}, D2={D2}, expected C={C_expected}")
            y = None
            # try safe reshape attempt
            try:
                H = W = int(math.sqrt(D2))
                y = x.reshape(B, D1, H, W)
                print("Reshaped by assuming D2 is L -> shape", tuple(y.shape))
            except Exception as e:
                print("Reshape attempt failed:", e)

    elif x.dim() == 4:
        print("Already 4D; pass-through")
        y = x
    else:
        print("Unsupported dim:", x.dim())
        y = None

    if y is not None:
        print_tensor_info(y, prefix=" after reshape:")
        # quick BatchNorm test
        try:
            bn = torch.nn.BatchNorm2d(y.shape[1]).to(y.device)
            out = bn(y)
            print("BatchNorm applied successfully; out shape", tuple(out.shape))
        except Exception as e:
            print("BatchNorm failed:", e)
    else:
        print("No reshaped tensor available")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='checkpoints/LeJepa_coatnet_detach/best_backbone.pth')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    device = torch.device(args.device)
    print('Loading model from', args.weights, 'onto', device)
    try:
        model = load_model(args.weights, device)
    except Exception:
        print('Failed to load weights; aborting')
        traceback.print_exc()
        return

    model.eval()
    imgs = torch.randn(args.batch, 1, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        feats = model(imgs)

    print('\nBackbone returned:')
    print_tensor_info(feats, prefix='backbone:')

    print('\nNow inspect what the decoder would receive and how reshaping behaves:\n')
    try_decoder_reshape(feats, in_channels=768)


if __name__ == '__main__':
    main()
