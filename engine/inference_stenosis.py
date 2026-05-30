import argparse, contextlib, csv, sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, Tuple
import cv2, numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn

ENGINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = ENGINE_DIR.parent
sys.path.extend([str(ENGINE_DIR), str(ROOT_DIR)])

from trainv2 import VesselNetV2, VesselNetV2EfficientEncoder
try: import segmentation_models_pytorch as smp
except Exception: smp = None
VesselNetV3 = _tv3e = None
try: from trainv3 import VesselNetV3
except Exception as e: _tv3e = e
try: from zoo.unext import UNeXt_S
except Exception: UNeXt_S = None

# ── image helpers ──────────────────────────────────────────────────────────
class InferencePreprocessor:
    def __init__(self, img_size=512):
        self.img_size = int(img_size)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    def __call__(self, image):
        orig = image.size
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(image, dtype=np.uint8)
        c1 = self.clahe.apply(arr)
        c2 = cv2.morphologyEx(arr, cv2.MORPH_TOPHAT, self.kernel)
        c3 = cv2.morphologyEx(arr, cv2.MORPH_BLACKHAT, self.kernel)
        blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=10)
        c4 = cv2.addWeighted(arr, 4.0, blur, -4.0, 128)
        t = torch.from_numpy(np.stack([c1, c2, c3, c4], -1)).permute(2, 0, 1).float() / 255.0
        return t, orig

def _load_gray(p): return Image.open(p).convert("L")
def _gray_tensor(im, sz):
    r = im.resize((sz, sz), Image.BILINEAR)
    return torch.from_numpy(np.array(r, dtype=np.float32)/255.0).unsqueeze(0)

# ── inference ──────────────────────────────────────────────────────────────
def _extract_logits(o):
    if isinstance(o, dict):
        if "seg_logits" in o: return o["seg_logits"]
        if "seg" in o: return o["seg"]
        if len(o) == 1: return next(iter(o.values()))
        raise KeyError(f"Unknown output keys: {list(o.keys())}")
    return o

def _autocast(dev, use_amp, precision):
    if not use_amp or dev.type != "cuda": return contextlib.nullcontext()
    dt = torch.bfloat16 if precision.lower() == "bfloat16" else torch.float16
    return torch.amp.autocast("cuda", enabled=True, dtype=dt)

@torch.no_grad()
def predict_probs(model, inp, device, tta, use_amp, precision):
    if not tta:
        with _autocast(device, use_amp, precision):
            return torch.sigmoid(_extract_logits(model(inp)))
    preds = []
    for aug, deaug in [(lambda x: x, lambda x: x),
                       (lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-1])),
                       (lambda x: torch.flip(x, [-2]), lambda x: torch.flip(x, [-2]))]:
        with _autocast(device, use_amp, precision):
            logits = _extract_logits(model(aug(inp)))
        preds.append(torch.sigmoid(deaug(logits)))
    return torch.stack(preds, 0).mean(0)

# ── checkpoint loading (LoRA injection) ────────────────────────────────────
class _LoRAModule(nn.Module):
    """Replaces a timm Conv2d with: conv(x) + lora_B(lora_A(x)).
    Submodule layout matches the LoRA checkpoint:
        .conv    –  original Conv2d
        .lora_A  –  low-rank projection
        .lora_B  –  output projection
    """
    def __init__(self, conv: nn.Conv2d, la_w: torch.Tensor, lb_w: torch.Tensor):
        super().__init__()
        self.conv = conv
        r, k = la_w.shape[0], int(la_w.shape[2])
        s = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
        self.lora_A = nn.Conv2d(conv.in_channels, r, k, stride=s, padding=k//2, bias=False)
        self.lora_B = nn.Conv2d(r, conv.out_channels, 1, bias=False)
        self.lora_A.weight.data.copy_(la_w)
        self.lora_B.weight.data.copy_(lb_w)

    def forward(self, x):
        return self.conv(x) + self.lora_B(self.lora_A(x))

def _strip_prefix(sd, pfx):
    ks = list(sd.keys())
    if ks and all(k.startswith(pfx) for k in ks):
        return {k[len(pfx):]: v for k, v in sd.items()}
    return sd

def _maybe_strip(sd):
    for pfx in ("module.", "backbone.", "model."):
        sd = _strip_prefix(sd, pfx)
    return sd

def _inject_loras(model, sd):
    """For each LoRA A/B pair in *sd*, wrap the model Conv2d with _LoRAModule.

    Checkpoint layout (example for conv_dw):
        encoder.blocks.3.0.conv_dw.conv.weight   (base conv weight)
        encoder.blocks.3.0.conv_dw.lora_A.weight (LoRA A)
        encoder.blocks.3.0.conv_dw.lora_B.weight (LoRA B)

    The LoRA prefix is "encoder.blocks.3.0.conv_dw." (no ".conv." in it).
    The base weight key has ".conv." inserted: "encoder.blocks.3.0.conv_dw.conv.weight".
    In the model, "encoder.blocks.3.0.conv_dw" IS the Conv2d directly.
    """
    injected = 0
    for key in sorted(sd.keys()):
        if not key.endswith("lora_A.weight"):
            continue
        prefix = key[:-len("lora_A.weight")]          # e.g. "encoder.blocks.3.0.conv_dw."
        b_key = prefix + "lora_B.weight"
        if b_key not in sd:
            continue

        # The module path is the prefix without trailing dot
        mod_path = prefix.rstrip(".")                  # e.g. "encoder.blocks.3.0.conv_dw"

        # Walk to the module at mod_path
        parts = mod_path.split(".")
        obj = model
        for p in parts:
            try:                    obj = getattr(obj, p)
            except AttributeError:  obj = obj[int(p)]

        # Case 1: obj IS the Conv2d (e.g. conv_dw, conv_stem, se.conv_reduce)
        if isinstance(obj, nn.Conv2d):
            grandparent = model
            for p in parts[:-1]:
                try:                    grandparent = getattr(grandparent, p)
                except AttributeError:  grandparent = grandparent[int(p)]
            child_name = parts[-1]
            wrapper = _LoRAModule(obj, sd[key].clone(), sd[b_key].clone())
            try:                    setattr(grandparent, child_name, wrapper)
            except AttributeError:  grandparent[int(child_name)] = wrapper
            injected += 1
            continue

        # Case 2: obj is a parent module with a "conv" child Conv2d
        if hasattr(obj, "conv") and isinstance(obj.conv, nn.Conv2d):
            wrapper = _LoRAModule(obj.conv, sd[key].clone(), sd[b_key].clone())
            obj.conv = wrapper
            injected += 1
            continue

    return injected

def load_checkpoint(model, ckpt_path, strict=False, lora_scale=None,
                    lora_alpha=None, skip_mismatch=False, dw_fix=None):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    if not isinstance(state, dict):
        raise RuntimeError("Bad checkpoint format")

    state = _maybe_strip(state)
    if any("lora_" in k for k in state):
        n = _inject_loras(model, state)
        print(f"[INFO] {n} LoRA modules injected (exact training forward pass).")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"[WARN] Missing keys: {missing}")
    if unexpected: print(f"[WARN] Unexpected keys: {unexpected}")

# ── model construction ─────────────────────────────────────────────────────
def build_model(args):
    mn = args.model_name
    if mn == "VesselNetV2":
        return VesselNetV2(in_chans=args.in_chans, num_classes=args.num_classes,
            dims=tuple(int(x) for x in args.dims.split(",")),
            depths=tuple(int(x) for x in args.depths.split(",")),
            drop_path_rate=args.drop_path_rate)
    if mn in {"VesselNetV2EfficientEncoder", "VesselNetV3EfficientNet"}:
        return VesselNetV2EfficientEncoder(in_chans=args.in_chans, num_classes=args.num_classes,
            encoder_name=args.encoder_name, encoder_pretrained=args.encoder_pretrained,
            encoder_img_size=args.encoder_img_size, drop_path_rate=args.drop_path_rate)
    if mn == "VesselNetV3":
        raise RuntimeError("VesselNetV3 not available")
    if mn in {"GlobalRefinementUNet", "SMP_Unet"}:
        if smp is None: raise RuntimeError("smp required")
        return smp.Unet(encoder_name=args.encoder_name, encoder_weights=None,
            in_channels=args.in_chans, classes=args.num_classes)
    if mn == "UNeXt_S":
        if UNeXt_S is None: raise RuntimeError("UNeXt_S not available")
        return UNeXt_S(in_channels=args.in_chans, num_classes=args.num_classes,
            base_channels=args.unext_base_channels,
            depths=[int(x) for x in args.unext_depths.split(",")],
            mlp_ratio=args.unext_mlp_ratio, drop_rate=args.unext_drop_rate,
            attention=args.unext_attention, use_checkpoint=False)
    raise ValueError(f"Unknown model: {mn}")

# ── helpers ────────────────────────────────────────────────────────────────
def _prepare_mask(mask, img_size, thr):
    if mask.ndim == 2: mask = mask.unsqueeze(0)
    if mask.shape[-2:] != (img_size, img_size): raise ValueError("mask size")
    if thr is not None: mask = (mask > float(thr)).float()
    return mask

def _build_input_tensor(image, prep, sz, mode, mask=None, image4=None, mask_threshold=None):
    if mode == "image4":
        return image4 if image4 is not None else prep(image)[0]
    if mode == "image1":
        return _gray_tensor(image, sz)
    if mode in {"image4+mask", "image4*mask"}:
        i4 = image4 if image4 is not None else prep(image)[0]
        m = _prepare_mask(mask, sz, mask_threshold)
        return torch.cat([i4, m], 0) if mode == "image4+mask" else i4 * m
    if mode == "image1+mask":
        return torch.cat([_gray_tensor(image, sz), _prepare_mask(mask, sz, mask_threshold)], 0)
    if mode == "mask_only":
        return _prepare_mask(mask, sz, mask_threshold)
    raise ValueError(f"Unknown mode: {mode}")

def _ensure_channels(t, exp, label):
    if t.shape[0] != exp: raise ValueError(f"{label}: {t.shape[0]}ch vs {exp}")

def _parse_color(v, default):
    if not v: return default
    return tuple(max(0, min(255, int(p.strip()))) for p in v.split(","))

def _rgb_image(im): return np.stack([np.array(im.convert("L"), dtype=np.uint8)]*3, -1)
def _norm_prob(pm):
    if pm.dtype != np.float32: pm = pm.astype(np.float32)
    if pm.max() > 1.0: pm /= 255.0
    return np.clip(pm, 0, 1)
def _overlay_mask(base, mask, color, alpha):
    o = base.astype(np.float32)
    o[mask.astype(bool)] = (1-alpha)*o[mask.astype(bool)] + alpha*np.array(color, np.float32)
    return o

def _overlay_vessel_stenosis(img, vp, sp, vthr, sthr, alpha, vc, sc):
    base = _rgb_image(img)
    vm = _norm_prob(vp) > vthr
    sm = _norm_prob(sp) > sthr
    o = _overlay_mask(base, vm, vc, alpha)
    return _overlay_mask(o.astype(np.uint8), sm, sc, alpha).astype(np.uint8)

def iter_images(in_dir, max_images):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if in_dir.is_file() and in_dir.suffix.lower() in exts: return [in_dir]
    ps = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in exts)
    return ps[:int(max_images)] if max_images else ps

def _load_patient_filter(csv_path, pid):
    if not csv_path or not pid: return None
    p = Path(csv_path)
    if not p.is_file(): return None
    pid = str(pid); al = set()
    with open(p, "r") as f:
        for row in csv.DictReader(f):
            if str(row.get("patient_id","")) != pid: continue
            fk = row.get("file") or row.get("file_name") or row.get("image") or row.get("image_path")
            if fk: al.add(Path(fk).name)
    return al if al else None

def _filter_paths(paths, pid, csv):
    if not pid: return list(paths)
    al = _load_patient_filter(csv, pid)
    if al is not None: return [p for p in paths if p.name in al]
    tok = str(pid)
    return [p for p in paths if tok in p.stem or tok in p.name]

def _resolve_prefixed(args, prefix, name):
    v = getattr(args, f"{prefix}_{name}", None)
    return v if v is not None else getattr(args, name)

def _make_model_args(args, prefix):
    return SimpleNamespace(
        model_name=_resolve_prefixed(args, prefix, "model_name"),
        in_chans=_resolve_prefixed(args, prefix, "in_chans"),
        num_classes=_resolve_prefixed(args, prefix, "num_classes"),
        dims=args.dims, depths=args.depths, drop_path_rate=args.drop_path_rate,
        encoder_name=_resolve_prefixed(args, prefix, "encoder_name"),
        encoder_pretrained=args.encoder_pretrained,
        encoder_img_size=_resolve_prefixed(args, prefix, "encoder_img_size"),
        unext_base_channels=args.unext_base_channels,
        unext_depths=args.unext_depths, unext_mlp_ratio=args.unext_mlp_ratio,
        unext_drop_rate=args.unext_drop_rate, unext_attention=args.unext_attention)

def _save_mask_and_prob(out, pm, thr, save_prob):
    mask = (pm > float(thr)).astype(np.uint8) * 255
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), mask)
    if save_prob:
        cv2.imwrite(str(out.with_name(out.stem+"_prob.png")),
                    np.clip(pm*255, 0, 255).astype(np.uint8))

# ── main ───────────────────────────────────────────────────────────────────
def _run_single(args):
    if not args.checkpoint: raise ValueError("--checkpoint required")
    
    if args.device:
        dev = torch.device(args.device)
    else:
        cuda_available = torch.cuda.is_available()
        dev = torch.device("cuda" if cuda_available else "cpu")
        if not cuda_available:
            print("[WARN] CUDA not available to PyTorch. Falling back to CPU. Check your drivers and torch version.")
            
    print(f"Device: {dev}")
    model = build_model(args)
    load_checkpoint(model, args.checkpoint, strict=args.strict, lora_scale=args.lora_scale,
                    lora_alpha=args.lora_alpha, skip_mismatch=args.skip_mismatch, dw_fix=args.dw_fix)
    model.to(dev).eval()
    prep = InferencePreprocessor(args.img_size)
    paths = _filter_paths(list(iter_images(Path(args.input_dir), args.max_images)),
                          args.patient_id, args.patient_csv)
    if not paths: raise RuntimeError("No images")
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
    for p in tqdm(paths, desc="Running"):
        img = _load_gray(p)
        i4, osz = prep(img)
        t = _build_input_tensor(image=img, prep=prep, sz=args.img_size, mode=args.input_mode, image4=i4)
        _ensure_channels(t, args.in_chans, "Model")
        probs = predict_probs(model, t.unsqueeze(0).to(dev, non_blocking=True),
                              device=dev, tta=args.tta, use_amp=args.use_amp, precision=args.precision)
        pm = cv2.resize(probs.squeeze().detach().cpu().numpy(), osz, cv2.INTER_LINEAR)
        _save_mask_and_prob((od / p.relative_to(Path(args.input_dir))).with_suffix(".png"),
                            pm, args.threshold, args.save_prob)
    print("Done.")

def _run_pipeline(args):
    missing = [n for n in ("vessel_checkpoint","refiner_checkpoint","stenosis_checkpoint") if not getattr(args, n)]
    if missing: raise ValueError(f"Missing: {missing}")
    
    if args.device:
        dev = torch.device(args.device)
    else:
        cuda_available = torch.cuda.is_available()
        dev = torch.device("cuda" if cuda_available else "cpu")
        if not cuda_available:
            print("[WARN] CUDA not available to PyTorch. Falling back to CPU.")
            
    print(f"Device: {dev}")
    va = _make_model_args(args, "vessel"); ra = _make_model_args(args, "refiner"); sa = _make_model_args(args, "stenosis")
    vm = build_model(va); rm = build_model(ra); sm = build_model(sa)
    for m, ck in [(vm, args.vessel_checkpoint), (rm, args.refiner_checkpoint), (sm, args.stenosis_checkpoint)]:
        load_checkpoint(m, ck, strict=args.strict, lora_scale=args.lora_scale,
                        lora_alpha=args.lora_alpha, skip_mismatch=args.skip_mismatch, dw_fix=args.dw_fix)
    vm.to(dev).eval(); rm.to(dev).eval(); sm.to(dev).eval()
    indir = Path(args.input_dir); od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    vd = od/"vessel"; rd = od/"refiner"; sd = od/"stenosis"
    prep = InferencePreprocessor(args.img_size)
    paths = _filter_paths(list(iter_images(indir, args.max_images)), args.patient_id, args.patient_csv)
    if not paths: raise RuntimeError("No images")
    ithr = args.intermediate_threshold or args.threshold
    for p in tqdm(paths, desc="Running pipeline"):
        img = _load_gray(p); i4, osz = prep(img)
        # vessel
        vi = _build_input_tensor(image=img, prep=prep, sz=args.img_size, mode=args.vessel_input_mode, image4=i4)
        _ensure_channels(vi, va.in_chans, "Vessel")
        vp = predict_probs(vm, vi.unsqueeze(0).to(dev, non_blocking=True), dev, args.tta, args.use_amp, args.precision)
        vpm = vp.squeeze().detach().cpu()
        # refiner
        ri = _build_input_tensor(image=img, prep=prep, sz=args.img_size, mode=args.refiner_input_mode, mask=vpm, image4=i4, mask_threshold=args.input_mask_threshold)
        _ensure_channels(ri, ra.in_chans, "Refiner")
        rp = predict_probs(rm, ri.unsqueeze(0).to(dev, non_blocking=True), dev, args.tta, args.use_amp, args.precision)
        rpm = rp.squeeze().detach().cpu()
        # stenosis
        si = _build_input_tensor(image=img, prep=prep, sz=args.img_size, mode=args.stenosis_input_mode, mask=rpm, image4=i4, mask_threshold=args.input_mask_threshold)
        _ensure_channels(si, sa.in_chans, "Stenosis")
        sp = predict_probs(sm, si.unsqueeze(0).to(dev, non_blocking=True), dev, args.tta, args.use_amp, args.precision)
        spm = sp.squeeze().detach().cpu().numpy()
        rp_rel = p.relative_to(indir)
        if args.save_intermediate:
            _save_mask_and_prob((vd/rp_rel).with_suffix(".png"), cv2.resize(vpm.numpy(), osz, cv2.INTER_LINEAR), ithr, args.save_prob)
            _save_mask_and_prob((rd/rp_rel).with_suffix(".png"), cv2.resize(rpm.numpy(), osz, cv2.INTER_LINEAR), ithr, args.save_prob)
        _save_mask_and_prob((sd/rp_rel).with_suffix(".png"), cv2.resize(spm, osz, cv2.INTER_LINEAR), args.threshold, args.save_prob)
        if args.save_overlay:
            ovd = Path(args.overlay_dir) if args.overlay_dir else od/"overlay"
            oo = (ovd/rp_rel).with_suffix(".png"); oo.parent.mkdir(parents=True, exist_ok=True)
            ov = _overlay_vessel_stenosis(img, cv2.resize(vpm.numpy(), osz, cv2.INTER_LINEAR),
                cv2.resize(spm, osz, cv2.INTER_LINEAR),
                args.overlay_vessel_threshold or ithr, args.overlay_stenosis_threshold or args.threshold,
                float(args.overlay_alpha), _parse_color(args.vessel_color, (0,255,0)), _parse_color(args.stenosis_color, (255,0,0)))
            cv2.imwrite(str(oo), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    print(f"Done. Pipeline outputs saved to {od}")

def main():
    p = argparse.ArgumentParser(description="Inference for stenosis models")
    p.add_argument("--pipeline", action="store_true", default=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--input_dir", type=str, default="data/ARCADE/processed/stenoza/data")
    p.add_argument("--output_dir", type=str, default="outputs/pipeline_inference")
    p.add_argument("--model_name", type=str, default="VesselNetV2EfficientEncoder")
    p.add_argument("--in_chans", type=int, default=4); p.add_argument("--num_classes", type=int, default=1)
    p.add_argument("--img_size", type=int, default=512); p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tta", action="store_true", default=False); p.add_argument("--save_prob", action="store_true", default=False)
    p.add_argument("--max_images", type=int, default=None); p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--skip_mismatch", action="store_true", default=False)
    p.add_argument("--device", type=str, default=None); p.add_argument("--patient_id", type=str, default="26")
    p.add_argument("--patient_csv", type=str, default="data/ARCADE/uq/summary_stenosis.csv")
    p.add_argument("--input_mode", type=str, default="image4", choices=["image4","image1","image4+mask","image4*mask","image1+mask","mask_only"])
    p.add_argument("--use_amp", action="store_true", default=True); p.add_argument("--precision", type=str, default="bfloat16")
    p.add_argument("--lora_scale", type=float, default=None); p.add_argument("--lora_alpha", type=float, default=None)
    p.add_argument("--dw_fix", type=str, default="none", choices=["none","mean","diag"])
    p.add_argument("--dims", type=str, default="64,128,256,512"); p.add_argument("--depths", type=str, default="2,2,3,2")
    p.add_argument("--drop_path_rate", type=float, default=0.0)
    p.add_argument("--encoder_name", type=str, default="efficientnetv2_s")
    p.add_argument("--encoder_pretrained", action="store_true", default=False)
    p.add_argument("--encoder_img_size", type=int, default=512)
    p.add_argument("--vessel_checkpoint", type=str, default="Demo/best_model.pth")
    p.add_argument("--refiner_checkpoint", type=str, default="Demo/model_refiner_v2/best_refiner.pth")
    p.add_argument("--stenosis_checkpoint", type=str, default="Demo/best_stenosis.pth")
    p.add_argument("--vessel_model_name", type=str, default="VesselNetV2EfficientEncoder")
    p.add_argument("--refiner_model_name", type=str, default="VesselNetV2EfficientEncoder")
    p.add_argument("--stenosis_model_name", type=str, default="VesselNetV2EfficientEncoder")
    p.add_argument("--vessel_in_chans", type=int, default=4)
    p.add_argument("--refiner_in_chans", type=int, default=5)
    p.add_argument("--stenosis_in_chans", type=int, default=4)
    p.add_argument("--vessel_encoder_name", type=str, default=None)
    p.add_argument("--refiner_encoder_name", type=str, default=None)
    p.add_argument("--stenosis_encoder_name", type=str, default=None)
    p.add_argument("--vessel_encoder_img_size", type=int, default=None)
    p.add_argument("--refiner_encoder_img_size", type=int, default=None)
    p.add_argument("--stenosis_encoder_img_size", type=int, default=None)
    p.add_argument("--vessel_input_mode", type=str, default="image4", choices=["image4","image1"])
    p.add_argument("--refiner_input_mode", type=str, default="image4+mask",
                   choices=["image4","image1","image4+mask","image4*mask","image1+mask","mask_only"])
    p.add_argument("--stenosis_input_mode", type=str, default="image4",
                   choices=["image4","image1","image4+mask","image4*mask","image1+mask","mask_only"])
    p.add_argument("--input_mask_threshold", type=float, default=0.45)
    p.add_argument("--intermediate_threshold", type=float, default=0.45)
    p.add_argument("--save_intermediate", action="store_true", default=True)
    p.add_argument("--save_overlay", action="store_true", default=True)
    p.add_argument("--overlay_dir", type=str, default=None); p.add_argument("--overlay_alpha", type=float, default=0.5)
    p.add_argument("--overlay_vessel_threshold", type=float, default=None)
    p.add_argument("--overlay_stenosis_threshold", type=float, default=None)
    p.add_argument("--vessel_color", type=str, default="0,255,0"); p.add_argument("--stenosis_color", type=str, default="255,0,0")
    p.add_argument("--unext_base_channels", type=int, default=128)
    p.add_argument("--unext_depths", type=str, default="2,1,1"); p.add_argument("--unext_mlp_ratio", type=float, default=4.0)
    p.add_argument("--unext_drop_rate", type=float, default=0.0)
    p.add_argument("--no_unext_attention", action="store_false", dest="unext_attention", default=True)
    args = p.parse_args()
    _run_pipeline(args) if args.pipeline else _run_single(args)

if __name__ == "__main__":
    main()
