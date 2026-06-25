# VasoJEPA v2 — Coronary Angiography Latent Space Pretraining

<div align="center">

**EMA-free JEPA with Manifold-Aware Regularization for Coronary Vessel SSL**

[![Status](https://img.shields.io/badge/status-in%20development-yellow)](https://github.com)
[![Target](https://img.shields.io/badge/target-MICCAI%202027-blue)](https://miccai.org)
[![GPU](https://img.shields.io/badge/GPU-8GB%2B-green)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

</div>

---

## Overview

**VasoJEPA v2** is a novel self-supervised learning framework for coronary X-ray angiography. It builds on the JEPA (Joint Embedding Predictive Architecture) paradigm with three key innovations:

1. **EMA-free architecture** — No teacher network. Collapse is prevented by the CGLT regularizer (from UR-JEPA), cutting parameters and memory by ~50%.
2. **Manifold-aligned regularization** — Replaces SIGReg's isotropic Gaussian target with CGLT's uniform n-rectifiability, matching the intrinsically low-dimensional structure of coronary vessels.
3. **Learned vessel prior (LDS)** — A Latent Denoising Score branch bootstrapped by a Robust Consensus Prior (Frangi + Sato + Hessian majority vote) that self-corrects during training.

> **Target venue**: MICCAI 2027 / Medical Image Analysis  
> **Hardware**: Single GPU ≥ 8 GB VRAM

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  VasoJEPA v2 (EMA-free)                      │
├──────────────────────────────────────────────────────────────┤
│  Input: Angiogram 224×224 → 196 patches                     │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  ONLINE ENCODER (ViT-S/16) — the ONLY encoder          │  │
│  │  Layer 4 → f₄  |  Layer 8 → f₈  |  Layer 12 → f₁₂    │  │
│  └────────────────────────────────────────────────────────┘  │
│         │                                                    │
│  ═══════════════ THREE BRANCHES ════════════════════════     │
│                                                              │
│  BRANCH 1: Dense Predictor (main learning signal)            │
│  ├─ Targets: f₄, f₈, f₁₂ with stop_gradient (.detach())     │
│  ├─ Context: 50% visible patches from f₁₂                    │
│  ├─ Predictor: 4-block ViT → ALL 196 positions               │
│  └─ Loss: Vessel-weighted MSE (weight = 1 + β·score)         │
│                                                              │
│  BRANCH 2: CGLT Regularizer (collapse prevention)            │
│  ├─ Projector MLP: 384 → 2048 → 2048 → D=32                 │
│  ├─ CGLT Carleson loss + AD anchor (n=7, K=5)                │
│  └─ Modulated: weight = (1 - α·vessel_score)                 │
│                                                              │
│  BRANCH 3: Latent Denoising Score — LDS (vessel prior)       │
│  ├─ Denoiser: MLP with T=50 diffusion steps                  │
│  ├─ Vessel head: MLP → vessel_score ∈ [0,1]                  │
│  └─ Guidance: soft BCE vs Robust Consensus Prior (annealed)  │
│                                                              │
│  L = 1.0·L_dense + 0.02·10³·(L_CGLT + 0.1·L_AD) + 0.2·L_lds│
└──────────────────────────────────────────────────────────────┘
```

### Why EMA-free?

| Resource | With EMA Teacher | EMA-free (ours) | Saving |
|----------|:---------------:|:---------------:|:------:|
| Parameters | ~44M (2× ViT-S) | ~22M (1× ViT-S) | **~50%** |
| VRAM @batch=64 | ~12 GB | **~6 GB** | Fits 8 GB GPU |
| Anti-collapse | Implicit (slow teacher) | Explicit (CGLT) | More principled |

---

## Key Innovations

| Contribution | Description | Novelty |
|-------------|-------------|:-------:|
| **C1** | First application of UR-JEPA / CGLT to medical imaging | High |
| **C2** | EMA-free dense JEPA for coronary angiograms | Medium-High |
| **C3** | Anatomy-adaptive CGLT (vessel-modulated n-rectifiability) | **Very High** |
| **C4** | LDS branch with Robust Consensus Prior + guidance annealing | High |
| **C5** | Diagnosis: SIGReg fails on sparse tubular structures | Medium |

---

## Project Structure

```
.
├── README.md                          # This file
├── .gitignore
│
├── docs/                              # Design documents & architecture specs
│   ├── ur_jepa_analysis.md            # UR-JEPA analysis & integration strategy
│   ├── vasojepa_diagrams.md           # Architecture diagrams & walkthrough
│   ├── vasojepa_project_guide.md      # Complete implementation guide
│   └── vasojepa_v2_robust_prior.md    # Robust Consensus Prior design
│
├── XA-SSL-REPO/                       # Main codebase (VasoMIM + VasoJEPA)
│   ├── vasojepa/                      # VasoJEPA model package
│   │   ├── model.py                   # Full model (⚠️ needs EMA-free redesign)
│   │   ├── encoder.py                 # ViT-S/16 encoder (layers 4, 8, 12)
│   │   ├── predictor.py               # Dense predictor (4-block ViT)
│   │   ├── lds.py                     # Latent Denoising Score branch
│   │   ├── sigreg.py                  # SIGReg regularizer (ablation baseline)
│   │   ├── dataset.py                 # Dataset loader
│   │   └── __init__.py
│   ├── robust_prior.py                # Frangi+Sato+Hessian consensus prior (RCP)
│   ├── frangi_filter.py               # Frangi filter implementation
│   ├── main_pretrain.py               # Pretraining entry point
│   ├── engine_pretrain.py             # Pretraining engine
│   ├── models_vmae.py                 # VMAE model variants
│   ├── test_model.py                  # Unit tests
│   ├── test_robust_prior.py           # Prior tests
│   ├── pretrain_vasomim.sh            # VasoMIM pretraining script
│   ├── pretrain_simmim.sh             # SimMIM pretraining script
│   ├── segmodel/                      # Segmentation models (UNeXt)
│   ├── util/                          # Utilities (LR schedules, pos embed)
│   └── src/                           # Figures
│
├── engine/                            # Legacy training & inference scripts
│   ├── trainv2.py / trainv3.py        # Training variants
│   ├── inference_stenosis.py          # Stenosis inference
│   └── inference_teacher.py           # Teacher model inference
│
├── scripts/                           # Fine-tuning scripts
│   └── finetune_vessel.py             # Vessel segmentation fine-tuning
│
├── utils/                             # Shared utilities
│   └── helpers.py
│
├── zoo/                               # Model architectures
│   ├── mim.py                         # Masked Image Modeling
│   └── unext.py                       # UNeXt architecture
│
├── data/                              # Dataset & dataloaders
│   ├── dataloader.py
│   └── ARCADE/                        # ARCADE dataset
│
├── checkpoints/                       # Model weights
│   ├── vasojepa/                      # VasoJEPA checkpoints
│   ├── eff_unet/                      # EfficientUNet checkpoints
│   └── point_unet/                    # PointUNet checkpoints
│
├── XA-170K/                           # Dataset subset
│   └── dataset/
│
└── configs/                           # (future) configuration files
```

---

## Implementation Status

| Component | File | Status |
|-----------|------|:------:|
| Robust Consensus Prior (RCP) | `XA-SSL-REPO/robust_prior.py` | ✅ Done & tested |
| ViT-S/16 Encoder | `XA-SSL-REPO/vasojepa/encoder.py` | ✅ Done |
| Dense Predictor | `XA-SSL-REPO/vasojepa/predictor.py` | ✅ Done |
| Latent Denoising Score | `XA-SSL-REPO/vasojepa/lds.py` | ✅ Done |
| SIGReg (ablation baseline) | `XA-SSL-REPO/vasojepa/sigreg.py` | ✅ Done |
| Dataset loader | `XA-SSL-REPO/vasojepa/dataset.py` | ✅ Done |
| Unit tests | `XA-SSL-REPO/test_model.py` | ✅ Done |
| **CGLT Regularizer** | `XA-SSL-REPO/vasojepa/cglt.py` | ❌ **NEXT** |
| **EMA-free model redesign** | `XA-SSL-REPO/vasojepa/model.py` | ❌ **NEXT** |
| **Training script** | `XA-SSL-REPO/train.py` | ❌ Not started |
| **Ablation framework** | `XA-SSL-REPO/ablation_runner.py` | ❌ Not started |

---

## Setup

### Requirements

```bash
pip install torch torchvision timm==1.0.20
pip install scikit-image opencv-python numpy
```

### Hardware

- **Training**: 1× GPU ≥ 8 GB VRAM (EMA-free design)
- **Inference**: CPU or GPU

### Dataset

The project uses the **XA-170K** dataset (171,478 X-ray angiograms from CADICA, SYNTAX, XCAD, CoronaryDominance). A subset is included in `XA-170K/dataset/`.

For the full dataset, see the [XA-170K Hugging Face repo](https://huggingface.co/datasets/waha2000huang/XA-170K).

---

## Quick Start

> ⚠️ Training pipeline is under active development. See `docs/vasojepa_project_guide.md` for the full implementation roadmap.

### Precompute Robust Consensus Priors

```bash
cd XA-SSL-REPO
python robust_prior.py --input_dir ../XA-170K/dataset --output_dir ../data/priors
```

### Run tests

```bash
cd XA-SSL-REPO
python test_model.py
python test_robust_prior.py
```

---

## CGLT Hyperparameters (from UR-JEPA)

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| `proj_dim` D | 32 | Projector output dimension |
| `target_dim` n | 7 | Target intrinsic dimension (plateau: n ∈ [6,10]) |
| `dyadic_scales` K | 5 | Number of dyadic scales |
| `λ_AD` | 0.1 | AD-regularity anchor weight |
| `scale` s | 10³ | CGLT loss scaling |
| `λ` | 0.02 | Regularization weight |

---

## Ablation Plan

| Run | Config | Purpose |
|-----|--------|---------|
| A0 | I-JEPA vanilla (EMA, no SIGReg) | JEPA baseline |
| A1 | LeJEPA vanilla (SIGReg, no EMA) | LeJEPA baseline |
| A2 | VasoJEPA-Lite (SIGReg + Frangi static) | Static prior baseline |
| A3 | A0 + Dense multi-scale prediction | Dense benefit |
| A4 | A3 + CGLT (no vessel modulation) | CGLT baseline |
| **A5** | **A3 + CGLT + vessel modulation** | **Proposed method** |
| A6 | A5 + LDS (no Frangi guidance) | LDS without prior |
| **A7** | **Full VasoJEPA v2 (A5 + LDS + RCP)** | **Full method** |

---

## References

- **UR-JEPA**: Le, T.M. "Uniform Rectifiability for Joint Embedding Predictive Architecture." arXiv:2606.01443, 2026.
- **VasoMIM**: Huang, D.X. et al. "Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis." arXiv:2602.11536, 2026.
- **VasoMIM-v1**: Huang, D.X. et al. "VasoMIM: Vascular anatomy-aware masked image modeling for vessel segmentation." AAAI 2026.
- **I-JEPA**: Assran, M. et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." CVPR 2023.

---

## License

This project builds upon [MAE](https://github.com/facebookresearch/mae) (CC-BY-NC 4.0). The XA-170K dataset is licensed under CC-BY-NC 4.0.

---

*Project under active development. See `docs/` for detailed design documents and implementation guides.*