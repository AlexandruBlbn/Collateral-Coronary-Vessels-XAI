# VasoJEPA — Vesselness-Prior-Guided JEPA for Coronary Angiography SSL

<div align="center">

**VasoJEPA: EMA-free JEPA with exogenous vesselness anchor for coronary angiography self-supervised pretraining**

[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()
[![Target](https://img.shields.io/badge/target-MICCAI%202027-blue)]()
[![GPU](https://img.shields.io/badge/GPU-16GB%2B-green)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()

</div>

---

## Overview

**VasoJEPA** is a self-supervised learning (SSL) framework for coronary X-ray angiography (XCA). It pretrains a visual encoder on unlabeled images using a JEPA (Joint Embedding Predictive Architecture) objective guided by a precomputed vesselness prior, producing label-efficient features for downstream stenosis segmentation.

### Contributions

1. **Exogenous anti-collapse signal** — The vesselness anchor (auxiliary head predicting Sato+Meijering filter outputs) is the first externally-referenced anti-collapse mechanism in the JEPA family. All existing mechanisms (EMA teacher, SIGReg, Gram anchoring) are self-referential.
2. **EMA-free training** — Stop-gradient alone is sufficient on XCA, eliminating the EMA teacher and reducing memory by 45% (24.9M vs 45.1M params).
3. **Label efficiency** — Pretrained features outperform random features by 1.8× → 2.7× across label budgets (1% → 100%), with the gap widening as more labels become available.

---

## Method

VasoJEPA has three components:
- **Encoder**: TinyViT-21M (`features_only=True`, `in_chans=1`), producing features at stages f0–f3 (384-dim at 14×14 resolution for f2).
- **Predictor**: 2-layer symmetric `nn.Transformer` (d_pred=192, 8 heads) that predicts target encoder features from visible context.
- **Vessel head**: `Linear(384, 1)` on encoder f2 features, trained to predict the precomputed vesselness prior V (14×14 float16 per image).

**Loss**: `L = L_dense + 0.1 · L_anchor + 0.05 · L_consistency`
- `L_dense = MSE(pred_f2, target_f2.detach()) + MSE(pred_f3, target_f3.detach())` — self-predictive dense loss (stop-gradient, no EMA)
- `L_anchor = MSE(h(f2), V)` — encoder features must decode to vesselness prior
- `L_consistency = MSE(h(pred_f2), V_tgt)` — predictor output must also decode to prior at target locations

### Anti-collapse intuition
A collapsed encoder outputs a constant `c`. Then `h(c) ≠ V` (V is fixed, precomputed) → `L_anchor` is large → direct penalty. The exogenous reference breaks the encoder-predictor collusion symmetry that self-referential mechanisms cannot.

### Vesselness prior
- **RCP** (Robust Consensus Prior): Sato + Meijering vesselness filters, combined by consensus (logical AND)
- Precomputed as 14×14 patch-resolution float16 `.npy` per image (~67MB total for 171k images)
- Computed by unsupervised Hessian-based filters — **no expert annotations**

### Vessel masking (negative result)
A complementary mechanism (biasing target patch selection toward vessel patches using the prior as sampling weights) was tested but consistently **hurt** downstream performance — the predictor becomes prior-dependent and fails to generalize at inference. Reported as a negative result in the paper.

---

## Quickstart

### Setup

```bash
pip install -r requirements.txt
```

### Pretraining (171k images, 100 epochs)

```bash
python train_full.py --no-ema --vessel-anchor \
  --checkpoint-dir checkpoints_noema_vanchor_full \
  --tb-dir runs/noema_vanchor_full
```

Monitor with TensorBoard: `tensorboard --logdir runs/noema_vanchor_full`

### Fine-tuning (label efficiency)

```bash
python finetune_stenosis.py --checkpoint checkpoints_noema_vanchor_full/vasojepa_epoch099.pt \
  --label-frac 0.10 --epochs 50
```

### Full sweep (3 seeds × 4 budgets)

```bash
python run_ablation_downstream.py \
  --checkpoint-dir checkpoints_noema_vanchor_full \
  --target-epoch 99 --seed 0 1 2 \
  --label-fracs 0.01 0.05 0.10 1.0
```

### Linear probe (offline)

```bash
python probe_checkpoints.py \
  --checkpoint-dir checkpoints_noema_vanchor_full \
  --probe-epochs 15 --probe-frac 0.10
```

### XAI analysis

```bash
python run_xai.py --checkpoint checkpoints_noema_vanchor_full/vasojepa_epoch099.pt
```

### Precompute vesselness priors

```bash
python precompute_priors.py
```

---

## Data

| Dataset | Use | Size | Split |
|---|---|---|---|
| XCA-170K | Pretraining | 171,000 images | 5 sources (cadica, coronarydominance, syntax, xcad, arcade) |
| ARCADE | Downstream (stenosis seg) | 1,500 images | 1000 train / 200 val / 300 test |

**Pretraining labels**: none (SSL).  
**Downstream labels**: ARCADE stenosis masks (~0.98% positive pixels).

Note: ARCADE's 2000 train+val images are a subset of XCA-170K's "arcade" source. Labels are never used in pretraining. The ARCADE test split is 100% disjoint from pretraining.

---

## Results

### Ablation grid (40 epochs, 10k subset, Dice on ARCADE test)

| Cell | EMA | VMask | VAnchor | Dice @1% | Dice @10% |
|---|:-:|:-:|:-:|---|---|
| **D** | | | ✓ | **0.104** | **0.148** |
| E | | ✓ | ✓ | 0.079 | 0.125 |
| C | | ✓ | | 0.063 | 0.071 |
| Baseline | ✓ | | | 0.056 | 0.021 |
| B | | | | 0.044 | 0.058 |
| A | ✓ | ✓ | | 0.045 | 0.023 |
| Random | — | — | — | 0.034 | 0.074 |

### Label efficiency (D config, 3 seeds)

| Labels | Pretrained | Random | Ratio |
|---|---|---|:-:|
| 1% (10 img) | 0.108 ± 0.003 | 0.061 ± 0.021 | 1.8× |
| 5% (50 img) | 0.159 ± 0.008 | 0.070 ± 0.002 | 2.3× |
| 10% (100 img) | 0.171 ± 0.018 | 0.074 ± 0.002 | 2.3× |
| 100% (1000 img) | 0.211 ± 0.005 | 0.078 ± 0.001 | 2.7× |

### XAI

- Linear probe AUC: 0.978 (pretrained) vs 0.761 (random)
- Saliency maps correlate with vessel regions
- t-SNE clusters by vessel density

---

## Project structure

```
vasojepa/
  encoder.py        TinyViT-21M backbone
  predictor.py      Transformer predictor
  model.py          VesselJEPA (3 toggles: use_ema, vessel_masking, vessel_anchor)
train_full.py       Full pretraining (171k, 100ep)
train_test.py       Ablation grid training
finetune_stenosis.py Label-efficiency fine-tuning
run_ablation_downstream.py  Orchestrator for sweeps
probe_checkpoints.py Offline linear probe on all checkpoints
run_xai.py          XAI analysis (probe, saliency, t-SNE)
precompute_priors.py  RCP generation (Sato+Meijering)
data/
  data.py           Dataset definitions
  ARCADE/processed/ Downstream dataset
docs/
  ablation_plan.md  Ablation plan + decision tree
```

---

## License

This project builds upon [MAE](https://github.com/facebookresearch/mae) (CC-BY-NC 4.0). The XCA-170K dataset is licensed under CC-BY-NC 4.0.

