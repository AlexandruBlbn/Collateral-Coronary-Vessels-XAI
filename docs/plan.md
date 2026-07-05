# VasoJEPA v2 — Pre-Training and Implementation Plan

## 1. Project Abstraction & Current State

- **Dataloader (Ready)**: The dataset implementations are complete in [data.py](file:///D:/Coronary%20Angiography%20Latent%20Space%20Pretraining/data/data.py) for both pretraining (loading over 170k grayscale coronary angiography frames) and finetuning (loading ARCADE vessel segmentations).
- **Environment & Helpers (Ready)**: Core modules, seeds, and standard imports are configured in [helpers.py](file:///D:/Coronary%20Angiography%20Latent%20Space%20Pretraining/utils/helpers.py).
- **VasoJEPA Model (Pending)**: The core modules (`encoder`, `cglt` regularizer, `predictor`, and `lds`) and training/ablation loops need to be created.

---

## 2. Phase-by-Phase Task List

### Phase 1: Robust Consensus Prior (RCP) Generation
Because the Latent Denoising Score (LDS) branch uses a soft vesselness map to guide representation learning, you need prior scores.
- [x] **Task 1.1**: `utils/vessels_prior.py` — Sato + Meijering consensus with morphological opening and small-object removal. Includes FOV mask (auto-detects black X-ray intensifier border and erodes 15px inward to kill gradient false positives).
- [x] **Task 1.2**: `precompute_priors.py` — multiprocessing batch script. Outputs: `float16` `.npy` at 14×14 patch resolution (~67 MB for all 170k images). Mirrors image paths: `data/pretrain/dataset/cadica/1.png` → `data/pretrain/priors/cadica/1.npy`. Also includes a verification and plotting utility `visualize_prior.py` for debugging.
- [x] **Task 1.3**: Update `data/data.py` to load precomputed RCP maps during pre-training.

### Phase 2: Core Model Components (`vasojepa/`)
Create the sub-package directory `vasojepa` and implement its modular architecture.
- [x] **Task 2.1**: Initialize `vasojepa/__init__.py`.
- [x] **Task 2.2**: Implement `vasojepa/encoder.py` using a `timm` ViT-S/16 wrapper to extract features at layers 4, 8, and 12.
- [ ] **Task 2.3**: Implement `vasojepa/predictor.py` with a 4-block lightweight dense predictor that projects visible patches to all 196 patch positions.
- [ ] **Task 2.4**: Implement `vasojepa/cglt.py` containing the uniform n-rectifiability Carleson loss and its MLP projection heads.
- [ ] **Task 2.5**: Implement `vasojepa/lds.py` containing the Latent Denoising Score (diffusion denoiser + vessel classification head).

### Phase 3: Unified EMA-Free Model
- [ ] **Task 3.1**: Create `vasojepa/model.py` to bind all components together without a target encoder, using `detach()` on online features for prediction targets.
- [ ] **Task 3.2**: Create a smoke test script `engine/test.py` to run a batch forward pass and confirm gradients flow as expected.

### Phase 4: Training Pipeline
- [ ] **Task 4.1**: Create `train.py` implementing mixed precision, cosine annealing learning rate scheduler with linear warmup, and checkpoint saving.

### Phase 5: Downstream Fine-Tuning & Evaluation
- [ ] **Task 5.1**: Implement `eval_linear_probe.py` to evaluate frozen features on vessel classification.
- [ ] **Task 5.2**: Implement `finetune.py` to train a UNet decoder on top of pre-trained weights for the 126 labeled images, tracking Dice and clDice.
- [ ] **Task 5.3**: Implement `ablation_runner.py` to run baseline comparisons (A0–A7) for publication.
