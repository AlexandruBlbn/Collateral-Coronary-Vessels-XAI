# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

**Paper:** Randall Balestriero, Yann LeCun (2025) — arXiv:2511.08544  
**Code:** https://github.com/rbalestr-lab/lejepa  
**Published:** November 2025, FAIR at Meta / Brown University / NYU

---

## 1. Summary of the Paper

### Core Problem
Joint-Embedding Predictive Architectures (JEPAs) learn representations by predicting embeddings of one view from another view of the same data. However, JEPAs suffer from **representation collapse** — the encoder maps all inputs to identical or low-dimensional outputs, making the representations useless. Current solutions rely on ad-hoc heuristics: stop-gradient, teacher-student networks with EMA, asymmetric architectures, feature whitening, and delicate hyperparameter balancing.

### Key Theoretical Contribution
The paper proves that the **isotropic Gaussian distribution** is the unique optimal embedding distribution for foundation models. This is established through rigorous analysis of:
- **Linear probing** (Lemmas 1-2): Anisotropic embeddings amplify both bias (with Tikhonov regularization) and variance of linear probe estimators.
- **Nonlinear probing** (Theorem 1): For k-NN and kernel regression probes, isotropic Gaussian uniquely minimizes integrated squared bias under scalar covariance constraints.

### SIGReg: Sketched Isotropic Gaussian Regularization
SIGReg enforces the isotropic Gaussian distribution constraint using a novel hypothesis-testing framework:

1. **Directional projection**: Instead of comparing full multivariate distributions (quadratic complexity), project embeddings onto `M` random unit-norm directions and compare univariate distributions.
2. **Epps-Pulley characteristic function test**: The preferred test statistic compares empirical characteristic functions (Fourier transforms of density) against the target Gaussian CF. Key advantages over alternatives:
   - **Moment-based tests** (Jarque-Bera): Finite moments are insufficient to guarantee distributional match (Theorem 3); gradients scale as `O(k)` making training unstable.
   - **CDF-based tests** (Cramér-von Mises, Anderson-Darling): Require sorting (`O(N log N)`), breaking embarrassingly parallel SGD and multi-GPU scaling; non-differentiable operations.
   - **Epps-Pulley**: Naturally differentiable, bounded gradients and curvature (Theorem 4), DDP-friendly via `all_reduce`, `O(N)` complexity.
3. **Beating curse of dimensionality**: With sufficient Sobolev smoothness `alpha` of the embedding density, `M = O(K)` directions suffice (Theorem 5); SGD resampling of directions at each step provides additional coverage.

### LeJEPA Objective
```
Loss = (1 - lambda) * Prediction_Loss + lambda * SIGReg
```
Where:
- **Prediction loss**: MSE between embeddings of global views (context) and local views (target), computed as `||mean(global_embeddings) - local_embedding||^2`.
- **SIGReg**: Epps-Pulley test on all projected embedding tokens.
- **lambda**: Single trade-off hyperparameter; authors recommend `lambda = 0.05`.

### Key Properties
- **No stop-gradient** — collapse prevented by SIGReg alone.
- **No teacher-student** — single encoder, no EMA.
- **No predictor network** — direct embedding comparison (though optional predictor can be added, providing small ViT boost via SWA effect).
- **Linear time/memory** — `O(N)` with batch size `N`.
- **~50 lines of core PyTorch code** — algorithms 1 and 2 in the paper.
- **Architecture-agnostic** — works on ResNets, ViTs, ConvNeXts, Swin Transformers, EfficientNets (validated on 60+ architectures).
- **Stable across hyperparameters** — batch sizes 128-1024, view counts 2-10, embedding dimensions 64-1024 all work.
- **Training loss correlates with downstream performance** — Spearman correlation ~94% between LeJEPA training loss and linear probe accuracy (with `alpha=0.4` scaling, reaches 99%).

### Empirical Results
- ViT-H/14: 79% top-1 on ImageNet-1K (frozen backbone, linear probe).
- ViT-Large/14: 75% with only 100 pretraining epochs.
- ConvNeXtV2-H: 78.5% online linear probe.
- In-domain pretraining (Galaxy10, Food101) outperforms DINOv2/DINOv3 transfer learning.
- 1.8B ViT-g trained stably without heuristics.
- Emergent semantic segmentation from PCA of last-layer features.

---

## 2. Key Insights Relevant to the Project

### Insight 2.1: Collapse Prevention Without Heuristics for Medical Imaging
The project's domain — coronary X-ray angiography — has limited labeled data. SSL pretraining is critical. LeJEPA eliminates the need for complex collapse-prevention mechanisms, making it ideal for medical imaging where:
- Domain-specific pretraining is essential (ImageNet-pretrained models don't transfer well to X-rays).
- Training stability with small datasets is paramount.
- Hyperparameter tuning budgets are limited.

The project already implements dense LeJEPA pretraining via [`engine/lejepa_pretrain.py`](engine/lejepa_pretrain.py:1). However, **the implementation is currently broken** — it imports from `zoo.jepa_models` and `zoo.sigreg` which do not exist as files.

### Insight 2.2: Isotropic Gaussian Embeddings Minimize Downstream Risk
For the vessel segmentation downstream task, LeJEPA's theoretical guarantee that isotropic Gaussian embeddings minimize worst-case downstream risk is directly applicable. The proof covers:
- **Linear probes** (used for vessel/non-vessel classification from frozen features).
- **k-NN probes** (used in the project's validation at [`engine/lejepa_validate.py`](engine/lejepa_validate.py:304)).

The current k-NN validation framework is perfectly aligned with Theorem 1's analysis — the feature quality directly determines nearest-neighbor retrieval quality.

### Insight 2.3: SIGReg With Frangi-Enhanced Input
The project uses 2-channel input: [CLAHE grayscale, Frangi vesselness]. LeJEPA's SIGReg needs to operate on the projected token space, not the input space. Key consideration: Frangi responses produce naturally anisotropic distributions (vessel pixels are sparse, background dominates). SIGReg should help produce well-conditioned token distributions despite this input anisotropy.

### Insight 2.4: Training Loss as Model Selection Signal
LeJEPA's training loss correlates strongly with downstream performance (Figure 10-11 in paper). This is especially valuable for medical imaging where labeled validation data is scarce. The project can use the training loss trajectory for:
- Hyperparameter selection without labels.
- Early stopping decisions.
- Architecture comparison without downstream evaluation.

The recommended `alpha=0.4` scaling factor for maximizing Spearman correlation should be adopted.

### Insight 2.5: In-Domain Pretraining Over Transfer Learning
LeJEPA demonstrates that domain-specific SSL pretraining (e.g., on Galaxy10) outperforms frontier model transfer learning (DINOv2/v3). For coronary angiography, this suggests:
- Pretraining directly on ARCADE/XA-170K data with LeJEPA should outperform using pretrained DINO models.
- The Frangi vesselness channel provides domain-specific prior knowledge.

### Insight 2.6: Implementation Simplicity
The SIGReg implementation (Algorithm 1) requires only 17 trapezoidal quadrature points, 1024 random projection directions, and an integration domain of `[-5, 5]`. These defaults are shown to be robust across architectures and datasets.

---

## 3. Potential Implementation Approaches Inspired by the Paper

### Approach 3.1: Complete Missing SIGReg Implementation
**Priority: CRITICAL.** The files [`zoo/sigreg.py`](zoo/sigreg.py) and [`zoo/jepa_models.py`](zoo/jepa_models.py) are imported but do not exist.

**SIGReg module** should implement Algorithm 1 from the paper:
```
Input: x (N, K) tensor, global_step, num_slices=256
1. Sample M random unit-norm directions (synced across GPUs via seed)
2. Compute quadrature points t in [-5, 5] (17 points)
3. Compute theoretical CF: exp(-0.5 * t^2) for N(0,1)
4. Project: x_proj = x @ directions  (N, M)
5. Compute empirical CF: mean(exp(i * x_proj * t))
6. Weighted L2 distance with Gaussian weight exp(-t^2)
7. Integrate via trapezoidal rule, multiply by N
```

**DenseLeJepaModel** should implement:
- Encoder backbone (ViT/Swin/ConvNeXt).
- Projection head (Linear -> LayerNorm -> Linear).
- Optional predictor (transformer-based, returning dense token predictions).
- Integration with box-based crop prediction (global crop -> predict local crop tokens).

### Approach 3.2: Implement Epps-Pulley Test Variants for Domain-Specific Tuning
The paper evaluates multiple test statistics (Extended Jarque-Bera, Cramér-von Mises, Watson, Anderson-Darling, Epps-Pulley). While Epps-Pulley is recommended, medical X-ray images may benefit from exploration of:
- **Moment matching** for explicit variance/invariance/covariance control (connecting to VICReg).
- **CDF-based tests** if distribution shapes in vessel projections are highly non-Gaussian.

The theoretical framework (Theorem 3) warns against pure moment matching — only Epps-Pulley fully identifies the Gaussian distribution.

### Approach 3.3: Adopt LeJEPA's Prediction Loss Simplification
The paper's prediction loss (Equation 7) simplifies to:
```
Loss = ||mean(embedding_of_global_views) - embedding_of_view_v'||^2
```
This is ALREADY IMPLEMENTED in [`engine/lejepa_pretrain.py`](engine/lejepa_pretrain.py:476) as `F.mse_loss(pred_dense, tgt_flat)`. The key insight is that **all views predict the centroid of global views**, not individual global views. This is more stable than pairwise matching.

### Approach 3.4: Leverage Training Loss for Model Selection
Implement the `alpha=0.4` scaling of the training loss for model selection:
```python
scaled_loss = loss / (lambda ** alpha)
```
This gives ~99% Spearman correlation with downstream performance. Use this for:
- Architecture search across ViT-Swin-ConvNeXt for vessel features.
- Learning rate and weight decay selection.
- Early stopping without validation labels.

### Approach 3.5: Dense Token Prediction for Vessel Segmentation
The current implementation in [`engine/lejepa_pretrain.py`](engine/lejepa_pretrain.py:461-470) already predicts dense tokens from context (global view) to target (local view). This aligns with the "Dense Prediction" concept. Key enhancement: **apply SIGReg separately to context AND target tokens** (already done at line 479-486).

### Approach 3.6: Cross-Architecture Validation
LeJEPA works across 60+ architectures. The project should validate pretraining on:
- SwinV2 (already used in SimMIM).
- ConvNeXtV2 (lightweight, good for medical imaging).
- ViT variants (for attention-based interpretability).

### Approach 3.7: No Predictor Training Variant
The paper shows that the predictor network is not necessary to prevent collapse. Removing it:
- Reduces parameters by ~30-50%.
- Simplifies the training loop.
- May slightly reduce ViT performance (mitigated by SWA).
- For ConvNeXt/ResNet backbones, the predictor provides no benefit.

This is relevant if compute budget is limited for medical imaging pretraining.

### Approach 3.8: SIGReg Regularization Strength Tuning
The paper recommends `lambda = 0.05` as default. For medical X-ray data with strong local structure (vessels):
- Higher `lambda` may be needed if Frangi channel causes embedding anisotropy.
- Monitor SIGReg loss alongside prediction loss during training.
- The 2D loss plane (SIGReg vs Prediction, Figure 10) reveals optimal trade-off fronts.

---

## 4. Code Architecture Notes

### Existing Implementation Status
| Component | File | Status |
|-----------|------|--------|
| DenseLeJepaModel | [`zoo/jepa_models.py`](zoo/jepa_models.py) | **MISSING** — imported but file doesn't exist |
| SIGRegLoss | [`zoo/sigreg.py`](zoo/sigreg.py) | **MISSING** — imported but file doesn't exist |
| LeJepaDenseDataset | [`data/dataloader.py`](data/dataloader.py:18) | EXISTS — well-implemented with Frangi caching |
| FrangiCache | [`data/frangi_cache.py`](data/frangi_cache.py) | EXISTS — precomputation pipeline |
| Training loop | [`engine/lejepa_pretrain.py`](engine/lejepa_pretrain.py:328) | EXISTS — full AMP, checkpointing, TensorBoard |
| k-NN Validation | [`engine/lejepa_validate.py`](engine/lejepa_validate.py:332) | EXISTS — DINOv3-style dense feature evaluation |
| Backbone factory | [`zoo/backbones.py`](zoo/backbones.py:57) | EXISTS — ViT, Swin, ConvNeXt, EfficientNet |

### Missing Implementation Priority
1. **`zoo/sigreg.py`** — 50 lines per Algorithm 1; core of LeJEPA collapse prevention.
2. **`zoo/jepa_models.py`** — DenseLeJepaModel with encoder, projection, and predictor.
3. Integration testing with existing training loop.
