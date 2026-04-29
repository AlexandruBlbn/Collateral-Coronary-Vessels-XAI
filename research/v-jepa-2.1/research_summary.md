# V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning

**Paper:** Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, Adrien Bardes (2026) — arXiv:2603.14482  
**Code:** https://github.com/facebookresearch/vjepa2  
**Published:** March 2026, FAIR at Meta / Universidad de Zaragoza

---

## 1. Summary of the Paper

### Core Problem
V-JEPA (Video Joint-Embedding Predictive Architecture) learns strong global video representations through latent-space mask-denoising, but produces **poor dense (per-patch) features**. PCA visualizations reveal noisy, fragmented feature maps with no coherent spatial structure. Dense tasks (segmentation, depth estimation, object tracking) perform poorly with frozen V-JEPA encoders (22.2 mIoU on ADE20K, 0.682 RMSE on NYUv2).

### Root Cause Analysis
The original V-JEPA loss is applied **only to masked tokens** — context (visible) tokens have no incentive to encode local spatial information. The predictor uses context tokens as global aggregators (similar to register tokens in DINOv2), sacrificing spatial grounding. This is a form of implicit collapse for dense features: global understanding works, but local structure is lost.

### Four Key Innovations

#### 1. Dense Predictive Loss (Section 2.3.1)
Apply the self-supervised loss to **both masked AND context (visible) tokens**:
```
L_dense = L_predict (masked tokens) + L_ctx (context tokens)
```
Where `L_ctx` uses **distance-weighted** coefficients:
```
lambda_i = lambda / sqrt(d_min(i, M))
```
Here `d_min(i, M)` is the spatio-temporal distance (in blocks) between context token `i` and its nearest masked token. This weighting:
- Emphasizes patches near masked regions → enforces local continuity.
- Prevents context tokens from becoming global aggregators.
- Achieves 22.2 → 33.9 mIoU on ADE20K, 0.682 → 0.473 RMSE on NYUv2.
- **Naive equal weighting hurts global tasks** (72.8 → 62.5 on SSv2) → needs distance weighting + warmup schedule.

#### 2. Deep Self-Supervision (Section 2.3.2)
Apply the JEPA loss **hierarchically at multiple intermediate encoder layers**:
1. Concatenate outputs from 4 equally-spaced encoder blocks.
2. Lightweight MLP fuses multi-level representations and reduces dimensionality.
3. Predictor produces 4 outputs corresponding to 4 encoder levels.
4. Both `L_predict` and `L_ctx` applied at each level.

**Key effect**: Information flows from intermediate layers to final layers. This:
- Restores global task performance (SSv2: 62.5 → 72.1).
- Further improves dense tasks (ADE20K: 33.9 → 38.6).
- Eliminates need for multi-scale evaluation at probe time.
- Acts as an implicit regularizer that prevents local information loss in deep layers.

#### 3. Multi-Modal Tokenizer (Section 2.3.4)
Separate patch embeddings for images and videos:
- **Images**: 2D convolution (16x16).
- **Videos**: 3D convolution (16x16x2 tubelet).
- **Modality embedding**: Learnable token added to encoder/predictor input, explicitly encoding image vs video.

Benefits:
- No temporal duplication of images (V-JEPA 2 duplicated images as 16-frame static videos).
- Computational efficiency.
- Better dense task performance (ADE20K: 40.8 → 41.4).

#### 4. Data and Model Scaling (Sections 2.3.3, 2.3.5)
- **VisionMix-163M**: Combines LVD-142M (curated images) + video datasets with rebalanced sampling weights (more motion-rich content).
- **Model scaling**: ViT-L (300M) → ViT-g (1B) → ViT-G (2B).
- **Cool-down phase**: 12K iterations at higher resolution (images 512x512, videos 64 frames at 384x384), decaying learning rate.

### Ablation Progression (ViT-L baseline)
| Component | ADE20K mIoU | SSv2 Acc |
|-----------|-------------|----------|
| V-JEPA 2 (baseline) | 22.2 | 72.8 |
| + Context Loss | 33.8 | 62.5 |
| + Deep Self-Supervision | 38.6 | 72.1 |
| + VisionMix data | 40.8 | 72.6 |
| + Multi-modal Tokens | 41.4 | 72.6 |
| + Model Scaling (ViT-G) | 47.1 | 76.1 |
| + Cool-down | 47.9 | 77.7 |

### SOTA Results
| Task | Metric | V-JEPA 2.1 ViT-G |
|------|--------|-------------------|
| NYUv2 Depth | 0.307 RMSE | SOTA (beats DINOv3 7B) |
| SSv2 Action Recognition | 77.7% | SOTA |
| Ego4D STA | 7.71 mAP | SOTA (+35% vs previous) |
| EK100 Action Anticipation | 40.8 Rec@5 | SOTA |
| Robot Grasping | 80% success | +20% vs V-JEPA 2 |
| TartanDrive Navigation | 5.687 ATE | 10x faster planning |
| DAVIS VOS | 69.0 J&F | Near SOTA |
| ADE20K Segmentation | 47.9 mIoU | Competitive |

### Distillation
ViT-G (2B) distilled to ViT-L (300M) and ViT-B (80M). Key differences from pretraining: frozen teacher instead of EMA, no deep self-supervision, 12-block predictor (down from 24). Distilled ViT-L nearly matches ViT-G on most tasks.

---

## 2. Key Insights Relevant to the Project

### Insight 2.1: Context Token Supervision for Vessel Feature Maps
The project's LeJEPA pretraining in [`engine/lejepa_pretrain.py`](engine/lejepa_pretrain.py:461-470) predicts **target (local view) tokens from context (global view) tokens**, but the loss is computed only between predicted and actual target tokens. The **context tokens themselves are never directly supervised** for local structure quality.

**Direct applicability**: The V-JEPA 2.1 finding that context tokens become "global aggregators" without explicit supervision applies identically to the LeJEPA setup. The current implementation computes:
```python
inv_loss = F.mse_loss(pred_dense, tgt_flat)  # line 476
```
This only supervises the predictor's output against target tokens. The context tokens (ctx_tokens) produced by the encoder have no direct constraint to preserve spatial structure.

**Proposed fix**: Add a context self-supervision term where context tokens are also predicted and compared to their actual values, with distance-based weighting to emphasize tokens near target regions.

### Insight 2.2: Deep Self-Supervision for Multi-Scale Vessel Features
Coronary vessels exist at multiple scales — from large main arteries to fine collateral vessels. V-JEPA 2.1's deep self-supervision (loss at multiple encoder layers) ensures that:
- **Shallow layers** capture fine-grained local vessel structure.
- **Deep layers** capture global coronary tree topology.

The current LeJEPA implementation only uses the final encoder output. Adding intermediate layer supervision would:
- Improve fine vessel detection (collateral vessels are thin, low-contrast).
- Preserve global vessel tree structure.
- Eliminate the need for multi-scale feature extraction at probe time.

### Insight 2.3: Multi-Modal Tokenizer for Dual-Channel Input
The project uses 2-channel input: [CLAHE grayscale, Frangi vesselness]. This is analogous to V-JEPA 2.1's image+video multi-modal setup. Instead of simply concatenating channels:
- **Separate patch embeddings** for grayscale and Frangi channels.
- **Modality embedding** to condition processing on the Frangi signal.
- This could improve the model's ability to distinguish between structural (grayscale) and vessel-enhanced (Frangi) information.

### Insight 2.4: Distance-Weighted Context Loss for Vessel Continuity
The weighted context loss:
```
lambda_i = lambda / sqrt(d_min(i, M))
```
is designed to enforce **local continuity between masked and context regions**. For vessel segmentation:
- Vessels are continuous tubular structures.
- Masking in pretraining breaks vessel continuity.
- Distance-weighted loss encourages the model to maintain vessel continuity predictions across mask boundaries.
- This is especially relevant for collateral vessels which are thin and prone to fragmentation.

### Insight 2.5: Cool-Down Phase for High-Resolution Vessel Details
V-JEPA 2.1's cool-down phase (higher resolution + learning rate decay) provides disproportionate benefits for depth estimation (0.365 → 0.307 RMSE). For coronary X-rays:
- High-resolution fine-tuning could dramatically improve detection of small collateral vessels.
- The recommended schedule: 12K iterations at 2x spatial resolution with LR decay from 6e-4 to 1e-6.

### Insight 2.6: Model Distillation for Clinical Deployment
Clinical deployment requires efficient models. V-JEPA 2.1's distillation protocol shows that:
- A 2B teacher can distill to an 80M student with minimal performance loss.
- The distilled ViT-B achieves ViT-L (trained from scratch) performance.
- Distillation uses the same JEPA loss without deep self-supervision.
- EMA of student weights serves as the final model (not used in loss computation).

This is directly applicable for creating lightweight models for real-time catheterization lab use.

### Insight 2.7: Emergent Dense Features from Predictive Objectives
V-JEPA 2.1 demonstrates that **predictive objectives alone** (without reconstruction or contrastive losses) can produce high-quality dense features when:
1. The loss is applied to ALL tokens (not just masked ones).
2. Multiple encoder layers are supervised.
3. Distance-based weighting provides spatial grounding.

This validates the JEPA paradigm for medical image feature learning and suggests that reconstruction-based pretraining (SimMIM in [`engine/MIM_Train.py`](engine/MIM_Train.py:205)) may be unnecessary if dense prediction is properly implemented.

---

## 3. Potential Implementation Approaches Inspired by the Paper

### Approach 3.1: Dense Predictive Loss for LeJEPA (HIGH PRIORITY)
Modify the existing LeJEPA pretraining in [`engine/lejepa_pretrain.py`](engine/lejepa_pretrain.py:328) to add context token supervision:

```python
# Current (line 476):
inv_loss = F.mse_loss(pred_dense, tgt_flat)

# Proposed addition:
# 1. Also predict context tokens from context tokens (self-prediction)
ctx_pred = model.predictor(ctx_for_pred, ctx_boxes_for_pred, ctx_boxes_for_pred, L_ctx)
ctx_loss = F.mse_loss(ctx_pred, ctx_for_pred)

# 2. Distance-weighted combination
# Weight context loss by inverse distance to nearest target token
d_min = compute_min_distance_to_target_tokens(ctx_boxes, target_boxes)
ctx_weights = 1.0 / torch.sqrt(d_min + 1e-6)
weighted_ctx_loss = (ctx_loss * ctx_weights).mean()

# 3. Combined dense loss
inv_loss = (1 - alpha) * tgt_loss + alpha * weighted_ctx_loss
```

Key implementation details:
- Distance should be computed in the **original image coordinate space** using the box coordinates already stored in `global_boxes` and `local_boxes`.
- Warmup `alpha` from epochs 50-100 as in V-JEPA 2.1.
- Use `lambda = 0.5` for weighting (paper's recommendation for video; 0.7 for images).

### Approach 3.2: Deep Self-Supervision in LeJEPA (MEDIUM PRIORITY)
Extend the encoder to output intermediate features:

```python
# DenseLeJepaModel modification:
def encode(self, x, return_intermediate=True):
    if return_intermediate:
        # Return features from multiple encoder stages
        feats = self.backbone(x)  # list of [B, C_i, H_i, W_i]
        # Project each level
        tokens_list = []
        for i, f in enumerate(feats):
            t = f.flatten(2).permute(0, 2, 1)
            t = self.level_projs[i](t)  # per-level projection
            tokens_list.append(t)
        return tokens_list  # List of [B, L_i, D]
    # ... existing single-level logic
```

The config already supports deep supervision parameters:
```python
deep_supervision=model_cfg.get("deep_supervision", False),
deep_supervision_out_indices=model_cfg.get("deep_supervision_out_indices", (2, 3)),
```
These parameters are passed to [`DenseLeJepaModel`](zoo/jepa_models.py) but the model file doesn't exist yet.

### Approach 3.3: Multi-Modal Patch Embedding for Grayscale+Frangi (MEDIUM PRIORITY)
Instead of concatenating grayscale and Frangi at the input level:
```python
# Current: 2-channel input [gray, frangi]
# Proposed: Separate embeddings
class DualPatchEmbed(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768):
        self.gray_proj = nn.Conv2d(1, embed_dim, patch_size, patch_size)
        self.frangi_proj = nn.Conv2d(1, embed_dim, patch_size, patch_size)
        self.modality_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        gray, frangi = x[:, :1], x[:, 1:2]
        gray_tokens = self.gray_proj(gray).flatten(2).permute(0, 2, 1)
        frangi_tokens = self.frangi_proj(frangi).flatten(2).permute(0, 2, 1)
        # Concatenate along sequence dimension or add modality embedding
        tokens = torch.cat([gray_tokens, frangi_tokens], dim=1)
        tokens = tokens + self.modality_embed  # tell the model which is which
        return tokens
```
This preserves the distinct statistical properties of grayscale vs Frangi channels and allows the model to learn modality-specific processing.

### Approach 3.4: High-Resolution Cool-Down Phase (LOW PRIORITY, HIGH IMPACT)
Add a second training phase after standard LeJEPA pretraining:
```python
# After main training loop (e.g., 300 epochs at 256x256)
# Cool-down phase (12K iterations)
cfg["data"]["global_size"] = 384  # Or 448 for better vessel detail
cfg["data"]["local_size"] = 160
cfg["optimizer"]["lr"] = 6e-4  # Higher initial LR for cool-down
cfg["optimizer"]["final_lr"] = 1e-6
# Train for ~50 epochs with cosine decay to final_lr
```
This is particularly important for detecting collateral vessels which may be < 1mm in diameter and require high spatial resolution.

### Approach 3.5: Context Token Distance-Weighted SIGReg
Apply different SIGReg strengths to context tokens based on their distance to predicted regions:
```python
# Current: uniform SIGReg on all tokens
sigreg_loss_val = sigreg_loss_fn(all_tokens)

# Proposed: distance-weighted SIGReg
distances = compute_token_distances(ctx_tokens, tgt_tokens)  # in embedding space
weights = 1.0 / torch.sqrt(distances + 1e-6)
weighted_sigreg = sigreg_loss_fn.weighted(all_tokens, weights)
```
This ensures stronger Gaussian regularization on tokens that need to support fine-grained predictions.

### Approach 3.6: Distillation Pipeline for Clinical Deployment
Train a lightweight UNeXt (already implemented in [`zoo/unext.py`](zoo/unext.py:150)) as a student of the LeJEPA-pretrained ViT encoder:
1. Pretrain ViT-G/L with LeJEPA on ARCADE.
2. Distill to UNeXt using JEPA loss (cosine similarity between student and teacher token embeddings).
3. UNeXt is already optimized for medical image segmentation (72x fewer params than UNet).
4. Frozen EMA of student as final model.

### Approach 3.7: Unified Dense Prediction Architecture
Combine V-JEPA 2.1's dense prediction with the project's vessel-specific components:
1. **Encoder**: SwinV2/ViT with deep self-supervision (4 intermediate layers).
2. **Dense loss**: Distance-weighted context + mask prediction.
3. **SIGReg**: Applied per-level to prevent collapse at all depths.
4. **Frangi integration**: Via multi-modal tokenizer with separate patch embedding.
5. **Validation**: k-NN retrieval (existing in [`engine/lejepa_validate.py`](engine/lejepa_validate.py:332)) extended to multiple layers.

---

## 4. Code Architecture Integration

### Files to Create/Modify

| File | Action | Content |
|------|--------|---------|
| [`zoo/jepa_models.py`](zoo/jepa_models.py) | **CREATE** | DenseLeJepaModel with deep supervision support, dual patch embed option |
| [`zoo/sigreg.py`](zoo/sigreg.py) | **CREATE** | SIGRegLoss per Algorithm 1 from LeJEPA paper |
| [`engine/lejepa_pretrain.py`](engine/lejepa_pretrain.py:328) | MODIFY | Add context loss, deep supervision, cool-down phase |
| [`data/dataloader.py`](data/dataloader.py:18) | MODIFY | Option for separate grayscale/Frangi paths |
| [`engine/lejepa_validate.py`](engine/lejepa_validate.py:332) | MODIFY | Multi-level k-NN evaluation |

### Training Recipe Inspired by V-JEPA 2.1

| Phase | Epochs | Resolution | LR | Key Changes |
|-------|--------|------------|-----|-------------|
| Primary | 135K iters / ~300 epochs | 256x256 | 5.25e-4 constant | Standard LeJEPA + context loss |
| Warmup | Epochs 50-100 | 256x256 | Ramp to 5.25e-4 | Gradual context loss weight increase |
| Cool-down | 12K iters / ~50 epochs | 384x384 / 512x512 | 6e-4 → 1e-6 | Higher resolution + LR decay |

### Hyperparameter Recommendations from Paper

| Parameter | V-JEPA 2.1 Value | Project Adaptation |
|-----------|------------------|--------------------|
| Context loss lambda | 0.5 (video), 0.7 (image) | 0.5 for X-ray (static but structured) |
| Distance weighting | `lambda / sqrt(d_min)` | Same — preserves vessel continuity |
| Deep supervision levels | 4 equally-spaced | 3-4 levels for SwinV2/ViT |
| EMA coefficient | 0.99925 | Same |
| Weight decay | 0.04 | Same |
| Predictor blocks | 24 (12 for distillation) | 12-24 depending on model size |
| Image batch size (global) | 2304 | Scale to available GPU memory |
| Cool-down resolution | 384-512 | 384 (balance detail vs memory) |

---

## 5. Relationship to LeJEPA

V-JEPA 2.1 and LeJEPA are complementary frameworks from the same FAIR/Meta group:

| Aspect | LeJEPA | V-JEPA 2.1 |
|--------|--------|------------|
| **Primary goal** | Prevent representation collapse | Improve dense feature quality |
| **Core mechanism** | SIGReg (isotropic Gaussian constraint) | Dense prediction on all tokens |
| **Collapse prevention** | Explicit distribution matching | Implicit via context loss + EMA |
| **Use of stop-gradient** | None (removed) | Yes (on target encoder) |
| **Teacher-student** | None (single encoder) | Yes (EMA teacher) |
| **Deep supervision** | Not explored | Core contribution |
| **Modality handling** | Single input | Multi-modal tokenizer |
| **Architecture** | Agnostic (60+ validated) | ViT-focused |

**Synergy opportunity**: Combine LeJEPA's SIGReg (principled collapse prevention) with V-JEPA 2.1's dense prediction and deep supervision. This would produce a framework that:
1. Prevents collapse without heuristics (SIGReg).
2. Produces high-quality dense features (context loss + deep supervision).
3. Works across architectures (LeJEPA's architecture agnosticism).
4. Has a single hyperparameter (lambda from LeJEPA) plus context loss weight.
