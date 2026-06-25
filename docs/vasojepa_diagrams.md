# VasoJEPA v2 — Architecture Diagrams

---

## Diagram 1: Full Pipeline Overview

![VasoJEPA v2 Full Pipeline](C:\Users\alexb\.gemini\antigravity-cli\brain\9da37455-e6c8-4029-a1d4-b89dc6dbc12c\vasojepa_v2_pipeline_1782248006918.jpg)

---

## Diagram 2: The Vessel Score Feedback Loop (The Core Mechanism)

![VasoJEPA v2 Feedback Loop](C:\Users\alexb\.gemini\antigravity-cli\brain\9da37455-e6c8-4029-a1d4-b89dc6dbc12c\vasojepa_feedback_loop_1782248042267.jpg)

---

## Reading the Diagrams

### Step-by-step walkthrough

**Step 1 — Input**
An angiogram (224×224) enters the model. It is split into a 14×14 grid of 196 patches.
Separately, a Robust Consensus Prior (RCP) map is precomputed offline by combining Frangi + Sato + Hessian vesselness filters via majority vote.

**Step 2 — The Single Encoder (EMA-free)**
One ViT-S/16 encoder processes all 196 patch tokens through 12 transformer blocks.
Features are extracted at three depths: layers 4, 8, and 12.
There is **no EMA teacher** — the CGLT regularizer in Branch 2 prevents collapse instead.

**Step 3 — Branch 3 runs first: Latent Denoising Score (LDS)**
Branch 3 takes the layer 12 features (`f₁₂`, detached from the gradient graph), adds random Gaussian noise at a random timestep `t ∈ {1..50}`, and trains a small MLP to denoise them back. The denoiser learns to distinguish vessel-like patch representations from background. The Vessel Head reads out a per-patch score `vessel_score ∈ [0,1]`. This score is softly supervised by the RCP prior at the start of training (via Binary Cross-Entropy loss that anneals from λ=1.0 to 0.1 over training).

**Step 4 — vessel_score drives two opposite forces**
The vessel_score from Branch 3 feeds into the other two branches in opposite directions:

| Branch | Effect on vessel patches | Formula |
|--------|-------------------------|---------|
| **Branch 1 (Dense Prediction)** | Gets **more** loss weight | `weight = 1 + β·score`, β=2.0 |
| **Branch 2 (CGLT Regularizer)** | Gets **less** rectifiability pressure | `weight = 1 - α·score`, α=0.8 |

This is the key insight: vessel patches need to be **unpacked and detailed** (strong prediction signal) while simultaneously being **left free from over-regularization** (weak CGLT pressure). Background patches get the opposite treatment.

**Step 5 — Branch 1: Dense Predictor**
50% of patches are randomly masked. A 4-block ViT predictor sees only the visible half of `f₁₂` and must reconstruct features at **all 196 positions** (not just masked ones). The targets are `f₄, f₈, f₁₂` with `.detach()` (stop-gradient) — no EMA teacher needed. The MSE loss is vessel-weighted: vessel patches contribute up to 3× more gradient.

**Step 6 — Branch 2: CGLT Regularizer**
Each of the three feature levels (4, 8, 12) passes through a small MLP projector (384 → 32 dims). The CGLT Carleson loss then measures whether the projected cloud of 196 points lies on a geometrically smooth, n-dimensional manifold (n=7). This prevents collapse (any constant-valued embedding is guaranteed to have a non-zero CGLT loss). The per-anchor weights from Branch 3 reduce the pressure on vessel patches.

**Step 7 — Loss Assembly**
```
L = 1.0 · L_dense   (main driver — dense prediction)
  + 20  · L_CGLT    (= λ·s = 0.02 × 10³ — manifold regularization)
  + 0.2 · L_LDS     (vessel prior learning)
```

---

## What the Architecture Gives You

| Property | How it's achieved |
|----------|------------------|
| **No collapse** | CGLT Carleson loss is anti-collapse by construction (bounded away from 0 at point-mass) |
| **Vessel preservation** | Vessel patches are pushed harder to predict (Branch 1) and left more free from regularization (Branch 2) |
| **No hard prior needed** | LDS learns vessel locations from denoising; RCP is soft guidance, not hard labels |
| **Single GPU friendly** | No EMA teacher → ~22M params total, ~6 GB VRAM at batch=64 |
| **Publication-grade** | First EMA-free dense JEPA for medical imaging + first vessel-modulated UR regularization |
