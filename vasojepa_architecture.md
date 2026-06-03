# VasoJEPA — Specificație Arhitectură

> **VasoJEPA: Multi-Scale Joint-Embedding Prediction with Vessel-Aware Regularization for Coronary Angiogram Analysis**
>
> Propunere pentru lucrare de licență + publicație.
> Bazată pe LeJEPA (SIGReg, teacher-free) + V-JEPA 2.1 (dense features) + ghidaj anatomic.

---

## 1. Problema

| Ce | Detalii |
|---|---------|
| **Context** | Segmentare vase coronariene din angiograme |
| **Date** | ~30k imagini neetichetate, ~126 adnotate manual |
| **Problemă** | LeJEPA converge (SIGReg scade), dar reprezentările nu se transferă la segmentare — *information stripping* pe angiograme |
| **Ipoteză** | Regularizarea SIGReg șterge informația spațială fină pe structuri vasculare. Un ghidaj anatomic slab poate preveni acest efect. |

---

## 2. Contribuții (C1–C8)

| # | Contribuție | Tip | Publicabil |
|---|------------|-----|:----------:|
| C1 | **Diagnostic**: LeJEPA e insuficient pe angiograme (primul studiu) | Rezultat negativ | ✅ |
| C2 | **DeepSIGReg**: SIGReg multi-scală la layer 4, 8, 12 | Metodă nouă | ✅ |
| C3 | **Ghidaj consensual SAM+Hessian** ca ponderare SIGReg | Metodă nouă | ✅ |
| C4 | **Auto-echilibru**: DeepSIGReg + Predictor JEPA (forțe opuse) | Insight | ✅ |
| C5 | **Ablație sistematică** | Experiment | ✅ |
| C6 | **Evaluare stratificată** pe vase colaterale | Experiment | ✅ |
| C7 | **Cross-dataset** (CADICA, SYNTAX, XCAD) | Experiment | ✅ |
| C8 | **PCA embeddings**: vizualizare separare vas vs fond | Vizualizare | ✅ |

---

## 3. Arhitectura VasoJEPA (Varianta A — 1+3)

### 3.1 Diagrama completă

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VASOJEPA                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: 224×224 → Patchify (14×14, patch_size=16) → 196 patch-uri  │
│                                                                     │
│  + 2 Register Tokeni anatomici: [REG_VESSEL, REG_BG] (198 tokeni)   │
│                                                                     │
│  ViT Encoder (timm vit_small_patch16_224, 12 blocuri, hidden=384)   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  Bloc 1-4                                                   │    │
│  │  [patch_1 ... patch_196, reg_vessel, reg_bg]                │    │
│  │       ↓ self-attention                                      │    │
│  │  Bloc 4 → emb4 (198 × 384)                                  │    │
│  │       ↓                                                     │    │
│  │  Proj_4: MLP(384 → 2048 → 128) + BatchNorm1d               │    │
│  │       ↓                                                     │    │
│  │  DeepSIGReg_4 (doar pe patch-uri, ponderat de consens)      │    │
│  │       ↓                                                     │    │
│  │  REG_VESSEL, REG_BG trec mai departe la blocul 5             │    │
│  │                                                             │    │
│  │  Bloc 5-8                                                   │    │
│  │  Bloc 8 → emb8 (198 × 384)                                  │    │
│  │       ↓                                                     │    │
│  │  Proj_8: MLP(384 → 2048 → 128) + BatchNorm1d               │    │
│  │       ↓                                                     │    │
│  │  DeepSIGReg_8 (doar pe patch-uri, ponderat de consens)      │    │
│  │                                                             │    │
│  │  Bloc 9-12                                                  │    │
│  │  Bloc 12 → emb12 + CLS (199 × 384)                          │    │
│  │       ↓                        ↓                            │    │
│  │  Proj_12: MLP(384→2048→128)  Proj_cls: MLP(384→2048→128)   │    │
│  │       ↓                        ↓                            │    │
│  │  DeepSIGReg_12 (ponderat)    SIGReg(cls)                    │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  RAMURA PREDICTOR (JEPA)                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  emb_cat = concat(emb4, emb8, emb12)_patch_only (196×1152)   │    │
│  │  consens_flat = consensus_per_patch (196×1)                   │    │
│  │  predictor_input = concat(emb_cat, consens_flat) (196×1153)   │    │
│  │       ↓                                                      │    │
│  │  Random mask (25% patch-uri, fără a masca register tokeni)   │    │
│  │       ↓                                                      │    │
│  │  Predictor Transformer (2 layer, 8 heads, dim=1153→1152)    │    │
│  │       ↓                                                      │    │
│  │  Prezice embeddings mascate → MSE(pred, target)               │    │
│  │  + SIGReg(pred) — regularizează și predicțiile               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  GHIDAJ ANATOMIC (consens SAM+Hessian)                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Per patch:                                                   │    │
│  │    h_sam = pool(SAM_mask, patch) → [0,1]                     │    │
│  │    h_hessian = pool(Frangi_multiscale, patch) → [0,1]        │    │
│  │                                                               │    │
│  │  consensus =                                                   │    │
│  │    1.0  if h_sam > 0.3 AND h_hessian > 0.3   (vas, sigur)    │    │
│  │    1.0  if h_sam < 0.1 AND h_hessian < 0.1   (fond, sigur)   │    │
│  │    0.3  if unul da, altul nu                 (incert)        │    │
│  │    0.0  if scoruri mari dar invers           (conflict)      │    │
│  │                                                               │    │
│  │  weight_per_patch = 1 - 0.8 × consensus                       │    │
│  │    → patch sigur de vas: SIGReg ~20% din forță                │    │
│  │    → patch sigur de fond: SIGReg ~100% din forță              │    │
│  │    → patch incert: SIGReg ~76% din forță                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Loss total

```
L_total = L_sigreg_cls
        + 0.3 · L_deepsigreg_layer_4
        + 0.6 · L_deepsigreg_layer_8
        + 1.0 · L_deepsigreg_layer_12
        + 0.5 · L_predictor_mse
        + 0.1 · L_predictor_sigreg
```

Toate ponderile sunt **validate prin ablație** (C5) pe setul de validare (126 imagini).

### 3.3 Flow de antrenare

```
Faza 1: Pregătire date (2-3 zile)
  1. Rulezi SAM pe 30k imagini → pseudo-măști per imagine
  2. Calculezi Frangi multi-scală (σ=1,2,4) pe 30k imagini
  3. Calculezi harta de consens per patch (196 per imagine)
  4. Pregătești cele 126 imagini adnotate pentru downstream

Faza 2: Pre-training VasoJEPA (1-2 zile pe 1 GPU)
  - ViT-S/16, 30k imagini, batch_size=64, LR=5e-4
  - AdamW, weight_decay=5e-2
  - Cosine annealing, warmup 10 epoci
  - Checkpoint la fiecare 10 epoci
  
Faza 3: Evaluare (1-2 zile)
  - Linear probing (cls token → clasificare vas/fond per patch)
  - Fine-tuning UNet cu encoder înghețat
  - Fine-tuning UNet complet
  - Ablație: LeJEPA, VasoJEPA fără ghidaj, VasoJEPA fără DeepSIGReg
  - Cross-dataset: test pe CADICA, SYNTAX, XCAD
  - PCA pe embeddings: LeJEPA vs VasoJEPA
```

---

## 4. Varianta B (2+3) — Cross-attention între nivele

> Păstrată ca **Future Work** în lucrare.

Diferența față de Varianta A:
- Predictorul NU primește consensul ca input (predictor JEPA standard)
- În schimb, proiectoarele DeepSIGReg de la layer 8 și 12 primesc **cross-attention** de la nivelele anterioare

```
Proj_8_enhanced = CrossAttention(Q=proj_8, KV=concat(proj_4, proj_8))
Proj_12_enhanced = CrossAttention(Q=proj_12, KV=concat(proj_4, proj_8, proj_12))
```

**Efect**: Nivelul 8 păstrează detalii fine descoperite de nivelul 4. Nivelul 12 le cumulează pe toate.

---

## 5. Justificare JEPA (de ce rămâne JEPA pur)

| Componentă | E JEPA? | Justificare |
|-----------|:-------:|-------------|
| SIGReg(cls) | ✅ Da | Regularizare în spațiu latent |
| DeepSIGReg(layer_i) | ✅ Da | Regularizare în spațiu latent, pe patch-uri |
| Predictor + MSE latent | ✅ Da | Prezicere în spațiu latent — inima JEPA |
| SIGReg(predictor_output) | ✅ Da | Regularizare a predicțiilor în latent |
| Ponderare cu consens | ✅ Da | Rescaling a loss-ului, nu task extern |
| Tokeni anatomici | ✅ Da | Modificare encoder, nu ies din latent |
| **Decoder de vesselness** | ❌ **NU** | Ar fi task auxiliar — eliminat |
| **Reconstrucție imagine** | ❌ **NU** | Ar fi MAE, nu JEPA — eliminat |

---

## 6. Poziționare în literatură

| Metoda | Teacher-free | Spațiu latent | Dense features | Multi-scală | Ghidaj | Angiograme |
|--------|:-----------:|:-------------:|:--------------:|:-----------:|:------:|:----------:|
| MAE | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| I-JEPA | ❌ (EMA) | ✅ | ❌ | ❌ | ❌ | ❌ |
| V-JEPA 2.1 | ❌ (EMA) | ✅ | ✅ | ✅ | ❌ | ❌ |
| LeJEPA | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| DINOv3 | ❌ (teacher) | ❌ | ❌ | ❌ | ❌ | ❌ |
| VasoMIM | ✅ | ❌ (MIM) | ❌ | ❌ | ✅ | ✅ |
| **VasoJEPA** | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** |

---

## 7. Originalitate (scor: ~8/10)

| Criteriu | Scor | Explicație |
|----------|:----:|-----------|
| C1: Diagnostic | 7/10 | Nimeni n-a testat LeJEPA pe medical |
| C2: DeepSIGReg | 6/10 | Extensie naturală, dar nerealizată |
| C3: Ghidaj consensual | 8/10 | Combinație originală SAM+Hessian |
| C4: Auto-echilibru | 7/10 | Insight nou, demonstrabil |
| **Per total** | **~8/10** | Conferință (MICCAI, ISBI) sau jurnal |

---

## 8. Dicționar de termeni

| Termen | Explicație |
|--------|-----------|
| **SIGReg** | Sketched Isotropic Gaussian Regularization — forțează embeddings să aibă distribuție Gaussiană. Din LeJEPA. |
| **DeepSIGReg** | SIGReg aplicat la multiple nivele intermediare (layer 4, 8, 12) pentru features multi-scală. |
| **Ghidaj consensual** | Ponderarea SIGReg-ului bazată pe acordul dintre SAM și Hessian per patch. |
| **Tokeni anatomici** | 2 register tokens (REG_VESSEL, REG_BG) care acumulează informație globală despre vase/fond. |
| **Predictor JEPA** | Transformer ușor care prezice embeddings mascate în spațiul latent. |
| **Information stripping** | Pierderea detaliilor spațiale fine cauzată de regularizare prea puternică. |

---

**Data:** 3 Iunie 2026
**Autori:** [Numele tău]
**Status:** Propunere arhitectură — de implementat și validat experimental.