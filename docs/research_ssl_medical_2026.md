# Research: Self-Supervised Learning for Medical Imaging (2025–2026)

## Summary

Self-supervised learning (SSL) for medical imaging has undergone a paradigm shift in 2025–2026, with three major families — Masked Autoencoders (MAE), contrastive methods (SimCLR/DINO/MoCo), and Joint-Embedding Predictive Architectures (JEPA) — each finding distinct niches. For coronary angiography specifically, anatomy-guided MAE variants (VasoMIM, AAAI 2026) and video-text contrastive learning (DeepCORO-CLIP, Mar 2026) have produced the strongest results to date, while JEPA has shown breakthrough performance in ultrasound and echocardiography (EchoJEPA, US-JEPA) but has **not yet been applied to coronary angiography**, representing a clear research gap.

---

## Key Findings

### 1. VasoMIM — State-of-the-Art SSL for X-Ray Angiography (AAAI 2026)

- **Method:** Masked Image Modeling (MAE) with **Frangi-filter-based vessel-aware masking** + anatomical consistency loss. Uses a ViT backbone.
- **Dataset:** XA-170K — the largest X-ray angiogram pre-training dataset ever assembled (171,478 images from CADICA, SYNTAX, XCAD, CoronaryDominance).
- **Performance:** Achieves state-of-the-art on vessel segmentation, stenosis segmentation, vessel segment segmentation, and stenosis detection across 6 downstream datasets — surpassing ImageNet-pretrained and vanilla MAE baselines.
- **Key insight:** Anatomy-guided masking (focusing on vessel structures via Frangi filtering) dramatically improves downstream representation quality over random masking.
- **Code:** https://github.com/Dxhuang-CASIA/XA-SSL
- **Dataset (HuggingFace):** https://huggingface.co/datasets/waha2000huang/XA-170K
- **Source:** arXiv:2602.11536 — Huang et al., CASIA + Alibaba DAMO Academy

### 2. CM-UNet — SSL Drastically Reduces Annotation Requirements for Coronary Segmentation

- **Method:** Compares **contrastive (MoCo v2)** and **masked (MAE/SparK)** SSL paradigms head-to-head for coronary artery segmentation in X-ray angiography.
- **Dataset:** FAME2 multicenter dataset.
- **Landmark finding:** Fine-tuning with only 18 annotated images (vs. 500) caused only a **15.2% Dice drop with SSL**, versus a **46.5% drop without SSL** — a ~96% reduction in needed annotations.
- **Key insight:** SSL pre-training massively reduces annotation requirements. Multiple SSL paradigms all help, but contrastive and masked approaches can be complementary.
- **Code:** https://github.com/CamilleChallier/Contrastive-Masked-UNet
- **Source:** arXiv:2507.17779 — Challier et al., EPFL

### 3. DeepCORO-CLIP — First Multi-View Video-Text Foundation Model for Coronary Angiography

- **Method:** **Video-text contrastive learning** (CLIP-style) on angiography videos paired with clinical reports.
- **Dataset:** 203,808 angiography videos from 28,117 patients across 32,473 studies (Montreal Heart Institute); externally validated at UCSF.
- **Performance:**
  - Stenosis detection: AUROC **0.888** (internal) / **0.89** (external)
  - QCA MAE: **13.6%** (vs. clinical reports at 19.0%)
  - 1-year MACE prediction: AUROC **0.79**
  - LVEF estimation: MAE **7.3%**
  - Inference time: **4.2 seconds** in hospital deployment
- **Key insight:** First large-scale multimodal foundation model purpose-built for coronary angiography. Code and weights publicly released.
- **Source:** arXiv:2603.17675 — Harrabi, Wu, Tison, Ouyang et al., MHI + UCSF

### 4. CORA — Pathology-Driven SSL for Coronary CT Angiography

- **Method:** **Pathology-centric synthesis-driven SSL** — uses an anatomy-guided lesion synthesis engine to bias representation learning toward disease-relevant features, unlike standard MAE which captures global anatomical statistics.
- **Dataset:** 12,801 unlabeled CCTA volumes; evaluated on 9 independent hospital datasets.
- **Performance:** **Up to 29% improvement** over other 3D vision foundation models (DINOv3, MAE) on plaque characterization, stenosis detection, and segmentation.
- **Key insight:** Standard MIM/MAE is biased toward global anatomy; pathology-driven SSL captures localized plaque features better — critical for CAD assessment.
- **Source:** arXiv:2603.24847 — Hao, Durak, Bagci, Zhou et al.

### 5. EchoJEPA — Breakthrough JEPA Results for Echocardiography (Feb 2026)

- **Method:** **V-JEPA-2 style** architecture with frozen teacher weights + random predictor initialization. Pre-trained on the largest echo corpus ever: 18M echocardiograms across 300K patients.
- **Performance:**
  - **LVEF estimation:** ~20% improvement over best foundation model baselines
  - **RVSP estimation:** ~17% improvement
  - **79% view classification accuracy with only 1% labeled data** (vs. 42% for best baseline at 100%)
  - Degrades only **2% under acoustic perturbations** (vs. 17% for competitors)
  - **Zero-shot pediatric performance surpasses fully fine-tuned baselines**
- **Key insight:** JEPA's latent-space prediction is inherently robust to speckle noise and domain shift — directly relevant to angiography which also has challenging acquisition conditions.
- **Source:** arXiv:2602.02603 — Munim, Fallahpour, Szasz, Attarpour, Wang et al., U of Toronto / UHN

### 6. US-JEPA — JEPA Adapted for Medical Ultrasound via SALT (Feb 2026)

- **Method:** I-JEPA adapted with **SALT** (Static-teacher Asymmetric Latent Training) — replaces the brittle EMA teacher with a frozen domain-specific teacher.
- **Evaluation:** First rigorous comparison of all ultrasound foundation models on the **UltraBench** benchmark.
- **Performance:** Competitive or superior to domain-specific and universal vision baselines under linear probing.
- **Key insight:** The frozen teacher (SALT) modification makes JEPA training more stable and practical for medical domains. Validated the principle that JEPA excels when pixel-level reconstruction is unreliable.
- **Source:** arXiv:2602.19322 — Radhachandran, Ivezić, Athreya et al., UCLA

### 7. RadJEPA — JEPA for Chest X-Rays (Jan 2026)

- **Method:** Standard I-JEPA (ViT-B/14 at 224×224) pre-trained on ~840K unlabeled chest X-rays.
- **Evaluation:** Radiology report generation with frozen Vicuna-7B decoder; also substituted into MedLLaVA, Qwen-2.5, BLIP-2, and Phi-4.
- **Key finding:** Matches or exceeds the strongest image-only and vision-language baselines **without any language supervision during pretraining**.
- **Source:** arXiv:2601.15891 — Khan, Husain, Jalan, Jadhav

### 8. SurgMotion — V-JEPA for Surgical Video (Feb 2026)

- **Method:** V-JEPA with three innovations: (1) motion-guided latent masked prediction, (2) spatiotemporal affinity self-distillation, (3) spatiotemporal feature diversity regularization (SFDR).
- **Dataset:** SurgMotion-15M — 3,658 hours of surgical video from 50 sources across 13 anatomical regions.
- **Performance (17 benchmarks):**
  - **+14.6% F1 on EgoSurgery** (workflow recognition)
  - **+10.3% on PitVis**
  - **39.54% mAP-IVT on CholecT50** (action triplet recognition)
- **Key insight:** Motion-guided masking is a powerful innovation for video-based JEPA — directly relevant to angiography video pretraining.
- **Source:** arXiv:2602.05638 — Wu, Holm, Chen, Wang, Navab, Lei et al.

### 9. Head-to-Head: MAE vs JEPA vs Contrastive for Medical Imaging

- **"Pretext Matters" (arXiv:2603.22649, Mar 2026):** First systematic comparison of JEAs (contrastive/DINO) vs JEPAs (predictive) vs MAE (reconstruction) across ultrasound and histopathology. **Key rule:** Contrastive methods excel for spatially localized signal (histopathology); JEPAs excel for globally structured information (ultrasound); MAE underperforms when pixel-level noise dominates.
- **MRI MAE vs JEPA (arXiv:2606.13315, Jun 2026):** First systematic comparison of MAE vs JEPA for 3D brain MRI disease detection. **Finding:** MAE with spectral-domain supervision consistently outperforms JEPA for MRI. The optimal SSL method is **fundamentally determined by task/modality structure**.
- **C-JEPA (NeurIPS 2024):** Unifies contrastive and JEPA approaches, achieving better results than DINO and MAE alone on standard benchmarks.
- **Bottom line:** No single SSL method dominates. Modality noise profile, signal spatial distribution, and downstream task all dictate the optimal choice.

### 10. Key Datasets for Coronary Angiography SSL Research

| Dataset | Type | Size | Source |
|---------|------|------|--------|
| **XA-170K** | X-ray angiogram | 171,478 images | HuggingFace (waha2000huang) |
| **CoronaryDominance** | X-ray angiogram | 160,320 images | Scientific Data 2025 |
| **ARCADE** | X-ray angiogram | ~1,000 patients | Zenodo |
| **CADICA** | X-ray angiogram | 6,594 images | Mendeley Data |
| **FAME2** | X-ray angiogram | 635 patients (multi-center) | EPFL |
| **DeepCORO-CLIP** | Angiography video | 203,808 videos (28K patients) | Montreal Heart Institute |

---

## Critical Research Gap: No JEPA for Coronary Angiography

- Every major JEPA-medical paper (EchoJEPA, US-JEPA, RadJEPA, SurgMotion) appeared in late 2025–2026, but **none targets coronary angiography**.
- Angiography shares characteristics with modalities where JEPA excels: video-like temporal sequences, noise artifacts, and a need for robustness to domain shift.
- VasoMIM (MAE-based, AAAI 2026) is the closest angiography SSL work, but uses pixel-level reconstruction which is suboptimal for noisy angiogram data per the "Pretext Matters" findings.
- A **hybrid approach** combining angiography-specific innovations (vessel-aware masking from VasoMIM, motion-guided masking from SurgMotion, frozen-teacher JEPA training from EchoJEPA/US-JEPA) represents a promising and unexplored research direction.

---

## References

1. **VasoMIM: Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis** — Huang et al., AAAI 2026. arXiv:2602.11536. Code: https://github.com/Dxhuang-CASIA/XA-SSL
2. **CM-UNet: A Self-Supervised Learning-Based Model for Coronary Artery Segmentation** — Challier et al., Jul 2025. arXiv:2507.17779. Code: https://github.com/CamilleChallier/Contrastive-Masked-UNet
3. **DeepCORO-CLIP: A Multi-View Foundation Model for Comprehensive Coronary Angiography Video-Text Analysis** — Harrabi et al., Mar 2026. arXiv:2603.17675
4. **CORA: A Pathology Synthesis Driven Foundation Model for Coronary CT Angiography Analysis** — Hao et al., Mar 2026. arXiv:2603.24847
5. **EchoJEPA: A Latent Predictive Foundation Model for Echocardiography** — Munim et al., Feb 2026. arXiv:2602.02603
6. **US-JEPA: A Joint Embedding Predictive Architecture for Medical Ultrasound** — Radhachandran et al., Feb 2026. arXiv:2602.19322
7. **RadJEPA: Radiology Encoder for Chest X-Rays via Joint Embedding Predictive Architecture** — Khan et al., Jan 2026. arXiv:2601.15891
8. **SurgMotion: A Video-Native Foundation Model for Universal Understanding of Surgical Videos** — Wu et al., Feb 2026. arXiv:2602.05638
9. **Pretext Matters: An Empirical Study of SSL Methods in Medical Imaging** — Ivezić et al., Mar 2026. arXiv:2603.22649
10. **Masked and Predictive Self-Supervised Foundation Models for 3D Brain MRI** — Ergün et al., Jun 2026. arXiv:2606.13315
11. **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning** — Assran et al., Jun 2025. arXiv:2506.09985
12. **I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture** — Assran et al., 2023. arXiv:2301.08243
13. **Cardiac-CLIP: A Vision-Language Foundation Model for 3D Cardiac CT Images** — Hu et al., Jul 2025. arXiv:2507.22024
14. **PINS-CAD: Physics-informed Self-supervised Learning for Coronary Artery Digital Twins** — Sun et al., Nov 2025. arXiv:2512.03055
15. **StenCE: Cross-Modal Contrastive Learning of ECG and Angiography** — Cenikj et al., May 2026. arXiv:2606.02605
16. **BioVFM-21M: Benchmarking and Scaling Self-Supervised Vision Foundation Models** — Liu et al., May 2025. arXiv:2505.09329
17. **C-JEPA: Connecting JEPA with Contrastive Self-Supervised Learning** — Mo, Tong, NeurIPS 2024
18. **VAMAE: Vessel-Aware Masked Autoencoders for OCT Angiography** — Abolade et al., Apr 2026. arXiv:2604.06583
19. **Deep Learning Approaches for Medical Imaging Under Varying Degrees of Label Availability** — Apr 2025. arXiv:2504.11588
20. **XA-170K Dataset** — HuggingFace: https://huggingface.co/datasets/waha2000huang/XA-170K