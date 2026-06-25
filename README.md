# Angio-JEPA — Coronary Angiography Latent Space Pretraining

<div align="center">

**Angio-JEPA: EMA-free JEPA with Manifold-Aware Regularization for Coronary Vessel SSL**

[![Status](https://img.shields.io/badge/status-in%20development-yellow)](https://github.com)
[![Target](https://img.shields.io/badge/target-MICCAI%202027-blue)](https://miccai.org)
[![GPU](https://img.shields.io/badge/GPU-8GB%2B-green)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

</div>

---

## Overview

** Angio-JEPA ** is a novel self-supervised learning framework for coronary X-ray angiography. It builds on the JEPA (Joint Embedding Predictive Architecture) paradigm with three key innovations:

1. **EMA-free architecture** — No teacher network. Collapse is prevented by the CGLT regularizer (from UR-JEPA), cutting parameters and memory by ~50%.
2. **Manifold-aligned regularization** — Replaces SIGReg's isotropic Gaussian target with CGLT's uniform n-rectifiability, matching the intrinsically low-dimensional structure of coronary vessels.
3. **Learned vessel prior (LDS)** — A Latent Denoising Score branch bootstrapped by a Robust Consensus Prior (Frangi + Sato + Hessian majority vote) that self-corrects during training.

> **Target venue**: MICCAI 2027 / Medical Image Analysis  
> **Hardware**: Single GPU ≥ 8 GB VRAM

--
## License

This project builds upon [MAE](https://github.com/facebookresearch/mae) (CC-BY-NC 4.0). The XA-170K dataset is licensed under CC-BY-NC 4.0.

