# VasoJEPA

Arhitectură și cod pentru pre-training SSL pe angiograme coronariene.

```
/
├── vasojepa/                    # Nucleul arhitecturii
│   ├── __init__.py
│   ├── model.py                 # VasoJEPA — ansamblul complet
│   ├── sigreg.py                # SIGReg + DeepSIGReg + ponderare
│   ├── anatomical_tokens.py     # Register tokeni anatomici
│   ├── predictor.py             # Predictor JEPA condiționat
│   └── dataset.py               # DataLoader + augmentări angiograme
│
├── scripts/                     # Scripturi de rulat
│   ├── prepare_sam.py           # Rulează SAM pe 30k imagini
│   ├── prepare_frangi.py        # Calculează Frangi multi-scală
│   ├── prepare_consensus.py     # Calculează harta de consens
│   ├── train_vasojepa.py        # Antrenare VasoJEPA
│   ├── train_baselines.py       # Antrenare baseline-uri (UNet, LeJEPA, VasoMIM)
│   ├── eval_linear_probe.py     # Linear probing
│   ├── eval_segmentation.py     # Fine-tuning + evaluare segmentare
│   ├── eval_ablation.py         # Rulează toate variantele de ablație
│   ├── eval_cross_dataset.py    # Cross-dataset evaluation
│   └── visualize_embeddings.py  # PCA pe embeddings
│
├── configs/                     # Configurații
│   ├── vasojepa.yaml            # Hiperparametri VasoJEPA
│   ├── baselines.yaml           # Config pentru baseline-uri
│   └── ablation.yaml            # Config pentru ablații
│
├── data/                        # Date (ignorate de git)
│   ├── images/                  # ~30k imagini originale
│   ├── pseudo_masks/            # Măști SAM (.npy)
│   ├── frangi/                  # Hărți Frangi (.npy)
│   ├── consensus/               # Hărți de consens (.npy)
│   ├── labels_126/              # Cele 126 adnotări manuale
│   └── splits/                  # Train/val/test split-uri
│
├── checkpoints/                 # Modele antrenate (ignorate de git)
│   ├── vasojepa/
│   ├── lejepa/
│   └── baselines/
│
├── results/                     # Rezultate, tabele, figuri
│   ├── tables/
│   ├── figures/
│   └── logs/
│
├── vasojepa_architecture.md     # Specificația arhitecturii
├── plan_implementare.md         # Planul pas cu pas
└── requirements.txt             # Dependințe
```

## Cum rulezi

```bash
# 1. Pregătire date
python scripts/prepare_sam.py
python scripts/prepare_frangi.py
python scripts/prepare_consensus.py

# 2. Antrenare
python scripts/train_vasojepa.py --config configs/vasojepa.yaml

# 3. Evaluare
python scripts/eval_segmentation.py --checkpoint checkpoints/vasojepa/best.pth
```