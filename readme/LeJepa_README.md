# LeJepa Training for Coronary Angiography

Implementare LeJepa pentru self-supervised learning pe imagini de angiografie coronariană.

## Descriere

Această implementare se bazează pe [LeJepa paper](https://arxiv.org/abs/2511.08544) și include:
- **SIGReg loss** - Spectral independence via random features
- **Invariance loss** - Pentru consistență între view-uri
- **Multi-view augmentation** - V view-uri augmentate per imagine
- **Configurare YAML** - Similar cu DINO training
- **Suport pentru date extra** - Include toate datele din ARCADE (stenoza, segmentare, syntax, extra)

## Structură

```
config/
  lejepa_config.yaml     # Configurare training
engine/
  LeJepa_trained.py      # Script principal de training
data/ARCADE/
  LeJepa.py             # Dataset wrapper cu augmentări
```

## Configurare

Editează `config/lejepa_config.yaml`:

```yaml
model:
  backbone: "vit_small_patch8_224"  # Backbone architecture
  proj_dim: 128                      # Projection dimension
  
data:
  batch_size: 32
  image_size: 256                    # 256x256 pentru angiografii
  num_views: 4                       # V views per image
  
optimizer:
  lr: 0.002
  epochs: 400
  lamb: 0.02                         # λ - balansează SIGReg și Invariance
```

## Rulare Training

```bash
cd /workspace/Collateral-Coronary-Vessels-XAI
python engine/LeJepa_trained.py
```

## Parametri Importanți

- **lamb (λ)**: Balanță între SIGReg și Invariance loss
  - `lamb=0.02` înseamnă 2% SIGReg, 98% Invariance
  - Valori recomandate: 0.01 - 0.05
  
- **num_views (V)**: Număr de view-uri augmentate
  - Valori recomandate: 2-4 pentru început
  - Mai multe view-uri = mai mult compute dar învățare mai bună

- **proj_dim**: Dimensiunea proiecției pentru SIGReg
  - Valori recomandate: 64-256
  - Mai mare = mai expresiv dar mai lent

## Loss-uri

Training-ul minimizează:
```
LeJepa Loss = λ * SIGReg(proj) + (1-λ) * Invariance(proj)
```

Unde:
- **SIGReg**: Spectral independence - forțează independență statistică între feature-uri
- **Invariance**: Consistență între view-uri - forțează reprezentări similare pentru aceeași imagine

## Augmentări

Pentru angiografii coronariene:
- RandomResizedCrop (0.08-1.0 scale)
- ColorJitter (contrast/brightness)
- GaussianBlur
- RandomSolarize
- Flip-uri (orizontale și verticale)

## Monitorizare

Training-ul loggează în TensorBoard:
```bash
tensorboard --logdir runs/lejepa_vit_small/
```

Metrici disponibile:
- `train/lejepa_loss` - Loss total
- `train/sigreg_loss` - SIGReg component
- `train/inv_loss` - Invariance component
- `train/lr` - Learning rate

## Checkpoints

Salvate automat în `checkpoints/lejepa/`:
- `lejepa_vit_small_checkpoint_last.pth` - Ultimul checkpoint
- `lejepa_vit_small_checkpoint_best.pth` - Best checkpoint (cel mai mic loss)
- `lejepa_vit_small_backbone_last.pth` - Doar backbone (pentru fine-tuning)

## Resume Training

Pentru a continua training-ul:
```yaml
# În lejepa_config.yaml
system:
  resume: true
  resume_from: "checkpoints/lejepa/lejepa_vit_small_checkpoint_last.pth"
```

## Fine-tuning

După pretraining, backbone-ul poate fi folosit pentru downstream tasks:

```python
import timm
import torch

# Încarcă backbone pretrenat
backbone = timm.create_model('vit_small_patch8_224', num_classes=0)
backbone.load_state_dict(
    torch.load('checkpoints/lejepa/lejepa_vit_small_backbone_last.pth')
)

# Adaugă classifier pentru task-ul tău
classifier = torch.nn.Linear(384, num_classes)
model = torch.nn.Sequential(backbone, classifier)
```

## Diferențe față de DINO

| Aspect | DINO | LeJepa |
|--------|------|--------|
| **Obiectiv** | Teacher-Student cu cross-entropy | SIGReg + Invariance |
| **Complexitate** | Necesită EMA teacher, stop-gradient | Un singur network, no stop-gradient |
| **Hyperparametri** | Multe (temp, momentum, etc.) | Doar λ |
| **Training** | Mai sensibil | Mai stabil |
| **View-uri** | Multi-crop (2 global + N local) | Multi-view (V augmentări) |

## Referințe

- Paper: [LeJepa: Provable and Scalable Self-Supervised Learning](https://arxiv.org/abs/2511.08544)
- Code: [GitHub - LeJepa](https://github.com/galilai-group/lejepa)
- Minimal Example: [MINIMAL.md](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md)
