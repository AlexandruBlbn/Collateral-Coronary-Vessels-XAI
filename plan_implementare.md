# Plan de Implementare — VasoJEPA

> **Obiectiv**: Implementarea, antrenarea și validarea arhitecturii VasoJEPA pentru lucrarea de licență.
> **Durată estimată**: ~4-6 săptămâni (în ritm constant, nu full-time).
> **Resurse necesare**: 1 GPU (≥8GB VRAM), Python + PyTorch, timm, datele.

---

## Faza 0: Setup și verificări (~1 săptămână)

### 0.1 Verifică datele disponibile
- [ ] Confirmă ce seturi ai: XA-170K (sau subset), ARCADE, CADICA, SYNTAX, XCAD
- [ ] Verifică formatul: imagini .png/.jpg, măști .png, rezoluție
- [ ] Verifică că cele ~126 adnotări sunt compatibile cu cele ~30k neetichetate
- [ ] Creează un mic script de statistici: dimensiuni, distribuție vas/fond, grosimi

### 0.2 Setup proiect
- [ ] Creează repo Git curat (sau fork al celui existent)
- [ ] Instalează dependințele: `pip install torch torchvision timm wandb scikit-image scipy`
- [ ] Verifică că LeJEPA rulează pe o imagine (MINIMAL.md)
- [ ] Rulează VasoMIM să vezi că ai baseline-ul funcțional

### 0.3 Rulează SAM pe ~30k imagini
```python
# scripts/run_sam.py
from segment_anything import sam_model_registry, SamPredictor
# SAM ViT-B (lightweight) — suficient pentru pseudo-labeluri
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
predictor = SamPredictor(sam)
for img_path in tqdm(image_paths):
    image = cv2.imread(img_path)
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=True)
    np.save(f"pseudo_masks/{img_path.stem}.npy", masks[0])  # cea mai bună mască
```
- ⏱ ~1-2 zile pe un GPU

### 0.4 Calculează Frangi multi-scală
```python
# scripts/compute_frangi.py
from skimage.filters import frangi, meijering
for img_path in tqdm(image_paths):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    frangi_map = frangi(img, sigmas=range(1, 5, 1), black_ridges=False)
    np.save(f"frangi/{img_path.stem}.npy", frangi_map)
```
- ⏱ ~1 zi pe CPU

### 0.5 Calculează harta de consens per patch
```python
# scripts/compute_consensus.py
for img_path in tqdm(image_paths):
    sam_mask = np.load(f"pseudo_masks/{img_path.stem}.npy")
    frangi_map = np.load(f"frangi/{img_path.stem}.npy")
    
    # Pool per patch (14×14 grid, patch_size=16)
    consensus_map = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            h_sam = sam_mask[i*16:(i+1)*16, j*16:(j+1)*16].mean()
            h_hessian = frangi_map[i*16:(i+1)*16, j*16:(j+1)*16].mean()
            
            if h_sam > 0.3 and h_hessian > 0.3:
                consensus_map[i,j] = 1.0  # vas, sigur
            elif h_sam < 0.1 and h_hessian < 0.1:
                consensus_map[i,j] = 1.0  # fond, sigur
            elif (h_sam > 0.3) != (h_hessian > 0.3):
                consensus_map[i,j] = 0.3  # incert
            else:
                consensus_map[i,j] = 0.0  # conflict
    
    np.save(f"consensus/{img_path.stem}.npy", consensus_map)
```
- ⏱ ~1 zi

---

## Faza 1: Implementare VasoJEPA (~1.5 săptămâni)

### 1.1 Implementează SIGReg (din LeJEPA MINIMAL.md)
```python
# vasojepa/sigreg.py
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, proj_dim=128):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.proj_dim = proj_dim

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
```
- **Testează**: Rulează pe un tensor random (B, 196, 128) — trebuie să dea un scalar.

### 1.2 Implementează DeepSIGReg cu ponderare
```python
# vasojepa/deep_sigreg.py
class DeepSIGReg(nn.Module):
    def __init__(self, in_dim=384, proj_dim=128, alpha=0.8):
        super().__init__()
        self.proj = MLP(in_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        self.sigreg = SIGReg(proj_dim=proj_dim)
        self.alpha = alpha  # puterea ghidajului
        
    def forward(self, embeddings, consensus_map=None):
        # embeddings: (B, N, 384) — doar patch tokens, fără register
        # consensus_map: (B, N) — scor 0-1 per patch, sau None
        proj = self.proj(embeddings)  # (B, N, 128)
        
        if consensus_map is not None:
            # weight per patch
            weight = 1.0 - self.alpha * consensus_map.unsqueeze(-1)  # (B, N, 1)
            # SIGReg per patch
            loss_per_patch = self.sigreg_per_patch(proj)  # (B, N)
            return (loss_per_patch * weight.squeeze(-1)).mean()
        else:
            return self.sigreg(proj)
    
    def sigreg_per_patch(self, proj):
        # aplică SIGReg pe fiecare patch în parte
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.sigreg.t
        err = (x_t.cos().mean(-3) - self.sigreg.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.sigreg.weights) * proj.size(-2)
        return statistic  # (B, N)
```

### 1.3 Implementează Tokenii Anatomici (register tokens)
```python
# vasojepa/anatomical_tokens.py
class AnatomicalTokens(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.register_vessel = None  # se inițializează la primul forward
        self.register_bg = None
        
    def initialize(self, embeddings, consensus_map):
        # embeddings: (B, N, 384) — patch embeddings de la layer 4
        # consensus_map: (B, N) — scor 0-1
        vessel_weight = (consensus_map > 0.5).float().unsqueeze(-1)  # (B, N, 1)
        bg_weight = (consensus_map < 0.5).float().unsqueeze(-1)  # (B, N, 1)
        
        # weighted mean
        sum_vessel = (embeddings * vessel_weight).sum(dim=1)
        count_vessel = vessel_weight.sum(dim=1).clamp(min=1)
        sum_bg = (embeddings * bg_weight).sum(dim=1)
        count_bg = bg_weight.sum(dim=1).clamp(min=1)
        
        self.register_vessel = (sum_vessel / count_vessel).unsqueeze(1)  # (B, 1, 384)
        self.register_bg = (sum_bg / count_bg).unsqueeze(1)  # (B, 1, 384)
        
        return torch.cat([embeddings, self.register_vessel, self.register_bg], dim=1)
        # → (B, N+2, 384)
        
    def remove(self, embeddings_with_registers):
        """Decupează register tokenii înainte de loss"""
        return embeddings_with_registers[:, :-2, :]  # (B, N, 384)
```

### 1.4 Implementează Predictorul JEPA condiționat
```python
# vasojepa/predictor.py
class ConditionedPredictor(nn.Module):
    def __init__(self, dim=1152, cond_dim=1, hidden_dim=2048, depth=2, heads=8):
        super().__init__()
        self.input_proj = nn.Linear(dim + cond_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_proj = nn.Linear(hidden_dim, dim)
        
    def forward(self, emb_cat, consensus_flat, mask):
        # emb_cat: (B, N, 1152) — concatenarea emb4+emb8+emb12
        # consensus_flat: (B, N, 1) — consensul per patch
        # mask: (B, N) — boolean, True pentru mascat
        
        # condiționează inputul
        x = torch.cat([emb_cat, consensus_flat], dim=-1)  # (B, N, 1153)
        x = self.input_proj(x)  # (B, N, 2048)
        
        # mask: înlocuiește patch-urile mascate cu un token learnable
        mask_token = nn.Parameter(torch.randn(1, 1, 2048), requires_grad=True)
        x = x * (~mask).unsqueeze(-1) + mask_token * mask.unsqueeze(-1)
        
        # predict
        x = self.transformer(x)
        pred = self.output_proj(x)  # (B, N, 1152)
        
        return pred
```

### 1.5 Asamblează VasoJEPA complet
```python
# vasojepa/model.py
class VasoJEPA(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        # Encoder ViT-S
        self.encoder = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,  # fără classification head
            img_size=224,
        )
        self.hidden_dim = self.encoder.embed_dim  # 384
        
        # Tokeni anatomici
        self.anatomical_tokens = AnatomicalTokens(dim=self.hidden_dim)
        
        # DeepSIGReg la 3 nivele
        self.deepsigreg_4 = DeepSIGReg(in_dim=self.hidden_dim, alpha=alpha)
        self.deepsigreg_8 = DeepSIGReg(in_dim=self.hidden_dim, alpha=alpha)
        self.deepsigreg_12 = DeepSIGReg(in_dim=self.hidden_dim, alpha=alpha)
        self.sigreg_cls = SIGReg(proj_dim=128)
        self.proj_cls = MLP(self.hidden_dim, [2048, 2048, 128], norm_layer=nn.BatchNorm1d)
        
        # Predictor condiționat
        self.predictor = ConditionedPredictor(dim=self.hidden_dim * 3, cond_dim=1)
        self.predictor_sigreg = SIGReg(proj_dim=self.hidden_dim * 3)
        
    def forward(self, x, consensus_map):
        # x: (B, 3, 224, 224)
        # consensus_map: (B, 196) — scor consens per patch
        
        # Encode (caz ideal: encoder care expune embeddings intermediare)
        # Alternativ, poți folosi forward_features cu blocks
        emb4 = self._get_intermediate(x, layer=4)  # (B, 197, 384) — are CLS
        emb8 = self._get_intermediate(x, layer=8)
        emb12, cls = self._get_intermediate(x, layer=12)  # cls separat
        
        # Elimină CLS pentru patch-only, adaugă register tokens
        emb4_patch = self.anatomical_tokens.initialize(emb4[:, 1:, :], consensus_map)  # (B, 198, 384)
        emb8_patch = self.anatomical_tokens.initialize(emb8[:, 1:, :], consensus_map)  # (B, 198, 384)
        emb12_patch = self.anatomical_tokens.initialize(emb12[:, 1:, :], consensus_map)  # (B, 198, 384)
        
        # DeepSIGReg — doar patch-uri, fără register
        loss_deep4 = self.deepsigreg_4(self.anatomical_tokens.remove(emb4_patch), consensus_map)
        loss_deep8 = self.deepsigreg_8(self.anatomical_tokens.remove(emb8_patch), consensus_map)
        loss_deep12 = self.deepsigreg_12(self.anatomical_tokens.remove(emb12_patch), consensus_map)
        loss_cls = self.sigreg_cls(self.proj_cls(cls))
        
        # Predictor condiționat
        emb_cat = torch.cat([
            self.anatomical_tokens.remove(emb4_patch),
            self.anatomical_tokens.remove(emb8_patch),
            self.anatomical_tokens.remove(emb12_patch),
        ], dim=-1)  # (B, 196, 1152)
        
        # mask random 25%
        mask = torch.rand(emb_cat.shape[0], emb_cat.shape[1], device=x.device) < 0.25
        consensus_flat = consensus_map.unsqueeze(-1).float()
        
        pred = self.predictor(emb_cat, consensus_flat, mask)
        loss_pred = F.mse_loss(pred[mask], emb_cat[mask])
        loss_pred_sigreg = self.predictor_sigreg(pred)
        
        # Loss total
        loss = (loss_cls 
                + 0.3 * loss_deep4 
                + 0.6 * loss_deep8 
                + 1.0 * loss_deep12
                + 0.5 * loss_pred
                + 0.1 * loss_pred_sigreg)
        
        return loss, {"cls": loss_cls.item(), "deep4": loss_deep4.item(), 
                      "deep8": loss_deep8.item(), "deep12": loss_deep12.item(),
                      "pred": loss_pred.item(), "pred_sigreg": loss_pred_sigreg.item()}
    
    def _get_intermediate(self, x, layer):
        """Extrage embeddings la un anumit layer"""
        # Implementare depinde de timm — poți folosi forward_intermediates
        pass
```

### 1.6 Creează DataLoader
```python
# vasojepa/dataset.py
class AngiogramDataset(Dataset):
    def __init__(self, image_paths, consensus_paths, augment=True):
        self.image_paths = image_paths
        self.consensus_paths = consensus_paths
        self.augment = augment
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        consensus = np.load(self.consensus_paths[idx])  # (14, 14)
        
        # Augmentări adaptate angiogramelor
        # Evită ColorJitter și GaussianBlur care distrug vase
        # Folosește: RandomAffine, RandomHorizontalFlip, ElasticTransform
        if self.augment:
            img = elastic_transform(img)
            img = random_affine(img, degrees=5, translate=0.05)
        
        # Ajustează consensul la transformări (dacă e cazul)
        consensus = consensus.flatten()  # (196,)
        
        return T.ToTensor()(img), torch.tensor(consensus, dtype=torch.float32)
```

---

## Faza 2: Antrenare VasoJEPA (~2 săptămâni)

### 2.1 Antrenare propriu-zisă
```bash
python train_vasojepa.py \
    --data-dir /path/to/images \
    --consensus-dir /path/to/consensus \
    --epochs 200 \
    --batch-size 64 \
    --lr 5e-4 \
    --weight-decay 5e-2 \
    --warmup-epochs 10 \
    --output-dir ./checkpoints/vasojepa
```

- Monitorizează în WandB: loss total, loss per component, gradient norm
- Salvează checkpoint la fiecare 10 epoci
- Rulează linear probing la fiecare 50 epoci să vezi dacă embeddings se îmbunătățesc

### 2.2 Validare intermediară (la fiecare 50 epoci)
```bash
python eval_linear_probe.py \
    --checkpoint ./checkpoints/vasojepa/epoch_50.pth \
    --train-labels ./data/train_labels_126 \
    --test-labels ./data/test_labels
```

- Dacă linear probing nu crește după 50 epoci, ajustezi α sau λ

### 2.3 Debugging comun și soluții
| Problemă | Posibilă cauză | Soluție |
|----------|---------------|---------|
| Loss explodează | SIGReg instabil | Redu λ, adaugă gradient clipping |
| Loss total scade, linear probing nu | Information stripping | Reduce α (ghidajul e prea slab) |
| Predictor loss crește | DeepSIGReg prea puternic | Reduce λ pe layer-urile joase |
| Tokenii anatomici nu se diferențiază | Inițializare greșită | Verifică pooling-ul consensului |
| SAM pseudo-labeluri prea zgomotoase | SAM prost pe angiograme | Crește pragul de consens la 0.5 |

---

## Faza 3: Evaluare și Baseline-uri (~1 săptămână)

### 3.1 Baseline-uri necesare

| # | Model | Pre-training | Fine-tuning |
|---|-------|:-----------:|:-----------:|
| 1 | UNet from scratch | ❌ | Pe 126 |
| 2 | UNet + LeJEPA pre-train | LeJEPA vanilla | Pe 126 |
| 3 | UNet + VasoJEPA (propus) | VasoJEPA | Pe 126 |
| 4 | UNet + VasoMIM | VasoMIM | Pe 126 |
| 5 | SAM + linear probe | SAM înghețat | Doar cap liniar |

### 3.2 Metrici de evaluare
| Metrică | Ce măsoară |
|---------|-----------|
| **Dice per imagine** | Calitate generală segmentare |
| **Dice colaterale** (vase < 2 pixeli) | Exact acolo unde ghidajul contează |
| **Precision / Recall** | Falsuri pozitive vs ratări |
| **Hausdorff distance** | Precizie a marginilor |

### 3.3 Vizualizări pentru lucrare
- [ ] PCA pe embeddings: LeJEPA vs VasoJEPA (C8)
- [ ] 3 exemple de segmentare: ground truth, UNet fără pre-training, VasoJEPA
- [ ] Hartă de eroare: unde greșește UNet, unde corectează VasoJEPA

---

## Faza 4: Experimente suplimentare (~1 săptămână)

### 4.1 Ablație sistematică (C5)

| Variantă | Componente |
|----------|-----------|
| A | VasoJEPA full |
| B | Fără ghidaj (consens) |
| C | Fără DeepSIGReg (doar SIGReg cls) |
| D | Fără predictor |
| E | Fără tokeni anatomici |
| F | Doar SIGReg cls + ghidaj (cel mai simplu) |

### 4.2 Cross-dataset (C7)
Antrenezi pe un set, testezi pe altul:
```
Train: XA-170K  → Test: CADICA
Train: XA-170K  → Test: SYNTAX  
Train: XA-170K  → Test: XCAD
```

### 4.3 Evaluare colaterale (C6)
Calculează grosimea vaselor din măști și raportează Dice separat:
- Vase groase (>5 pixeli)
- Vase medii (2-5 pixeli)
- Colaterale (<2 pixeli)

---

## Faza 5: Scriere lucrare (~2 săptămâni)

### Structură propusă
```
1. Introducere (1 pag)
2. Related Work (2 pag)
   2.1 SSL în medical imaging
   2.2 JEPA și variante
   2.3 Segmentare vase coronariene
3. Metodă (3 pag)
   3.1 DeepSIGReg
   3.2 Ghidaj consensual SAM+Hessian
   3.3 Tokeni anatomici
   3.4 Predictor condiționat
   3.5 Loss total
4. Experimente (3 pag)
   4.1 Setup
   4.2 Ablație
   4.3 Cross-dataset
   4.4 Evaluare colaterale
   4.5 PCA embeddings
5. Discuții și limitări (1 pag)
6. Concluzii (0.5 pag)
```

---

## Timeline estimat

| Săptămâna | Activitate | Rezultat |
|:---------:|-----------|----------|
| 1 | Setup, SAM, Frangi, consens | Date pregătite |
| 2-3 | Implementare VasoJEPA | Model funcțional |
| 4-5 | Antrenare + debugging | Checkpoint-uri |
| 6 | Evaluare + baseline-uri | Tabele cu rezultate |
| 7 | Ablație + cross-dataset | Experimente complete |
| 8-9 | Scriere lucrare | Draft |
| 10 | Revizuire + corecturi | Final |

---

## Comenzi rapide

```bash
# Faza 0: Setup
pip install torch torchvision timm wandb scikit-image scipy segment-anything

# Faza 0.3: SAM pe toate imaginile
python scripts/run_sam.py --input-dir data/images --output-dir data/pseudo_masks

# Faza 0.4: Frangi
python scripts/compute_frangi.py --input-dir data/images --output-dir data/frangi

# Faza 0.5: Consens
python scripts/compute_consensus.py --input-dir data/images --output-dir data/consensus

# Faza 2: Antrenare
python train_vasojepa.py --epochs 200 --batch-size 64 --lr 5e-4

# Faza 3: Linear probing
python eval_linear_probe.py --checkpoint checkpoints/vasojepa/best.pth

# Faza 3: Fine-tuning UNet
python train_unet.py --pretrained-encoder checkpoints/vasojepa/best.pth --epochs 100
```

---

**Data:** 3 Iunie 2026
**Status:** Plan de implementare — de urmat în ordine.