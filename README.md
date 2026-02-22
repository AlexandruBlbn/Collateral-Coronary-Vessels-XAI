# Collateral Coronary Vessels Segmentation & XAI 🫀

Acest repository conține un framework avansat de Deep Learning dedicat segmentării vaselor coronariene și a circulației colaterale din imagini de angiografie X-Ray. 
Proiectul abordează segmentarea imaginilor medicale folosind paradigme de Self-Supervised Learning (SSL) de ultimă generație, urmate de Transfer Learning și Fine-Tuning pe arhitecturi moderne de Dense Prediction.

## 🚀 Caracteristici Principale

* **Pre-antrenare Self-Supervised (SSL):** Suportă antrenarea backbone-urilor folosind algoritmi precum LeJEPA cu regularizare SIGReg.
* **SimMIM (Masked Image Modeling):** Suport pentru pre-antrenare prin reconstrucția patch-urilor mascate.
* **DINOv3:** Integrare Self-Distillation cu loss-uri avansate precum iBOT, KoLeo și Gram.
* **Linear Probing Simultan:** Evaluarea capacității de extragere a trăsăturilor în timp real în timpul pre-antrenării, folosind un cap de decodare multi-scale strict liniar.
* **Backbones Moderne:** Integrare cu librăria `timm` pentru suport nativ al modelelor hibride și Transformer precum SwinV2, ConvNeXt și ViT.
* **Segmentare (Fine-Tuning):** Decodere customizate incluzând Attention U-Net, U-Net++ (Nested) și UNeXt.

## 📂 Structura Proiectului

* `config/`: Fișiere de configurare YAML pentru experimente precum DINO, LeJEPA, SimMIM și segmentare.
* `data/`: Logica de încărcare a datelor folosind ARCADE dataset, augmentări SSL și Transforms.
* `engine/`: Scripturi principale de execuție pentru pre-antrenare (LeJEPA, SimMIM) și fine-tuning.
* `utils/`: Funcții ajutătoare pentru reproductibilitate (set_seed).
* `zoo/`: Definirea modelelor, componentelor SSL și funcțiilor de Loss.

## ⚙️ Instalare și Cerințe

Sistemul necesită un mediu cu suport GPU (CUDA). Pentru a instala dependențele, rulează:

```bash
pip install -r engine/requirements.txt


generat cu gepete because i was lazy