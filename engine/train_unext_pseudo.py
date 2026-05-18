"""
Train UNeXt on curated Frangi pseudo-labels.

Usage:
    python engine/train_unext_pseudo.py --epochs 200
"""
import os, sys, json, random, argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'XA-SSL-REPO', 'segmodel'))
from unext.model import UNext_S

PSEUDO_DIR = Path("data/frangi_pseudolabels/accepted")
CHECKPOINT_DIR = Path("checkpoints/unext_pseudo")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def load_pairs(pseudo_dir, min_vessel_pixels=50):
    pairs = []
    empty = 0
    for f in sorted(pseudo_dir.glob("*.png")):
        jp = Path(str(f) + ".json")
        if not jp.exists():
            continue
        with open(jp) as jf:
            meta = json.load(jf)
        ip = meta.get("data"); lp = meta.get("label")
        if not ip or not lp or not os.path.exists(ip) or not os.path.exists(lp):
            continue
        mask = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if mask is None or (mask > 0).sum() < min_vessel_pixels:
            empty += 1; continue
        pairs.append((ip, lp))
    print(f"Loaded {len(pairs)} pairs ({empty} empty/skipped)")
    return pairs

class PseudoDataset(Dataset):
    def __init__(self, pairs, img_size=224, augment=True):
        self.pairs = pairs; self.img_size = img_size; self.augment = augment
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        ip, lp = self.pairs[idx]
        img = cv2.imread(ip)
        if img is None: return self.__getitem__((idx+1)%len(self.pairs))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if mask is None: return self.__getitem__((idx+1)%len(self.pairs))
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        if self.augment:
            if random.random()>0.5: img=cv2.flip(img,1); mask=cv2.flip(mask,1)
            if random.random()>0.5: img=cv2.flip(img,0); mask=cv2.flip(mask,0)
            if random.random()>0.3:
                ang=random.uniform(-15,15)
                M=cv2.getRotationMatrix2D((self.img_size//2,self.img_size//2),ang,1.0)
                img=cv2.warpAffine(img,M,(self.img_size,self.img_size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
                mask=cv2.warpAffine(mask,M,(self.img_size,self.img_size),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT)
        img=img.astype(np.float32)/255.0
        mean=np.array([0.485,0.456,0.406],dtype=np.float32); std=np.array([0.229,0.224,0.225],dtype=np.float32)
        img=(img-mean)/std
        return torch.from_numpy(img).permute(2,0,1).float(), torch.from_numpy(mask).unsqueeze(0).float()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5, smooth=1e-5):
        super().__init__(); self.bce_w=bce_w; self.smooth=smooth
    def forward(self, p, t):
        p=torch.sigmoid(p); bce=F.binary_cross_entropy(p,t)
        pf=p.view(-1); tf=t.view(-1)
        inter=(pf*tf).sum()
        dice=1-(2*inter+self.smooth)/(pf.sum()+tf.sum()+self.smooth)
        return self.bce_w*bce+(1-self.bce_w)*dice

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--img-size",type=int,default=224)
    args=parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    pairs=load_pairs(PSEUDO_DIR)
    if not pairs: print("ERROR: no pairs"); return
    random.shuffle(pairs); split=int(0.8*len(pairs))
    train_ds=PseudoDataset(pairs[:split],args.img_size,augment=True)
    val_ds=PseudoDataset(pairs[split:],args.img_size,augment=False)
    tl=DataLoader(train_ds,args.batch_size,shuffle=True,num_workers=4)
    vl=DataLoader(val_ds,args.batch_size,shuffle=False,num_workers=4)
    model=UNext_S(num_classes=1,img_size=args.img_size).to(DEVICE)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    crit=BCEDiceLoss(); optim=AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    warmup=LinearLR(optim,0.1,1.0,total_iters=min(10,args.epochs))
    cosine=CosineAnnealingLR(optim,T_max=max(1,args.epochs-10))
    sched=SequentialLR(optim,[warmup,cosine],[min(10,args.epochs)])
    best=float('inf')
    for ep in range(args.epochs):
        model.train(); tl_loss=0
        for im,ma in tqdm(tl,desc=f"E{ep+1}/{args.epochs}"):
            im,ma=im.to(DEVICE),ma.to(DEVICE)
            optim.zero_grad(); pr=model(im); lo=crit(pr,ma); lo.backward(); optim.step()
            tl_loss+=lo.item()
        model.eval(); vl_loss=0
        with torch.no_grad():
            for im,ma in vl:
                im,ma=im.to(DEVICE),ma.to(DEVICE)
                vl_loss+=crit(model(im),ma).item()
        sched.step()
        tl_loss/=len(tl); vl_loss/=len(vl)
        print(f"  Train:{tl_loss:.4f} Val:{vl_loss:.4f} LR:{optim.param_groups[0]['lr']:.2e}")
        if vl_loss<best:
            best=vl_loss; torch.save(model.state_dict(),CHECKPOINT_DIR/"unext_pseudo_best.pth")
            print(f"  *** Saved best")
    torch.save(model.state_dict(),CHECKPOINT_DIR/"unext_pseudo_final.pth")
    print(f"Done. Best val: {best:.4f}")

if __name__=="__main__":
    main()
