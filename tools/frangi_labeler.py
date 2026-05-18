"""
Frangi Labeler — Gradio 6. Click Frangi to seed + flood fill. Eraser + Smart Grow.
Saves JSON with data/label paths for UNeXt training.

Usage:
    python tools/frangi_labeler.py  →  http://127.0.0.1:7860
"""
import os, json, random, time
from pathlib import Path
import numpy as np
import cv2
import gradio as gr

PRETRAIN_DIR = Path("data/extra/trainB")
ACCEPTED_DIR = Path("data/frangi_pseudolabels/accepted")
SKIPPED_LOG = Path("data/frangi_pseudolabels/skipped.txt")
STATE_FILE = Path("data/frangi_pseudolabels/state.json")
os.makedirs(ACCEPTED_DIR, exist_ok=True)

ALL_IMAGES = sorted(p for p in Path(PRETRAIN_DIR).rglob("*")
                     if p.suffix.lower() in {'.png','.jpg','.jpeg'})
ALL_IMAGES = [str(p) for p in ALL_IMAGES]
print(f"[Labeler] {len(ALL_IMAGES)} images")

_CURRENT_IDX = 0
try:
    with open(STATE_FILE) as f: _CURRENT_IDX = json.load(f).get("idx", 0)
except: pass

# ─── Frangi ───────────────────────────────
def norm(x, lo=1, hi=99):
    l, h = np.percentile(x, lo), np.percentile(x, hi)
    if h <= l + 1e-8: return np.clip((x-x.min())/(x.max()-x.min()+1e-8),0,1)
    return np.clip((x-l)/(h-l),0,1).astype(np.float32)

def frangi_v(img, smin=1, smax=4, beta=0.5, c=15.0):
    img_f = img.astype(np.float64)/255.0; resp = np.zeros_like(img_f,dtype=np.float64)
    for s in range(int(smin), int(smax)+1):
        bl = cv2.GaussianBlur(img_f,(0,0),sigmaX=s); s2=s*s
        dxx=cv2.Sobel(bl,cv2.CV_64F,2,0,ksize=3)*s2
        dxy=cv2.Sobel(bl,cv2.CV_64F,1,1,ksize=3)*s2
        dyy=cv2.Sobel(bl,cv2.CV_64F,0,2,ksize=3)*s2
        tr=dxx+dyy; det=dxx*dyy-dxy*dxy
        disc=np.maximum(0,tr*tr-4*det); sd=np.sqrt(disc)
        l1=0.5*(tr+sd); l2=0.5*(tr-sd)
        rb=np.abs(l1)/(np.abs(l2)+1e-8)
        sn=np.sqrt(l1*l1+l2*l2)
        v=np.exp(-(rb*rb)/(2*beta*beta))*(1-np.exp(-(sn*sn)/(2*c*c)))
        v[l2>0]=0; v=np.nan_to_num(v); resp=np.maximum(resp,v)
    return norm(resp)

def load_all(img_path, smin, smax, beta, c, pctl, min_a, invert, denoise, clahe, tophat):
    """Returns (orig_rgb, frangi_jet, mask_rgb, full_vesselness)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None, None, None
    _,fg=cv2.threshold(img,10,255,cv2.THRESH_BINARY)
    coords=cv2.findNonZero(fg)
    if coords is not None:
        x,y,w,h=cv2.boundingRect(coords)
        img=img[max(0,y-5):min(img.shape[0],y+h+5),max(0,x-5):min(img.shape[1],x+w+5)]
    out=img.astype(np.uint8)
    if denoise: out=cv2.bilateralFilter(out,7,30,30)
    if clahe: out=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(out)
    if tophat:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        out=cv2.addWeighted(out,1.0,cv2.morphologyEx(out,cv2.MORPH_TOPHAT,k),0.5,0)
    inp=255-out if invert else out
    v=frangi_v(inp,smin,smax,beta,c)
    thr=np.percentile(v,pctl)
    mask255=(v>thr).astype(np.uint8)*255
    if min_a>0:
        n,lbl,st,_=cv2.connectedComponentsWithStats(mask255,8)
        if n>1:
            c2=np.zeros_like(mask255)
            for i in range(1,n):
                if st[i,cv2.CC_STAT_AREA]>=min_a: c2[lbl==i]=255
            mask255=c2
    # Resize for display
    scale=min(512/out.shape[1],512/out.shape[0])
    nw,nh=int(out.shape[1]*scale),int(out.shape[0]*scale)
    orig_rgb=cv2.resize(cv2.cvtColor(cv2.imread(str(img_path)),cv2.COLOR_BGR2RGB),(nw,nh))
    v_jet=cv2.resize(cv2.applyColorMap((v*255).astype(np.uint8),cv2.COLORMAP_JET),(nw,nh))
    mask_rgb=cv2.resize(mask255,(nw,nh),interpolation=cv2.INTER_NEAREST)
    mask_rgb=cv2.cvtColor(mask_rgb,cv2.COLOR_GRAY2RGB)
    return orig_rgb, v_jet, mask_rgb, v

def smart_grow_img(img_path, smin, smax, beta, c, pctl, min_a, invert, de, cla, top, gthr, ga):
    img=cv2.imread(str(img_path),cv2.IMREAD_GRAYSCALE);
    if img is None: return None
    _,fg=cv2.threshold(img,10,255,cv2.THRESH_BINARY); coords=cv2.findNonZero(fg)
    if coords is not None:
        x,y,w,h=cv2.boundingRect(coords); img=img[max(0,y-5):min(img.shape[0],y+h+5),max(0,x-5):min(img.shape[1],x+w+5)]
    out=img.astype(np.uint8)
    if de: out=cv2.bilateralFilter(out,7,30,30)
    if cla: out=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(out)
    if top:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)); out=cv2.addWeighted(out,1.0,cv2.morphologyEx(out,cv2.MORPH_TOPHAT,k),0.5,0)
    inp=255-out if invert else out; v=frangi_v(inp,smin,smax,beta,c)
    seeds=(v>np.percentile(v,pctl)).astype(np.uint8)
    gb=(v>gthr).astype(np.uint8)
    n,lbl,st,_=cv2.connectedComponentsWithStats(gb.astype(np.uint8)*255,8)
    if n<=1: result=seeds
    else:
        result=np.zeros_like(seeds)
        for i in range(1,n):
            if st[i,cv2.CC_STAT_AREA]>=ga and np.any((lbl==i)&(seeds>0)): result[lbl==i]=255
        if result.max()==0: result=seeds
    sc=min(512/result.shape[1],512/result.shape[0])
    result=cv2.resize(result,(int(result.shape[1]*sc),int(result.shape[0]*sc)),interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(result,cv2.COLOR_GRAY2RGB)

def flood_from_click(img_path, ev: gr.SelectData, smin, smax, beta, c, pctl, min_a, invert, de, cla, top):
    """Click on Frangi image → flood fill from that point on vesselness."""
    ox, oy = ev.index  # click coords in displayed image
    img=cv2.imread(str(img_path),cv2.IMREAD_GRAYSCALE);
    if img is None: return None
    _,fg=cv2.threshold(img,10,255,cv2.THRESH_BINARY); coords=cv2.findNonZero(fg)
    if coords is not None:
        x,y,w,h=cv2.boundingRect(coords); img=img[max(0,y-5):min(img.shape[0],y+h+5),max(0,x-5):min(img.shape[1],x+w+5)]
    out=img.astype(np.uint8)
    if de: out=cv2.bilateralFilter(out,7,30,30)
    if cla: out=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(out)
    if top:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)); out=cv2.addWeighted(out,1.0,cv2.morphologyEx(out,cv2.MORPH_TOPHAT,k),0.5,0)
    inp=255-out if invert else out; v=frangi_v(inp,smin,smax,beta,c)
    oh,ow=inp.shape
    scale=min(512/ow,512/oh)
    fx=int(ox/scale); fy=int(oy/scale); fx=min(max(fx,0),ow-1); fy=min(max(fy,0),oh-1)
    sv=v[fy,fx]
    flood_img=(v*255).astype(np.uint8)
    mask=np.zeros((oh+2,ow+2),dtype=np.uint8)
    cv2.floodFill(flood_img,mask,(fx,fy),255,int(max(0,sv-0.15)*255),int((1-sv)*255),cv2.FLOODFILL_MASK_ONLY)
    ff=mask[1:-1,1:-1].astype(np.uint8)*255
    nw,nh=int(ow*scale),int(oh*scale); ff=cv2.resize(ff,(nw,nh),interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(ff,cv2.COLOR_GRAY2RGB)

def save_image(edited_img, smin, smax, beta, c, pctl, min_a, invert, de, cla, top):
    global _CURRENT_IDX
    if _CURRENT_IDX>=len(ALL_IMAGES): return "DONE", None, None, None
    ip=ALL_IMAGES[_CURRENT_IDX]
    if edited_img is None: return "No mask", None, None, None
    if isinstance(edited_img,dict): edited_img=edited_img.get('composite',edited_img.get('background'))
    if edited_img is None: return "No composite", None, None, None
    if isinstance(edited_img,np.ndarray):
        if edited_img.ndim==3 and edited_img.shape[2]==4: edited_img=cv2.cvtColor(edited_img,cv2.COLOR_RGBA2GRAY)
        elif edited_img.ndim==3: edited_img=cv2.cvtColor(edited_img,cv2.COLOR_RGB2GRAY)
    _,saved=cv2.threshold(edited_img,127,255,cv2.THRESH_BINARY)
    src_name=Path(ip).parent.name; base=Path(ip).stem; save_name=f"{src_name}_{base}_frangi.png"
    cv2.imwrite(str(ACCEPTED_DIR / save_name),saved)
    # JSON with data/label paths for UNeXt training
    meta={"data":str(ip),"label":str(ACCEPTED_DIR / save_name),
          "params":{"sigma_min":smin,"sigma_max":smax,"beta":beta,"c":c,
                    "percentile":pctl,"min_area":min_a,"invert":bool(invert),
                    "denoise":bool(de),"clahe":bool(cla),"tophat":bool(top)}}
    with open(str(ACCEPTED_DIR / save_name)+".json","w") as f: json.dump(meta,f,indent=2)
    _CURRENT_IDX+=1
    with open(STATE_FILE,"w") as f: json.dump({"idx":_CURRENT_IDX},f)
    return f"SAVED [{_CURRENT_IDX}/{len(ALL_IMAGES)}]", *load_display()

def skip_image():
    global _CURRENT_IDX
    if _CURRENT_IDX<len(ALL_IMAGES):
        with open(SKIPPED_LOG,"a") as f: f.write(ALL_IMAGES[_CURRENT_IDX]+"\n")
    _CURRENT_IDX+=1
    with open(STATE_FILE,"w") as f: json.dump({"idx":_CURRENT_IDX},f)
    return f"SKIPPED [{_CURRENT_IDX}/{len(ALL_IMAGES)}]", *load_display()

def load_display():
    global _CURRENT_IDX
    if _CURRENT_IDX>=len(ALL_IMAGES): return None,None,None,"DONE"
    o,v,m,_=load_all(ALL_IMAGES[_CURRENT_IDX],1,4,0.5,15,92,30,True,True,True,True)
    return o,v,m,f"[{_CURRENT_IDX+1}/{len(ALL_IMAGES)}] {Path(ALL_IMAGES[_CURRENT_IDX]).name}"

# ─── UI ───────────────────────────────────
with gr.Blocks(title="Frangi Labeler") as demo:
    gr.Markdown("# Frangi Labeler — Click, Grow, Erase, Save")
    gr.Markdown("**Click** on Frangi vesselness image → flood fill. Use **ImageEditor** to brush white/black. **Smart Grow** expands from high-percentile seeds.")

    # Images: full width, 3 columns horizontal
    with gr.Row(equal_height=True):
        orig_img = gr.Image(label="Original", height=400)
        frangi_img = gr.Image( height=400, interactive=True)
        mask_editor = gr.ImageEditor(
                                     height=400, eraser=True,
                                     brush=gr.Brush(colors=["#000000","#FFFFFF"],default_color="#000000"),
                                     layers=False, fixed_canvas=True)
    # Sliders: below images
    with gr.Group():
        with gr.Row():
            smin=gr.Slider(1,6,1,1,label="Sigma min",scale=1)
            smax=gr.Slider(1,6,4,1,label="Sigma max",scale=1)
            beta=gr.Slider(0.1,2.0,0.5,0.1,label="Beta",scale=1)
            cval=gr.Slider(1,30,15,0.5,label="C",scale=1)
            pctl=gr.Slider(50,99,92,1,label="Seed %ile",scale=1)
            min_a=gr.Slider(0,200,30,5,label="Filter area",scale=1)
        with gr.Row():
            gthr=gr.Slider(0.05,0.95,0.3,0.05,label="Grow thresh",scale=1)
            ga=gr.Slider(10,500,50,10,label="Min grow area",scale=1)
        with gr.Row():
            inv=gr.Checkbox(True,label="Invert")
            de=gr.Checkbox(True,label="Denoise")
            cla=gr.Checkbox(True,label="CLAHE")
            top=gr.Checkbox(True,label="Tophat")

    status=gr.Textbox(label="Status")
    hidden_path=gr.Textbox(visible=False)
    all_p=[smin,smax,beta,cval,pctl,min_a,inv,de,cla,top]
    grow_p=all_p+[gthr,ga]

    def refresh(*a):
        o,v,m,_=load_all(ALL_IMAGES[_CURRENT_IDX],*a)
        return o,v,m,ALL_IMAGES[_CURRENT_IDX]
    for ctrl in all_p:
        ctrl.change(fn=refresh,inputs=all_p,outputs=[orig_img,frangi_img,mask_editor,hidden_path])

    # Smart Grow
    grow_btn=gr.Button("🌱 Smart Grow",variant="secondary")
    grow_btn.click(fn=lambda *a: smart_grow_img(ALL_IMAGES[_CURRENT_IDX],*a) if _CURRENT_IDX<len(ALL_IMAGES) else None,
                   inputs=grow_p,outputs=mask_editor)

    # Click on Frangi → flood fill
    def click_flood(ev:gr.SelectData,*p):
        if _CURRENT_IDX>=len(ALL_IMAGES): return None
        return flood_from_click(ALL_IMAGES[_CURRENT_IDX],ev,*p)
    frangi_img.select(fn=click_flood,inputs=all_p,outputs=mask_editor)

    with gr.Row():
        save_btn=gr.Button("✅ SAVE",variant="primary",size="lg")
        skip_btn=gr.Button("⏭️ SKIP",size="lg")

    save_btn.click(fn=lambda *a: save_image(a[0],*a[1:11]),
                   inputs=[mask_editor]+all_p,
                   outputs=[status,orig_img,frangi_img,mask_editor])
    skip_btn.click(fn=skip_image,outputs=[status,orig_img,frangi_img,mask_editor])

    demo.load(fn=load_display,outputs=[orig_img,frangi_img,mask_editor,status])

if __name__=="__main__":
    demo.launch(server_name="127.0.0.1",server_port=7861,theme="soft")
