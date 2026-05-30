from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from pathlib import Path
import sys, shutil, json, threading
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_demo.crypto_utils import ensure_rsa_keys, hybrid_encrypt
from engine.inference_stenosis import _run_pipeline as _orig_run_pipeline
from types import SimpleNamespace
import torch, cv2, numpy as np

app = Flask(__name__)
BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "uploads"
ENCRYPTED_DIR = BASE / "encrypted"
RESULTS_DIR = BASE / "results"
STATUS_DIR = BASE / "status"

ensure_rsa_keys()

def _ns(**kw):
    return SimpleNamespace(**kw)

def _set_status(fname, status, progress):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_DIR / f"{Path(fname).stem}.json", "w") as f:
        json.dump({"status": status, "progress": progress}, f)

def _encrypt_file(image_path, out_dir, suffix=""):
    raw = open(image_path, "rb").read()
    pkg = hybrid_encrypt(raw)
    meta = {"original_name": image_path.name, "size": len(raw), **pkg}
    jp = out_dir / f"{image_path.stem}{suffix}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(jp, "w") as f:
        json.dump(meta, f, indent=2)
    return meta

def _make_split_overlays(src_path, result_dir, overlay_path):
    od = result_dir / "overlay"
    vessel_dir = result_dir / "vessel"
    stenosis_dir = result_dir / "stenosis"
    if not vessel_dir.is_dir() or not stenosis_dir.is_dir():
        return
    vfiles = sorted(vessel_dir.glob("*"))
    sfiles = sorted(stenosis_dir.glob("*"))
    if not vfiles or not sfiles:
        return
    vm = cv2.imread(str(vfiles[0]), cv2.IMREAD_GRAYSCALE)
    sm = cv2.imread(str(sfiles[0]), cv2.IMREAD_GRAYSCALE)
    if vm is None or sm is None:
        return
    h, w = vm.shape
    va = np.zeros((h, w, 4), dtype=np.uint8)
    va[(vm > 127)] = [0, 255, 0, 200]
    sa = np.zeros((h, w, 4), dtype=np.uint8)
    sa[(sm > 127)] = [255, 0, 0, 200]
    cv2.imwrite(str(od / f"{overlay_path.stem}_vessel.png"), va)
    cv2.imwrite(str(od / f"{overlay_path.stem}_stenosis.png"), sa)

# ── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return render_template("index.html", error="Selecteaza un fisier.")
    dest = UPLOAD_DIR / f.filename
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    f.save(dest)
    enc = _encrypt_file(dest, ENCRYPTED_DIR)
    _set_status(f.filename, "encrypted", "Criptat AES-256 + RSA")
    threading.Thread(target=_run_ai, args=(f.filename,), daemon=True).start()
    return render_template("analyzing.html", filename=f.filename, enc=enc)

@app.route("/view/<filename>")
def view(filename):
    rd = RESULTS_DIR / Path(filename).stem
    if not (rd / "stenosis").is_dir():
        return redirect(url_for("index"))
    ov = sorted((rd / "overlay").glob("*")) if (rd / "overlay").is_dir() else []
    erm = ENCRYPTED_DIR / f"{Path(filename).stem}_result.json"
    rem = json.load(open(erm)) if erm.is_file() else None
    return render_template("view.html", filename=filename,
        overlay_name=ov[0].name if ov else None, result_enc_meta=rem)

@app.route("/status/<filename>")
def status(filename):
    sf = STATUS_DIR / f"{Path(filename).stem}.json"
    if sf.is_file():
        return jsonify(json.load(open(sf)))
    return jsonify({"status": "starting", "progress": "..."})

@app.route("/original/<filename>")
def original(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)

@app.route("/result/<filename>/<img_name>")
def result_image(filename, img_name):
    return send_from_directory(str(RESULTS_DIR / Path(filename).stem / "overlay"), img_name)

@app.route("/encrypted/<filename>")
def encrypted_data(filename):
    ep = ENCRYPTED_DIR / f"{Path(filename).stem}.json"
    if ep.is_file():
        return jsonify(json.load(open(ep)))
    return jsonify({"error": "not found"}), 404

@app.route("/encrypted_result/<filename>")
def encrypted_result_data(filename):
    ep = ENCRYPTED_DIR / f"{Path(filename).stem}_result.json"
    if ep.is_file():
        return jsonify(json.load(open(ep)))
    return jsonify({"error": "not found"}), 404

# ── AI Thread ───────────────────────────────────────────────────────────

import tempfile
from pathlib import Path

def _run_ai(filename):
    try:
        src = UPLOAD_DIR / filename
        out_dir = RESULTS_DIR / src.stem
        if out_dir.is_dir():
            shutil.rmtree(out_dir)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            work_file = tmpdir / filename
            shutil.copy2(src, work_file)

            _set_status(filename, "running", "Pipeline AI in executie...")

            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args = _ns(
                device=str(dev), img_size=512, tta=False, use_amp=torch.cuda.is_available(),
                precision="bfloat16", strict=False, lora_scale=None, lora_alpha=None,
                skip_mismatch=False, dw_fix="none",
                patient_id=None, patient_csv=None,
                input_mask_threshold=0.45, intermediate_threshold=0.45,
                threshold=0.5, save_prob=False,
                save_intermediate=True, save_overlay=True,
                overlay_alpha=0.5, overlay_vessel_threshold=None, overlay_stenosis_threshold=None,
                vessel_color="0,255,0", stenosis_color="255,0,0",
                dims="64,128,256,512", depths="2,2,3,2", drop_path_rate=0.0,
                encoder_name="efficientnetv2_s", encoder_pretrained=False, encoder_img_size=512,
                model_name="VesselNetV2EfficientEncoder", in_chans=4, num_classes=1,
                unext_base_channels=128, unext_depths="2,1,1",
                unext_mlp_ratio=4.0, unext_drop_rate=0.0, unext_attention=True,
                vessel_model_name="VesselNetV2EfficientEncoder",
                refiner_model_name="VesselNetV2EfficientEncoder",
                stenosis_model_name="VesselNetV2EfficientEncoder",
                vessel_in_chans=4, refiner_in_chans=5, stenosis_in_chans=4,
                vessel_encoder_name=None, refiner_encoder_name=None, stenosis_encoder_name=None,
                vessel_encoder_img_size=None, refiner_encoder_img_size=None, stenosis_encoder_img_size=None,
                vessel_input_mode="image4", refiner_input_mode="image4+mask", stenosis_input_mode="image4",
                vessel_checkpoint="Demo/best_model.pth",
                refiner_checkpoint="Demo/model_refiner_v2/best_refiner.pth",
                stenosis_checkpoint="Demo/best_stenosis.pth",
                input_dir=str(tmpdir), output_dir=str(out_dir),
                max_images=1, overlay_dir=str(out_dir / "overlay"),
                pipeline=True, checkpoint=None, input_mode="image4")

            _orig_run_pipeline(args)

        ovs = sorted((out_dir / "overlay").glob("*"))
        if ovs:
            _set_status(filename, "running", "Generare overlays...")
            _make_split_overlays(src, out_dir, ovs[0])
            _set_status(filename, "running", "Criptare rezultat...")
            _encrypt_file(ovs[0], ENCRYPTED_DIR, suffix="_result")

        _set_status(filename, "done", "Gata!")
    except Exception as e:
        _set_status(filename, "error", str(e))
        import traceback; traceback.print_exc()

def main():
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

if __name__ == "__main__":
    main()