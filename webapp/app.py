"""
EmotiSense — SDT Multimodal Emotion Recognition
Flask backend — full integration of all paper equations + novelty
"""

import os, sys, base64, traceback, uuid, time, csv
from datetime import datetime
from collections import defaultdict

_WEBAPP_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_WEBAPP_DIR)
for _p in [_PROJECT_ROOT, _WEBAPP_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from configs.config import Config
from models.sdt_model import build_model

app = Flask(__name__,
    template_folder=os.path.join(_WEBAPP_DIR, "templates"),
    static_folder=os.path.join(_WEBAPP_DIR, "static"))
CORS(app)

UPLOAD_FOLDER = os.path.join(_WEBAPP_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

ALLOWED_AUDIO = {"wav","mp3","flac","ogg","m4a","webm"}
ALLOWED_VIDEO = {"mp4","avi","mov","mkv","webm"}

MELD_LABELS   = ["neutral","surprise","fear","sadness","joy","disgust","anger"]
MELD_EMOJI    = ["😐","😲","😨","😢","😄","🤢","😠"]
MELD_COLORS   = ["#6b7280","#8b5cf6","#06b6d4","#3b82f6","#f59e0b","#10b981","#ef4444"]
MELD_MAP      = {l:i for i,l in enumerate(MELD_LABELS)}
print("Feature Added.")

MODEL = CFG = None
DEVICE = "cpu"
SPEAKER_MAP = {}
CONV_HISTORY = []
PRED_LOG = []


# ── Feature extractors (lazy) ─────────────────────────────────────────────────
_roberta_tok = _roberta_mdl = None
_smile = None
_densenet = _densenet_tf = None

def extract_text(text: str) -> np.ndarray:
    global _roberta_tok, _roberta_mdl
    try:
        if _roberta_tok is None:
            from transformers import RobertaTokenizer, RobertaModel
            _roberta_tok = RobertaTokenizer.from_pretrained("roberta-large")
            _roberta_mdl = RobertaModel.from_pretrained("roberta-large").eval()
        enc = _roberta_tok(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = _roberta_mdl(**enc)
        return out.last_hidden_state[:, 0, :].squeeze(0).numpy().astype(np.float32)
    except Exception:
        words = text.lower().split()
        feat  = np.zeros(1024, dtype=np.float32)
        POS = {"happy","joy","great","love","excited","wonderful","laugh","fun","good","amazing","yes"}
        NEG = {"sad","angry","fear","hate","terrible","cry","upset","scared","awful","bad","no"}
        for i, w in enumerate(words[:64]):
            rng = np.random.RandomState(hash(w) & 0x7FFFFFFF)
            feat[i*16:(i+1)*16] = rng.randn(16).astype(np.float32)
        n = max(len(words), 1)
        feat[1020] = sum(1 for w in words if w in POS) / n
        feat[1021] = sum(1 for w in words if w in NEG) / n
        feat[1022] = len(words) / 50.0
        feat[1023] = len(text) / 200.0
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm

def extract_audio(audio_bytes: bytes) -> np.ndarray:
    global _smile
    try:
        import opensmile, soundfile as sf, io
        if _smile is None:
            _smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals)
        data, sr = sf.read(io.BytesIO(audio_bytes))
        feat = _smile.process_signal(data, sr).values.flatten()
        feat = feat[:300] if len(feat) >= 300 else np.pad(feat, (0, 300-len(feat)))
        return feat.astype(np.float32)
    except Exception:
        try:
            import librosa, io
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).mean(axis=1)
            chroma= librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
            spec  = librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1)[:128]
            feat  = np.concatenate([mfcc, chroma, spec])
            feat  = feat[:300] if len(feat) >= 300 else np.pad(feat, (0, 300-len(feat)))
            return feat.astype(np.float32)
        except Exception:
            return np.zeros(300, dtype=np.float32)

def extract_visual(image_bytes: bytes) -> np.ndarray:
    global _densenet, _densenet_tf
    try:
        import torch, torchvision.models as tvm, torchvision.transforms as T
        import cv2, numpy as np
        if _densenet is None:
            _densenet = tvm.densenet121(pretrained=False)
            _densenet.classifier = torch.nn.Identity()
            _densenet.eval()
            _densenet_tf = T.Compose([T.ToPILImage(), T.Resize(224),
                                      T.CenterCrop(224), T.ToTensor(),
                                      T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("bad image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t   = _densenet_tf(img).unsqueeze(0)
        with torch.no_grad():
            feat = _densenet(t).squeeze(0).numpy()
        feat = feat[:342] if len(feat) >= 342 else np.pad(feat, (0, 342-len(feat)))
        return feat.astype(np.float32)
    except Exception:
        return np.zeros(342, dtype=np.float32)

def extract_audio_from_video(path: str) -> np.ndarray:
    try:
        import subprocess, tempfile
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg","-y","-i",path,"-ar","16000","-ac","1",tmp],
                       capture_output=True, timeout=30)
        with open(tmp,"rb") as f: data = f.read()
        os.remove(tmp)
        return extract_audio(data)
    except Exception:
        return np.zeros(300, dtype=np.float32)

def extract_visual_from_video(path: str) -> np.ndarray:
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        feats = []
        for idx in np.linspace(0, max(total-1,0), 8, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret: continue
            _, buf = cv2.imencode(".jpg", frame)
            feats.append(extract_visual(buf.tobytes()))
        cap.release()
        return np.mean(feats, axis=0) if feats else np.zeros(342, dtype=np.float32)
    except Exception:
        return np.zeros(342, dtype=np.float32)


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(ckpt_path=None):
    global MODEL, CFG, DEVICE, SPEAKER_MAP
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes  = 7
    hidden_dim   = 256
    num_heads    = 8
    ffn_dim      = 512
    intra_layers = 2
    inter_layers = 2
    max_speakers = 9
    dropout      = 0.1
    temperature  = 4.0
    dataset_name = "meld"

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if ckpt.get("dataset") == "meld":
            num_classes  = ckpt.get("cfg_num_classes",  7)
            hidden_dim   = ckpt.get("cfg_hidden_dim",   256)
            num_heads    = ckpt.get("cfg_num_heads",    8)
            ffn_dim      = ckpt.get("cfg_ffn_dim",      512)
            intra_layers = ckpt.get("cfg_intra_layers", 2)
            inter_layers = ckpt.get("cfg_inter_layers", 2)
            dropout      = ckpt.get("cfg_dropout",      0.1)
            temperature  = ckpt.get("cfg_temperature",  4.0)
            SPEAKER_MAP  = ckpt.get("speaker_map",      {})
            max_speakers = max(len(SPEAKER_MAP)+1, 9)
            print(f"[App] MELD checkpoint: {num_classes} classes, hidden={hidden_dim}")
    else:
        ckpt = None
        print(f"[App] No checkpoint at '{ckpt_path}' — using random weights")

    CFG = Config(dataset=dataset_name)
    CFG.model.hidden_dim       = hidden_dim
    CFG.model.num_heads        = num_heads
    CFG.model.ffn_dim          = ffn_dim
    CFG.model.num_intra_layers = intra_layers
    CFG.model.num_inter_layers = inter_layers
    CFG.model.num_classes      = num_classes
    CFG.model.max_speakers     = max_speakers
    CFG.model.dropout          = dropout
    CFG.model.temperature      = temperature
    CFG.model.text_dim         = 1024
    CFG.model.audio_dim        = 300
    CFG.model.visual_dim       = 342

    MODEL = build_model(CFG).to(DEVICE)
    if ckpt:
        MODEL.load_state_dict(ckpt["model_state"])
        print(f"[App] ✅ Loaded epoch {ckpt.get('epoch','?')} | val_wF1={ckpt.get('val_wf1',0):.4f}")
    MODEL.eval()
    print(f"[App] Ready on {DEVICE}")


# ── Core inference ────────────────────────────────────────────────────────────
def run_inference(text_feat, audio_feat, visual_feat, speaker_id=0):
    global CONV_HISTORY, PRED_LOG
    CONV_HISTORY.append({"text": text_feat, "audio": audio_feat,
                          "visual": visual_feat, "speaker": speaker_id})
    history = CONV_HISTORY[-20:]
    T = len(history)

    text_t   = torch.tensor(np.stack([h["text"]   for h in history]),
                             dtype=torch.float32).unsqueeze(0).to(DEVICE)
    audio_t  = torch.tensor(np.stack([h["audio"]  for h in history]),
                             dtype=torch.float32).unsqueeze(0).to(DEVICE)
    visual_t = torch.tensor(np.stack([h["visual"] for h in history]),
                             dtype=torch.float32).unsqueeze(0).to(DEVICE)
    spk_t    = torch.tensor([[h["speaker"] for h in history]],
                             dtype=torch.long).to(DEVICE)
    mask_t   = torch.ones(1, T, dtype=torch.bool).to(DEVICE)

    with torch.no_grad():
        out    = MODEL(text_t, audio_t, visual_t, spk_t, mask_t)
        logits = out["logits"]                              # (1,T,C)
        probs  = torch.softmax(logits, dim=-1).squeeze(0)  # (T,C)

    n = CFG.model.num_classes
    raw = probs[-1].cpu().numpy()
    if len(raw) >= n:
        last_probs = raw[:n]
    else:
        pad = np.full(n - len(raw), 1.0/n, dtype=np.float32)
        last_probs = np.concatenate([raw, pad])
    last_probs = last_probs / (last_probs.sum() + 1e-8)

    pred_id = int(np.argmax(last_probs))
    label   = MELD_LABELS[pred_id] if pred_id < len(MELD_LABELS) else "unknown"
    emoji   = MELD_EMOJI[pred_id % len(MELD_EMOJI)]
    color   = MELD_COLORS[pred_id % len(MELD_COLORS)]

    PRED_LOG.append({
        "pred_id":    pred_id,
        "emotion":    label,
        "confidence": float(last_probs[pred_id]),
        "all_probs":  [float(last_probs[i]) for i in range(n)],
        "ts":         datetime.now().strftime("%H:%M:%S"),
        "input_type": "",
    })

    return {
        "emotion_id":  pred_id,
        "emotion":     label,
        "emoji":       emoji,
        "color":       color,
        "confidence":  float(last_probs[pred_id]),
        "probabilities": [
            {"label": MELD_LABELS[i], "emoji": MELD_EMOJI[i%len(MELD_EMOJI)],
             "color": MELD_COLORS[i%len(MELD_COLORS)], "prob": float(last_probs[i])}
            for i in range(n)
        ],
        "context_length": T,
        "modalities_active": {
            "text":   bool(np.any(text_feat   != 0)),
            "audio":  bool(np.any(audio_feat  != 0)),
            "visual": bool(np.any(visual_feat != 0)),
        },
        "novelty": {
            "method":      "Adaptive Modality Fusion",
            "paper_eq":    "ctx = (Z_a + Z_v) / 2  [fixed average]",
            "our_eq":      "ctx = α_a·Z_a + α_v·Z_v  [softmax-learned per utterance]",
            "description": "Content-aware weights suppress noisy modalities dynamically",
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded":  MODEL is not None,
        "device":        DEVICE,
        "history_len":   len(CONV_HISTORY),
        "num_classes":   CFG.model.num_classes if CFG else 7,
        "labels":        MELD_LABELS,
        "checkpoint_info": {
            "val_wf1": "see experiments/logs/meld_train.log",
        },
        "novelty": "AdaptiveModalityFusion replaces fixed (Z_a+Z_v)/2 with learned α weights",
    })

@app.route("/api/reset", methods=["POST"])
def reset():
    global CONV_HISTORY, PRED_LOG
    CONV_HISTORY = []
    PRED_LOG     = []
    return jsonify({"ok": True})

# ── Text prediction ───────────────────────────────────────────────────────────
@app.route("/api/predict/text", methods=["POST"])
def predict_text():
    try:
        data    = request.get_json()
        text    = (data.get("text") or "").strip()
        speaker = int(data.get("speaker_id", 0))
        if not text:
            return jsonify({"error": "No text provided"}), 400
        t_feat = extract_text(text)
        result = run_inference(t_feat, np.zeros(300,np.float32), np.zeros(342,np.float32), speaker)
        PRED_LOG[-1]["input_type"] = "text"
        result.update({"input_type": "text", "input_text": text})
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Audio upload ──────────────────────────────────────────────────────────────
@app.route("/api/predict/audio_upload", methods=["POST"])
def predict_audio_upload():
    path = None
    try:
        if "audio" not in request.files or not request.files["audio"].filename:
            return jsonify({"error": "No audio file"}), 400
        f       = request.files["audio"]
        text    = request.form.get("text","").strip()
        speaker = int(request.form.get("speaker_id","0"))
        ext     = f.filename.rsplit(".",1)[-1].lower()
        if ext not in ALLOWED_AUDIO:
            return jsonify({"error": f"Unsupported format: {ext}"}), 400
        path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
        f.save(path)
        with open(path,"rb") as fh: audio_bytes = fh.read()
        t_feat = extract_text(text) if text else np.zeros(1024,np.float32)
        a_feat = extract_audio(audio_bytes)
        result = run_inference(t_feat, a_feat, np.zeros(342,np.float32), speaker)
        PRED_LOG[-1]["input_type"] = "audio_upload"
        result.update({"input_type":"audio_upload","input_text":text or f"[Audio: {f.filename}]","filename":f.filename})
        return jsonify(result)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500
    finally:
        if path and os.path.exists(path): os.remove(path)

# ── Video upload ──────────────────────────────────────────────────────────────
@app.route("/api/predict/video_upload", methods=["POST"])
def predict_video_upload():
    path = None
    try:
        if "video" not in request.files or not request.files["video"].filename:
            return jsonify({"error": "No video file"}), 400
        f       = request.files["video"]
        text    = request.form.get("text","").strip()
        speaker = int(request.form.get("speaker_id","0"))
        ext     = f.filename.rsplit(".",1)[-1].lower()
        if ext not in ALLOWED_VIDEO:
            return jsonify({"error": f"Unsupported format: {ext}"}), 400
        path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
        f.save(path)
        t_feat = extract_text(text) if text else np.zeros(1024,np.float32)
        a_feat = extract_audio_from_video(path)
        v_feat = extract_visual_from_video(path)
        result = run_inference(t_feat, a_feat, v_feat, speaker)
        PRED_LOG[-1]["input_type"] = "video_upload"
        result.update({"input_type":"video_upload","input_text":text or f"[Video: {f.filename}]","filename":f.filename})
        return jsonify(result)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500
    finally:
        if path and os.path.exists(path): os.remove(path)

# ── Multimodal upload ─────────────────────────────────────────────────────────
@app.route("/api/predict/multimodal", methods=["POST"])
def predict_multimodal():
    audio_path = video_path = None
    try:
        text    = request.form.get("text","").strip()
        speaker = int(request.form.get("speaker_id","0"))
        t_feat  = extract_text(text) if text else np.zeros(1024,np.float32)
        a_feat  = np.zeros(300,np.float32)
        v_feat  = np.zeros(342,np.float32)

        if "audio" in request.files and request.files["audio"].filename:
            af = request.files["audio"]
            ext = af.filename.rsplit(".",1)[-1].lower()
            if ext in ALLOWED_AUDIO:
                audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
                af.save(audio_path)
                with open(audio_path,"rb") as fh: a_feat = extract_audio(fh.read())

        if "video" in request.files and request.files["video"].filename:
            vf = request.files["video"]
            ext = vf.filename.rsplit(".",1)[-1].lower()
            if ext in ALLOWED_VIDEO:
                video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
                vf.save(video_path)
                v_feat = extract_visual_from_video(video_path)
                if not np.any(a_feat != 0):
                    a_feat = extract_audio_from_video(video_path)

        result = run_inference(t_feat, a_feat, v_feat, speaker)
        PRED_LOG[-1]["input_type"] = "multimodal"
        result.update({"input_type":"multimodal","input_text":text or "[Multimodal]"})
        return jsonify(result)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500
    finally:
        for p in [audio_path, video_path]:
            if p and os.path.exists(p): os.remove(p)

# ── Live mic audio ────────────────────────────────────────────────────────────
@app.route("/api/predict/live_audio", methods=["POST"])
def predict_live_audio():
    try:
        data      = request.get_json()
        audio_b64 = data.get("audio_data","")
        text      = data.get("text","").strip()
        speaker   = int(data.get("speaker_id",0))
        if not audio_b64:
            return jsonify({"error": "No audio_data"}), 400
        if "," in audio_b64: audio_b64 = audio_b64.split(",",1)[1]
        audio_bytes = base64.b64decode(audio_b64)
        t_feat = extract_text(text) if text else np.zeros(1024,np.float32)
        a_feat = extract_audio(audio_bytes)
        result = run_inference(t_feat, a_feat, np.zeros(342,np.float32), speaker)
        PRED_LOG[-1]["input_type"] = "live_audio"
        result.update({"input_type":"live_audio","input_text":text or "[Microphone]"})
        return jsonify(result)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500

# ── Live camera ───────────────────────────────────────────────────────────────
@app.route("/api/predict/live_video", methods=["POST"])
def predict_live_video():
    try:
        data      = request.get_json()
        image_b64 = data.get("image_data","")
        audio_b64 = data.get("audio_data","")
        text      = data.get("text","").strip()
        speaker   = int(data.get("speaker_id",0))
        if not image_b64:
            return jsonify({"error": "No image_data"}), 400
        if "," in image_b64: image_b64 = image_b64.split(",",1)[1]
        t_feat = extract_text(text) if text else np.zeros(1024,np.float32)
        v_feat = extract_visual(base64.b64decode(image_b64))
        a_feat = np.zeros(300,np.float32)
        if audio_b64:
            if "," in audio_b64: audio_b64 = audio_b64.split(",",1)[1]
            a_feat = extract_audio(base64.b64decode(audio_b64))
        result = run_inference(t_feat, a_feat, v_feat, speaker)
        PRED_LOG[-1]["input_type"] = "live_video"
        result.update({"input_type":"live_video","input_text":text or "[Camera]"})
        return jsonify(result)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500

# ── MELD CSV dataset prediction ───────────────────────────────────────────────
@app.route("/api/predict/meld_csv", methods=["POST"])
def predict_meld_csv():
    """Predict emotions for all utterances in an uploaded MELD CSV."""
    try:
        if "csv" not in request.files:
            return jsonify({"error": "No CSV file"}), 400
        f = request.files["csv"]
        content = f.read().decode("utf-8").splitlines()
        reader  = csv.DictReader(content)
        results = []
        MELD_MAP_LOCAL = {"neutral":0,"surprise":1,"fear":2,"sadness":3,"joy":4,"disgust":5,"anger":6}
        for row in reader:
            text = row.get("Utterance","").strip()
            true_emo = row.get("Emotion","").strip().lower()
            if not text: continue
            t_feat = extract_text(text)
            res    = run_inference(t_feat, np.zeros(300,np.float32), np.zeros(342,np.float32), 0)
            results.append({
                "utterance":    text,
                "speaker":      row.get("Speaker",""),
                "true_emotion": true_emo,
                "pred_emotion": res["emotion"],
                "confidence":   res["confidence"],
                "correct":      true_emo == res["emotion"],
                "probabilities":res["probabilities"],
            })
        correct = sum(1 for r in results if r["correct"])
        acc = correct / max(len(results), 1)
        return jsonify({"results": results, "total": len(results),
                        "correct": correct, "accuracy": round(acc, 4)})
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500

# ── Metrics ───────────────────────────────────────────────────────────────────
@app.route("/api/metrics")
def get_metrics():
    if not PRED_LOG:
        return jsonify({"total": 0, "labels": MELD_LABELS, "predictions": [],
                        "emotion_counts": {}, "avg_confidence": 0.0})
    n = CFG.model.num_classes if CFG else 7
    labels = MELD_LABELS[:n]
    counts = {}
    for p in PRED_LOG:
        counts[p["emotion"]] = counts.get(p["emotion"], 0) + 1
    avg_conf = sum(p["confidence"] for p in PRED_LOG) / len(PRED_LOG)
    avg_probs = [0.0] * n
    for p in PRED_LOG:
        for i, v in enumerate(p.get("all_probs", [])):
            if i < n: avg_probs[i] += v
    avg_probs = [v/len(PRED_LOG) for v in avg_probs]
    chart = [{"idx":i+1,"emotion":p["emotion"],"confidence":round(p["confidence"]*100,1),
              "input_type":p.get("input_type",""),"ts":p["ts"]}
             for i,p in enumerate(PRED_LOG)]
    return jsonify({
        "total": len(PRED_LOG),
        "labels": labels,
        "predictions": chart,
        "emotion_counts": counts,
        "avg_confidence": round(avg_conf*100, 1),
        "avg_probs_per_class": [round(v*100,2) for v in avg_probs],
        "per_class": [{"label":labels[i],"avg_prob":round(avg_probs[i]*100,2),
                       "count":counts.get(labels[i],0)} for i in range(n)],
    })

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="../experiments/checkpoints/best_model_meld.pt")
    p.add_argument("--port", default=5000, type=int)
    args = p.parse_args()
    load_model(args.checkpoint)
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
