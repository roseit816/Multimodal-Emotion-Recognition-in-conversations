"""
real_features.py
================
Paper-accurate feature extractors for SDT.

Text   : RoBERTa-Large [CLS] token        → dim 1024
Audio  : openSMILE ComParE_2016 / IS09    → dim 384 (padded/truncated to 300)
Visual : DenseNet121 global-avg-pool      → dim 1024 → projected to 342

All three modules lazy-load on first call and are reused across requests.
If a dependency is missing the module falls back with a printed warning.
"""

import io, os, hashlib, tempfile, warnings
warnings.filterwarnings("ignore")

import numpy as np

# ── availability flags (set at import time) ──────────────────────────────────
ROBERTA_AVAILABLE  = False
OPENSMILE_AVAILABLE= False
DENSENET_AVAILABLE = False

try:
    from transformers import RobertaTokenizer, RobertaModel
    import torch as _t
    ROBERTA_AVAILABLE = True
except ImportError:
    pass

try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    try:
        import librosa
        OPENSMILE_AVAILABLE = False   # librosa used as fallback below
    except ImportError:
        pass

try:
    import torch as _t2
    import torchvision.models as _tvm
    import torchvision.transforms as _T
    import cv2 as _cv2
    DENSENET_AVAILABLE = True
except ImportError:
    pass

# ── singletons ────────────────────────────────────────────────────────────────
_roberta_tok   = None
_roberta_mdl   = None
_smile         = None
_densenet_mdl  = None
_densenet_tfm  = None

# Fixed projection matrix: DenseNet 1024-d → 342-d (paper visual dim)
_PROJ_SEED = 2024
_rng_proj  = np.random.RandomState(_PROJ_SEED)
_PROJ_MAT  = _rng_proj.randn(1024, 342).astype(np.float32) / np.sqrt(342)  # (1024,342)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  TEXT — RoBERTa-Large [CLS]   (dim = 1024)
# ═════════════════════════════════════════════════════════════════════════════
def _get_roberta():
    global _roberta_tok, _roberta_mdl
    if _roberta_tok is None:
        print("[Features] Loading RoBERTa-Large … (first call ~30s)")
        _roberta_tok = RobertaTokenizer.from_pretrained("roberta-large")
        _roberta_mdl = RobertaModel.from_pretrained("roberta-large")
        _roberta_mdl.eval()
        print("[Features] RoBERTa-Large ready ✓")
    return _roberta_tok, _roberta_mdl


def extract_text_roberta(text: str) -> np.ndarray:
    """RoBERTa-Large [CLS] → float32 (1024,)"""
    if not text or not text.strip():
        return np.zeros(1024, dtype=np.float32)

    if ROBERTA_AVAILABLE:
        try:
            import torch
            tok, mdl = _get_roberta()
            enc = tok(text.strip(), return_tensors="pt",
                      truncation=True, max_length=128)
            with torch.no_grad():
                out = mdl(**enc)
            return out.last_hidden_state[:, 0, :].squeeze(0).numpy().astype(np.float32)
        except Exception as e:
            print(f"[Features] RoBERTa error ({e}) — falling back")

    # ── deterministic word-hash fallback ─────────────────────────────────────
    words = text.lower().split()
    feat  = np.zeros(1024, dtype=np.float32)
    for i, w in enumerate(words[:64]):
        rng = np.random.RandomState(hash(w) & 0x7FFFFFFF)
        feat[i * 16:(i + 1) * 16] = rng.randn(16).astype(np.float32)
    POS = {"happy","joy","great","love","excited","wonderful","laugh","fun","good","amazing"}
    NEG = {"sad","angry","fear","hate","terrible","cry","upset","scared","awful","bad"}
    n   = max(len(words), 1)
    feat[1020] = sum(1 for w in words if w in POS) / n
    feat[1021] = sum(1 for w in words if w in NEG) / n
    feat[1022] = len(words) / 50.0
    feat[1023] = len(text)  / 200.0
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm


# ═════════════════════════════════════════════════════════════════════════════
# 2.  AUDIO — openSMILE ComParE_2016   (→ 300-d)
#     Falls back to librosa MFCC if opensmile not installed.
# ═════════════════════════════════════════════════════════════════════════════
def _get_smile():
    global _smile
    if _smile is None:
        print("[Features] Initialising openSMILE ComParE_2016 …")
        import opensmile
        _smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        print("[Features] openSMILE ready ✓")
    return _smile


def _audio_bytes_to_wav(audio_bytes: bytes) -> str:
    """Save audio bytes to a temp WAV file; returns path."""
    suffix = ".webm"
    # detect if it's already a WAV
    if audio_bytes[:4] == b"RIFF":
        suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def _convert_to_wav(path: str) -> str:
    """Convert any audio file to WAV 16 kHz mono using ffmpeg if available."""
    wav_path = path + "_conv.wav"
    ret = os.system(f'ffmpeg -y -i "{path}" -ar 16000 -ac 1 "{wav_path}" -loglevel quiet 2>/dev/null')
    if ret == 0 and os.path.exists(wav_path):
        return wav_path
    return path   # return original if ffmpeg not available


def extract_audio_opensmile(audio_bytes: bytes) -> np.ndarray:
    """
    openSMILE ComParE_2016 functionals → float32 (300,).

    ComParE_2016 gives 6373 features; we take the first 300
    (low-level descriptors: energy, MFCC, ZCR, F0, jitter, shimmer …).
    If opensmile isn't installed, falls back to librosa MFCC.
    """
    feat = np.zeros(300, dtype=np.float32)
    if not audio_bytes or len(audio_bytes) < 100:
        return feat

    if OPENSMILE_AVAILABLE:
        tmp_path = wav_path = None
        try:
            tmp_path = _audio_bytes_to_wav(audio_bytes)
            wav_path = _convert_to_wav(tmp_path)
            smile    = _get_smile()
            df       = smile.process_file(wav_path)
            raw      = df.values.flatten().astype(np.float32)
            L        = min(len(raw), 300)
            feat[:L] = raw[:L]
            # replace NaN/Inf
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            norm = np.linalg.norm(feat) + 1e-8
            return feat / norm
        except Exception as e:
            print(f"[Features] openSMILE error ({e}) — trying librosa fallback")
        finally:
            for p in [tmp_path, wav_path]:
                try:
                    if p and os.path.exists(p): os.remove(p)
                except Exception:
                    pass

    # ── librosa MFCC fallback ─────────────────────────────────────────────────
    try:
        import librosa
        buf = io.BytesIO(audio_bytes)
        try:
            y, sr = librosa.load(buf, sr=16000, mono=True)
        except Exception:
            raw  = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            y, sr= raw, 16000
        y    = y / (np.abs(y).max() + 1e-8)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        idx  = 0
        for stat in [mfcc.mean(1), mfcc.std(1),
                     librosa.feature.delta(mfcc).mean(1)]:
            n = min(len(stat), 300 - idx)
            feat[idx:idx+n] = stat[:n]; idx += n
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        if idx < 298:
            feat[idx]   = float(zcr.mean()); idx += 1
            feat[idx]   = float(rms.mean()); idx += 1
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm
    except Exception as e:
        print(f"[Features] librosa fallback error ({e})")
        return feat


def extract_audio_from_video(video_path: str) -> np.ndarray:
    """Extract audio track from a video file and compute audio features."""
    wav_path = video_path + "_audio.wav"
    ret = os.system(
        f'ffmpeg -y -i "{video_path}" -vn -ar 16000 -ac 1 "{wav_path}" -loglevel quiet 2>/dev/null'
    )
    if ret == 0 and os.path.exists(wav_path):
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        try: os.remove(wav_path)
        except Exception: pass
        return extract_audio_opensmile(audio_bytes)
    return np.zeros(300, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  VISUAL — DenseNet121 global-avg-pool → 342-d
# ═════════════════════════════════════════════════════════════════════════════
def _get_densenet():
    global _densenet_mdl, _densenet_tfm
    if _densenet_mdl is None:
        import torch
        import torchvision.models as tvm
        import torchvision.transforms as T
        from torchvision.models import DenseNet121_Weights
        print("[Features] Loading DenseNet121 …")
        mdl = tvm.densenet121(weights=DenseNet121_Weights.DEFAULT)
        mdl.classifier = torch.nn.Identity()   # keep 1024-d pool output
        mdl.eval()
        _densenet_mdl = mdl
        _densenet_tfm = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])
        print("[Features] DenseNet121 ready ✓")
    return _densenet_mdl, _densenet_tfm


def _image_bytes_to_array(image_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes → RGB uint8 ndarray."""
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2 could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _densenet_from_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Run DenseNet on an RGB ndarray → float32 (342,)."""
    import torch
    mdl, tfm = _get_densenet()
    tensor   = tfm(img_rgb).unsqueeze(0)          # (1,3,224,224)
    with torch.no_grad():
        pool = mdl(tensor).squeeze(0).numpy()     # (1024,)
    proj  = pool @ _PROJ_MAT                      # (342,)
    norm  = np.linalg.norm(proj) + 1e-8
    return (proj / norm).astype(np.float32)


def extract_visual_densenet(image_bytes: bytes) -> np.ndarray:
    """DenseNet121 pool → projected 342-d float32."""
    if not image_bytes or len(image_bytes) < 50:
        return np.zeros(342, dtype=np.float32)

    if DENSENET_AVAILABLE:
        try:
            img_rgb = _image_bytes_to_array(image_bytes)
            return _densenet_from_rgb(img_rgb)
        except Exception as e:
            print(f"[Features] DenseNet error ({e}) — pixel fallback")

    # ── pixel fallback ────────────────────────────────────────────────────────
    feat = np.zeros(342, dtype=np.float32)
    try:
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            img_s = cv2.resize(img, (32, 32)).astype(np.float32) / 255.0
            flat  = img_s.flatten()[:342]
            feat[:len(flat)] = flat
    except Exception:
        pass
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm


def extract_visual_from_video(video_path: str, n_frames: int = 8) -> np.ndarray:
    """
    Extract mean DenseNet features from n_frames evenly-sampled video frames.
    This is the method used in the paper for visual feature extraction.
    """
    if not DENSENET_AVAILABLE:
        return np.zeros(342, dtype=np.float32)
    try:
        import cv2
        cap    = cv2.VideoCapture(video_path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return np.zeros(342, dtype=np.float32)

        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        feats   = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feats.append(_densenet_from_rgb(img_rgb))
        cap.release()

        if not feats:
            return np.zeros(342, dtype=np.float32)
        mean_feat = np.mean(feats, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_feat) + 1e-8
        return mean_feat / norm
    except Exception as e:
        print(f"[Features] Video visual extraction error: {e}")
        return np.zeros(342, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Status report
# ═════════════════════════════════════════════════════════════════════════════
def print_feature_report():
    print("\n" + "─" * 54)
    print("  Feature Extractor Status")
    print("─" * 54)
    print(f"  Text   RoBERTa-Large   : {'✅ REAL  (1024-d [CLS] token)' if ROBERTA_AVAILABLE   else '⚠️  FALLBACK — pip install transformers'}")
    print(f"  Audio  openSMILE       : {'✅ REAL  (ComParE_2016 → 300-d)' if OPENSMILE_AVAILABLE else '⚠️  FALLBACK — pip install opensmile'}")
    print(f"  Visual DenseNet121     : {'✅ REAL  (pool → 342-d projected)' if DENSENET_AVAILABLE  else '⚠️  FALLBACK — pip install torchvision opencv-python'}")
    print("─" * 54 + "\n")


def get_extractor_status() -> dict:
    return {
        "text":   {"name": "RoBERTa-Large", "real": ROBERTA_AVAILABLE,
                   "dim": 1024, "install": "pip install transformers"},
        "audio":  {"name": "openSMILE ComParE_2016", "real": OPENSMILE_AVAILABLE,
                   "dim": 300,  "install": "pip install opensmile"},
        "visual": {"name": "DenseNet121",   "real": DENSENET_AVAILABLE,
                   "dim": 342,  "install": "pip install torchvision opencv-python"},
    }
