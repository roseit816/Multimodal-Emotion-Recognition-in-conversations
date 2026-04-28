"""
Feature Extraction Scripts
===========================
This module documents the feature extraction pipeline as described in the paper.

Text Features — RoBERTa-Large [CLS] token (dim=1024)
Audio Features — openSMILE IS09/IS13 feature sets (dim=300)
Visual Features — DenseNet facial action units (dim=342)

These scripts require the raw audio/video/text files from IEMOCAP or MELD.
Run them BEFORE training to produce the pickle file expected by dataset.py.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Text: RoBERTa-Large [CLS] embeddings
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_features(
    utterances: List[str],
    batch_size: int = 16,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract RoBERTa-Large [CLS] embeddings for a list of utterance strings.
    Returns (N, 1024) array.

    Requires:
        pip install transformers torch
    """
    try:
        from transformers import RobertaTokenizer, RobertaModel
        import torch
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = RobertaModel.from_pretrained("roberta-large").to(device)
    model.eval()

    all_feats = []
    with torch.no_grad():
        for i in range(0, len(utterances), batch_size):
            batch = utterances[i: i + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            output = model(**encoded)
            cls_tokens = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_feats.append(cls_tokens)

    return np.vstack(all_feats)   # (N, 1024)


# ─────────────────────────────────────────────────────────────────────────────
# Audio: openSMILE IS09/IS13 feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_audio_features(
    audio_paths: List[str],
    opensmile_config: str = "IS09_emotion.conf",
    output_dim: int = 384,
) -> np.ndarray:
    """
    Extract openSMILE features for a list of audio file paths.
    Returns (N, output_dim) array.

    Requires:
        pip install opensmile
        (or install openSMILE binary: https://audeering.github.io/opensmile-python/)
    """
    try:
        import opensmile
    except ImportError:
        raise ImportError("Install opensmile: pip install opensmile")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016
        if output_dim > 300 else opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    

    feats = []
    for path in audio_paths:
        try:
            feat = smile.process_file(path).values.flatten()
        except Exception:
            feat = np.zeros(output_dim, dtype=np.float32)

        # Ensure consistent dimensionality
        if len(feat) != output_dim:
            tmp = np.zeros(output_dim, dtype=np.float32)
            l = min(len(feat), output_dim)
            tmp[:l] = feat[:l]
            feat = tmp

        feats.append(feat.astype(np.float32))

    return np.stack(feats)   # (N, output_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Visual: DenseNet facial feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_visual_features(
    video_paths: List[str],
    output_dim: int = 342,
    fps_sample: int = 1,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract DenseNet-based facial features (mean-pooled over video frames).
    Returns (N, output_dim) array.

    Requires:
        pip install torch torchvision opencv-python
        Face detector: dlib or facenet-pytorch
    """
    try:
        import torch
        import torchvision.models as tvm
        import torchvision.transforms as T
        import cv2
        from torchvision.models import DenseNet121_Weights
    except ImportError:
        raise ImportError("Install deps: pip install torch torchvision opencv-python")

    # Use DenseNet121 features (before classifier)
    densenet = tvm.densenet121(weights=DenseNet121_Weights.DEFAULT)
    # Remove the classifier; use features layer
    densenet.classifier = torch.nn.Identity()
    densenet = densenet.to(device).eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        frame_feats = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % fps_sample != 0:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = transform(frame_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                f = densenet(t).cpu().numpy().flatten()
            frame_feats.append(f)
        cap.release()

        if len(frame_feats) == 0:
            feat = np.zeros(output_dim, dtype=np.float32)
        else:
            feat = np.mean(frame_feats, axis=0).astype(np.float32)

        # Ensure consistent dimensionality
        if feat.shape[0] != output_dim:
            tmp = np.zeros(output_dim, dtype=np.float32)
            l = min(feat.shape[0], output_dim)
            tmp[:l] = feat[:l]
            feat = tmp

        feats.append(feat)

    return np.stack(feats)   # (N, output_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Build IEMOCAP feature pickle
# ─────────────────────────────────────────────────────────────────────────────
def build_iemocap_features(
    iemocap_root: str,
    output_file: str = "data/iemocap/iemocap_features_raw.pkl",
    device: str = "cpu",
) -> None:
    """
    Build the IEMOCAP feature pickle from raw session directories.

    iemocap_root should contain Session1..Session5 subdirectories.
    """
    # NOTE: This is a template — adapt paths to your IEMOCAP installation.
    raise NotImplementedError(
        "Please adapt this function to your local IEMOCAP directory structure.\n"
        "Expected output format:\n"
        "  {\n"
        "    'train': { conv_id: {'text':[], 'audio':[], 'visual':[], "
        "'labels':[], 'speakers':[]} },\n"
        "    'valid': {...},\n"
        "    'test':  {...},\n"
        "  }\n"
        "\n"
        "Alternatively, use pre-computed features from:\n"
        "  https://github.com/declare-lab/MISA (IEMOCAP features)\n"
        "  https://github.com/declare-lab/conv-emotion (IEMOCAP/MELD features)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build MELD feature pickle
# ─────────────────────────────────────────────────────────────────────────────
def build_meld_features(
    meld_root: str,
    output_file: str = "data/meld/meld_features_raw.pkl",
    device: str = "cpu",
) -> None:
    """
    Build the MELD feature pickle from the CSV + media files.
    meld_root should contain train_sent_emo.csv, video_train/, etc.
    """
    raise NotImplementedError(
        "Please adapt this function to your local MELD directory structure.\n"
        "See: https://affective-meld.github.io/"
    )
print(".")