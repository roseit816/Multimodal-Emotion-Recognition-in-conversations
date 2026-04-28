"""
Configuration for SDT (Self-Distillation Transformer) model.
Paper: "A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations"
IEEE TMM, 2024
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    # ── Modality feature dimensions (paper defaults) ──────────────────────────
    text_dim: int = 1024        # RoBERTa-Large [CLS] embedding
    audio_dim: int = 300        # openSMILE features
    visual_dim: int = 342       # DenseNet facial features

    # ── Shared hidden dimension ───────────────────────────────────────────────
    hidden_dim: int = 256

    # ── Temporal Conv1D (Eq. 1) ───────────────────────────────────────────────
    conv_kernel_size: int = 3

    # ── Transformer architecture ──────────────────────────────────────────────
    num_heads: int = 8
    num_intra_layers: int = 2   # Intra-modal transformer layers
    num_inter_layers: int = 2   # Inter-modal transformer layers
    dropout: float = 0.1
    ffn_dim: int = 512          # Feed-forward network hidden size

    # ── Self-Distillation ─────────────────────────────────────────────────────
    temperature: float = 4.0    # Temperature T for KL divergence (Eq. 18)
    gamma1: float = 1.0         # Weight for task loss (Eq. 19)
    gamma2: float = 1.0         # Weight for student CE losses (Eq. 20)
    gamma3: float = 1.0         # Weight for KL distillation loss (Eq. 21)

    # ── Classification ────────────────────────────────────────────────────────
    num_classes_iemocap: int = 6
    num_classes_meld: int = 7
    num_classes: int = 6        # default; overridden per dataset

    # ── Speaker embeddings ────────────────────────────────────────────────────
    max_speakers: int = 9       # max unique speakers across datasets


@dataclass
class TrainConfig:
    # ── General ───────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"        # "cuda" or "cpu"
    num_epochs: int = 60
    patience: int = 10          # early stopping patience

    # ── Optimiser ────────────────────────────────────────────────────────────
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # ── Data ─────────────────────────────────────────────────────────────────
    batch_size: int = 32        # utterances per batch
    max_seq_len: int = 200      # max utterances per conversation window

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: str = "data"
    checkpoint_dir: str = "experiments/checkpoints"
    log_dir: str = "experiments/logs"


@dataclass
class DataConfig:
    # ── Dataset selection ─────────────────────────────────────────────────────
    dataset: str = "iemocap"    # "iemocap" or "meld"

    # ── IEMOCAP ───────────────────────────────────────────────────────────────
    iemocap_emotions: List[str] = field(default_factory=lambda: [
        "neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger",
        "frustration", "excited", "other"
    ])
    iemocap_label_map: dict = field(default_factory=lambda: {
        "hap": 0, "exc": 0,   # joy/excited → 0
        "sad": 1,              # sadness → 1
        "ang": 2,              # anger → 2
        "fru": 2,              # frustration → anger → 2
        "neu": 3,              # neutral → 3
        "sur": 4,              # surprise → 4
        "fea": 5,              # fear → 5
        "dis": 5,              # disgust → 5
    })
    iemocap_final_labels: int = 6

    # ── MELD ──────────────────────────────────────────────────────────────────
    meld_label_map: dict = field(default_factory=lambda: {
        "neutral": 0, "surprise": 1, "fear": 2,
        "sadness": 3, "joy": 4, "disgust": 5, "anger": 6
    })
    meld_final_labels: int = 7

    # ── Feature files (expected paths under data_dir) ─────────────────────────
    iemocap_feature_file: str = "iemocap/iemocap_features_raw.pkl"
    meld_feature_file: str = "meld/meld_features_raw.pkl"


# ── Convenience accessor ──────────────────────────────────────────────────────
class Config:
    def __init__(self, dataset: str = "iemocap"):
        self.model = ModelConfig()
        self.train = TrainConfig()
        self.data = DataConfig(dataset=dataset)

        # Resolve number of classes from dataset
        if dataset == "iemocap":
            self.model.num_classes = self.data.iemocap_final_labels
        else:
            self.model.num_classes = self.data.meld_final_labels

        # Make sure output directories exist
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train.log_dir, exist_ok=True)
