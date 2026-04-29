"""
Data preprocessing for IEMOCAP and MELD datasets.

Expected raw feature format (pickle):
  {
    'train': { conv_id: { 'text': [...], 'audio': [...], 'visual': [...],
                          'labels': [...], 'speakers': [...] } },
    'valid': { ... },
    'test':  { ... }
  }

Text features  : RoBERTa-Large [CLS] token (dim=1024)
Audio features : openSMILE IS09/IS13 feature sets (dim=300)
Visual features: DenseNet facial action units  (dim=342)
"""

import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────────────────────────────────────────
# Utterance-level sample
# ─────────────────────────────────────────────────────────────────────────────
class ConversationSample:
    """Holds one full conversation with aligned multimodal features."""

    def __init__(
        self,
        conv_id: str,
        text_features: np.ndarray,    # (T, D_t)
        audio_features: np.ndarray,   # (T, D_a)
        visual_features: np.ndarray,  # (T, D_v)
        labels: np.ndarray,           # (T,)
        speakers: np.ndarray,         # (T,)  integer speaker IDs
        speaker_to_id: Dict[str, int],
    ):
        self.conv_id = conv_id
        self.text = text_features
        self.audio = audio_features
        self.visual = visual_features
        self.labels = labels
        self.speakers = speakers
        self.speaker_to_id = speaker_to_id
        self.length = len(labels)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
class MultimodalDataset(Dataset):
    """Dataset that returns one conversation per item."""

    def __init__(
        self,
        samples: List[ConversationSample],
        max_seq_len: int = 200,
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        T = min(s.length, self.max_seq_len)

        # Truncate to max_seq_len
        text    = torch.tensor(s.text[:T],    dtype=torch.float32)   # (T, D_t)
        audio   = torch.tensor(s.audio[:T],   dtype=torch.float32)   # (T, D_a)
        visual  = torch.tensor(s.visual[:T],  dtype=torch.float32)   # (T, D_v)
        labels  = torch.tensor(s.labels[:T],  dtype=torch.long)      # (T,)
        speakers= torch.tensor(s.speakers[:T],dtype=torch.long)      # (T,)

        # Mask for valid (non-padded) positions
        mask = torch.ones(T, dtype=torch.bool)                        # (T,)

        return {
            "text":     text,
            "audio":    audio,
            "visual":   visual,
            "labels":   labels,
            "speakers": speakers,
            "mask":     mask,
            "length":   torch.tensor(T, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate function — pads conversations to the same length within a batch
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad conversations to the same length and stack into tensors."""
    keys = ["text", "audio", "visual", "labels", "speakers", "mask"]
    out = {}

    # Pad sequence tensors
    for key in keys:
        seqs = [item[key] for item in batch]
        if key == "mask":
            # pad mask with False
            out[key] = pad_sequence(seqs, batch_first=True, padding_value=0).bool()
        elif key == "labels" or key == "speakers":
            out[key] = pad_sequence(seqs, batch_first=True, padding_value=-1)
        else:
            out[key] = pad_sequence(seqs, batch_first=True, padding_value=0.0)

    out["lengths"] = torch.stack([item["length"] for item in batch])
    return out
    print("Added feature.")
    print(".")

# ─────────────────────────────────────────────────────────────────────────────
# Feature normalization helper
# ─────────────────────────────────────────────────────────────────────────────
def normalize_features(
    train_samples: List[ConversationSample],
    valid_samples: List[ConversationSample],
    test_samples:  List[ConversationSample],
    modality: str = "audio",
) -> None:
    """Z-score normalise audio and visual features using training statistics."""

    # Gather all training utterances for the modality
    all_feats = np.vstack([getattr(s, modality) for s in train_samples])
    mean = all_feats.mean(axis=0, keepdims=True)
    std  = all_feats.std(axis=0, keepdims=True) + 1e-8

    for split in [train_samples, valid_samples, test_samples]:
        for s in split:
            setattr(s, modality, (getattr(s, modality) - mean) / std)


# ─────────────────────────────────────────────────────────────────────────────
# IEMOCAP loader
# ─────────────────────────────────────────────────────────────────────────────
IEMOCAP_LABEL_MAP = {
    "hap": 0, "exc": 0,
    "sad": 1,
    "ang": 2, "fru": 2,
    "neu": 3,
    "sur": 4,
    "fea": 5, "dis": 5,
}

MELD_LABEL_MAP = {
    "neutral": 0, "surprise": 1, "fear": 2,
    "sadness": 3, "joy": 4, "disgust": 5, "anger": 6,
}


def _build_speaker_id_map(raw_data: dict) -> Dict[str, int]:
    """Assign integer IDs to all unique speaker names across splits."""
    names = set()
    for split_data in raw_data.values():
        for conv in split_data.values():
            names.update(conv.get("speakers", []))
    return {name: i for i, name in enumerate(sorted(names))}


def _load_split(
    split_data: dict,
    label_map: Dict[str, int],
    speaker_to_id: Dict[str, int],
    text_dim: int,
    audio_dim: int,
    visual_dim: int,
) -> List[ConversationSample]:
    """Convert a raw split dict to a list of ConversationSample objects."""
    samples = []

    for conv_id, conv in split_data.items():
        raw_labels   = conv.get("labels", [])
        raw_speakers = conv.get("speakers", [])
        text_feats   = conv.get("text",   [])
        audio_feats  = conv.get("audio",  [])
        visual_feats = conv.get("visual", [])

        # Filter utterances with known labels
        valid_idx = []
        for i, lbl in enumerate(raw_labels):
            mapped = label_map.get(str(lbl).lower(), None)
            if mapped is not None:
                valid_idx.append(i)

        if len(valid_idx) == 0:
            continue

        def _safe_array(lst, idx_list, dim):
            """Return (T, dim) array, zero-filling missing/None rows."""
            arr = []
            for i in idx_list:
                row = lst[i] if i < len(lst) else None
                if row is None or len(row) == 0:
                    arr.append(np.zeros(dim, dtype=np.float32))
                else:
                    row = np.array(row, dtype=np.float32)
                    if row.shape[0] != dim:
                        # Truncate or pad feature dimension
                        tmp = np.zeros(dim, dtype=np.float32)
                        l = min(row.shape[0], dim)
                        tmp[:l] = row[:l]
                        row = tmp
                    arr.append(row)
            return np.stack(arr, axis=0)

        t_arr = _safe_array(text_feats,   valid_idx, text_dim)
        a_arr = _safe_array(audio_feats,  valid_idx, audio_dim)
        v_arr = _safe_array(visual_feats, valid_idx, visual_dim)

        lbls  = np.array([label_map[str(raw_labels[i]).lower()] for i in valid_idx],
                         dtype=np.int64)
        spks  = np.array([speaker_to_id.get(str(raw_speakers[i]) if i < len(raw_speakers) else "UNK", 0)
                          for i in valid_idx], dtype=np.int64)

        samples.append(ConversationSample(
            conv_id=conv_id,
            text_features=t_arr,
            audio_features=a_arr,
            visual_features=v_arr,
            labels=lbls,
            speakers=spks,
            speaker_to_id=speaker_to_id,
        ))

    return samples


def load_dataset(
    feature_file: str,
    dataset: str = "iemocap",
    text_dim:   int = 1024,
    audio_dim:  int = 300,
    visual_dim: int = 342,
    normalize:  bool = True,
) -> Tuple[List[ConversationSample], List[ConversationSample], List[ConversationSample]]:
    """
    Load and preprocess a multimodal ERC dataset from a pickle file.

    Returns train, valid, test splits as lists of ConversationSample.
    """
    if not os.path.exists(feature_file):
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}\n"
            "Please extract features using the scripts in data/feature_extraction/ "
            "or place pre-computed features at the expected path."
        )

    with open(feature_file, "rb") as f:
        raw = pickle.load(f)

    label_map = IEMOCAP_LABEL_MAP if dataset == "iemocap" else MELD_LABEL_MAP
    speaker_to_id = _build_speaker_id_map(raw)

    kwargs = dict(
        label_map=label_map,
        speaker_to_id=speaker_to_id,
        text_dim=text_dim,
        audio_dim=audio_dim,
        visual_dim=visual_dim,
    )
    train_s = _load_split(raw["train"], **kwargs)
    valid_s = _load_split(raw.get("valid", raw.get("dev", {})), **kwargs)
    test_s  = _load_split(raw["test"],  **kwargs)

    print(f"[DataLoader] {dataset.upper()} — "
          f"train={len(train_s)} | valid={len(valid_s)} | test={len(test_s)} conversations")

    if normalize:
        for modality in ["audio", "visual"]:
            normalize_features(train_s, valid_s, test_s, modality=modality)

    return train_s, valid_s, test_s


def get_dataloaders(
    train_samples: List[ConversationSample],
    valid_samples: List[ConversationSample],
    test_samples:  List[ConversationSample],
    batch_size: int = 32,
    max_seq_len: int = 200,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap splits in DataLoader objects."""

    train_ds = MultimodalDataset(train_samples, max_seq_len)
    valid_ds = MultimodalDataset(valid_samples, max_seq_len)
    test_ds  = MultimodalDataset(test_samples,  max_seq_len)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(valid_ds, shuffle=False),
        make_loader(test_ds,  shuffle=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (for unit-testing without real data)
# ─────────────────────────────────────────────────────────────────────────────
def make_synthetic_dataset(
    n_conversations: int = 20,
    utterances_per_conv: Tuple[int, int] = (5, 20),
    num_classes: int = 6,
    text_dim: int = 1024,
    audio_dim: int = 300,
    visual_dim: int = 342,
    seed: int = 42,
) -> List[ConversationSample]:
    """Generate fake conversations for debugging the pipeline."""
    rng = np.random.RandomState(seed)
    speaker_to_id = {"A": 0, "B": 1}
    samples = []

    for i in range(n_conversations):
        T = rng.randint(*utterances_per_conv)
        samples.append(ConversationSample(
            conv_id=f"syn_{i:04d}",
            text_features=rng.randn(T, text_dim).astype(np.float32),
            audio_features=rng.randn(T, audio_dim).astype(np.float32),
            visual_features=rng.randn(T, visual_dim).astype(np.float32),
            labels=rng.randint(0, num_classes, size=T).astype(np.int64),
            speakers=rng.randint(0, 2, size=T).astype(np.int64),
            speaker_to_id=speaker_to_id,
        ))

    return samples
