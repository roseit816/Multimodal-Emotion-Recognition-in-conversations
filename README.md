# SDT — Self-Distillation Transformer for Multimodal ERC

Implementation of the complete **SDT** model from:

> **"A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations"**  
> IEEE Transactions on Multimedia, 2024

---

## Project Structure

```
sdt_project/
├── configs/
│   └── config.py              # All hyperparameters (ModelConfig, TrainConfig, DataConfig)
├── data/
│   ├── dataset.py             # Dataset, DataLoader, preprocessing, normalization
│   └── feature_extraction.py  # RoBERTa / openSMILE / DenseNet extraction scripts
├── models/
│   └── sdt_model.py           # Complete SDT architecture (ALL equations)
├── training/
│   └── trainer.py             # End-to-end training loop with early stopping
├── utils/
│   ├── metrics.py             # Accuracy, Weighted F1, classification report
│   └── inference.py           # Checkpoint loading, batch/single-conversation prediction
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

---

## Architecture — Equations Implemented

| Component | File | Equations |
|---|---|---|
| Temporal Conv1D projection | `sdt_model.py::TemporalConvProjector` | Eq. 1 |
| Positional Embedding (sinusoidal) | `sdt_model.py::PositionalEmbedding` | Eq. 2 |
| Speaker Embedding (learned lookup) | `sdt_model.py::SpeakerEmbedding` | Eq. 3–4 |
| Intra-modal Transformer | `sdt_model.py::IntraModalTransformer` | Eq. 5 |
| Inter-modal Transformer (cross-attention) | `sdt_model.py::InterModalTransformer` | Eq. 6 |
| Unimodal-level gated fusion | `sdt_model.py::HierarchicalGatedFusion` | Eq. 7–9 |
| Multimodal-level gated fusion | `sdt_model.py::HierarchicalGatedFusion` | Eq. 10–11 |
| Emotion Classifier (FC + softmax) | `sdt_model.py::EmotionClassifier` | Eq. 12–13 |
| Student classifiers (one per modality) | `sdt_model.py::SelfDistillationModule` | Eq. 15–16 |
| Student cross-entropy loss | `sdt_model.py::SelfDistillationModule` | Eq. 17 |
| KL divergence loss (temperature-scaled) | `sdt_model.py::SelfDistillationModule` | Eq. 18 |
| Combined loss (γ1·L_task + γ2·L_s + γ3·L_KL) | `sdt_model.py::SelfDistillationModule` | Eq. 19–21 |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Dry-run with synthetic data (no real data required)
```bash
python main.py --mode dryrun
```
Runs the entire pipeline (encoding, fusion, distillation, loss, metrics) with
randomly generated conversations to verify all modules work end-to-end.

### 3. Prepare real data

#### Option A — Use pre-computed features
Place existing pickle files (from [MISA](https://github.com/declare-lab/MISA) or
[conv-emotion](https://github.com/declare-lab/conv-emotion)) at:
```
data/iemocap/iemocap_features_raw.pkl
data/meld/meld_features_raw.pkl
```

Expected pickle format:
```python
{
  'train': {
    'conv_id_1': {
      'text':     [[1024-dim], ...],   # RoBERTa [CLS] token per utterance
      'audio':    [[300-dim],  ...],   # openSMILE features
      'visual':   [[342-dim],  ...],   # DenseNet facial features
      'labels':   ['hap', 'neu', ...], # emotion labels (strings)
      'speakers': ['A', 'B', ...],     # speaker names
    },
    ...
  },
  'valid': { ... },
  'test':  { ... },
}
```

#### Option B — Extract from scratch
See `data/feature_extraction.py` for extraction scripts using:
- **Text**: RoBERTa-Large (HuggingFace `transformers`)
- **Audio**: openSMILE Python bindings
- **Visual**: DenseNet121 (torchvision) + OpenCV frame extraction

### 4. Train
```bash
# IEMOCAP (6 classes)
python main.py --dataset iemocap --mode train --device cuda

# MELD (7 classes)
python main.py --dataset meld --mode train --device cuda
```

### 5. Evaluate
```bash
python main.py --dataset iemocap --mode eval \
    --checkpoint experiments/checkpoints/best_model.pt
```

---

## Hyperparameters (paper defaults)

| Parameter | Value | Description |
|---|---|---|
| `hidden_dim` | 256 | Shared hidden size |
| `num_heads` | 8 | Attention heads |
| `num_intra_layers` | 2 | Intra-modal transformer depth |
| `num_inter_layers` | 2 | Inter-modal transformer depth |
| `ffn_dim` | 512 | FFN hidden size |
| `temperature T` | 4.0 | KL distillation temperature |
| `γ1` | 1.0 | Task loss weight |
| `γ2` | 1.0 | Student CE loss weight |
| `γ3` | 1.0 | KL loss weight |
| `lr` | 1e-4 | Learning rate (AdamW) |
| `dropout` | 0.1 | Dropout probability |

---

## Key Design Decisions

### Inter-modal Transformer (Eq. 6)
Each modality m attends to the **element-wise average** of the other two modalities
as context. This is a principled simplification — the paper states each modality
attends to "other modalities" without specifying how they are combined; averaging
avoids dimensionality blow-up while preserving cross-modal signal.

### Gated Fusion (Eq. 7–11)
- **Unimodal gate**: sigmoid over concatenated [R_t, R_a, R_v] → per-modality weight
- **Multimodal gate**: sigmoid over [R_t, R_a, R_v, U] → final weighted combination
- Simple concatenation is **NOT** used (as required by the paper)

### Self-Distillation
- The **teacher** classifier processes the fused multimodal representation M
- Three **student** classifiers process R_t, R_a, R_v independently
- At inference, predictions come **only** from the teacher (Eq. 12–13)
- KL divergence aligns student distributions toward the teacher using temperature T

---

## Expected Results (from the paper)

| Dataset | Accuracy | Weighted F1 |
|---|---|---|
| IEMOCAP | ~70.1% | ~70.4% |
| MELD | ~65.7% | ~65.5% |

---

## Citation

```bibtex
@article{sdt_tmm2024,
  title   = {A Transformer-Based Model With Self-Distillation for
             Multimodal Emotion Recognition in Conversations},
  journal = {IEEE Transactions on Multimedia},
  year    = {2024},
}
```
