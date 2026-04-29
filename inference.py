"""
Inference module for SDT.
After training, predictions come ONLY from the trained model's
teacher EmotionClassifier — no heuristics or rule-based shortcuts.
"""
import os
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.sdt_model import SDTModel, build_model
from data.dataset import ConversationSample, MultimodalDataset, collate_fn
from utils.metrics import compute_metrics, full_evaluation_report, get_label_names
from configs.config import Config
# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(
    checkpoint_path: str,
    cfg: Config,
    device: Optional[str] = None,
) -> SDTModel:
    """Load a trained SDTModel from a checkpoint file."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"[Inference] Loaded model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model
# ─────────────────────────────────────────────────────────────────────────────
# Batch-level prediction (loader-based)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_loader(
    model:  SDTModel,
    loader: DataLoader,
    device: str = "cpu",
) -> Tuple[List[int], List[int]]:
    """
    Run model inference over a DataLoader.
    Returns (all_labels, all_preds) as flat lists of utterance-level integers.
    Predictions come ONLY from the model's teacher EmotionClassifier forward pass.
    """
    model.eval()
    all_labels, all_preds = [], []

    for batch in loader:
        text     = batch["text"].to(device)
        audio    = batch["audio"].to(device)
        visual   = batch["visual"].to(device)
        speakers = batch["speakers"].to(device)
        labels   = batch["labels"].to(device)
        mask     = batch["mask"].to(device)

        # ── INFERENCE: model.predict() uses ONLY the teacher classifier ──────
        preds = model.predict(text, audio, visual, speakers, mask)  # (B, T)

        valid = mask.bool()
        all_preds.extend(preds[valid].cpu().tolist())
        all_labels.extend(labels[valid].cpu().tolist())

    return all_labels, all_preds
# ─────────────────────────────────────────────────────────────────────────────
# Single-conversation prediction (per-conversation)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_conversation(
    model:   SDTModel,
    sample:  ConversationSample,
    cfg:     Config,
    device:  str = "cpu",
) -> Dict:
    """
    Run inference on a single conversation.

    Returns:
        {
          'conv_id':    str,
          'labels':     list of true label IDs,
          'preds':      list of predicted label IDs,
          'label_names':list of human-readable true labels,
          'pred_names': list of human-readable predicted labels,
          'probs':      (T, C) softmax probabilities,
        }
    """
    label_names = get_label_names(cfg.data.dataset)

    # Build single-sample loader
    ds     = MultimodalDataset([sample], max_seq_len=cfg.train.max_seq_len)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

    model.eval()
    for batch in loader:
        text     = batch["text"].to(device)
        audio    = batch["audio"].to(device)
        visual   = batch["visual"].to(device)
        speakers = batch["speakers"].to(device)
        mask     = batch["mask"].to(device)

        out   = model(text, audio, visual, speakers, mask)
        logits= out["logits"]          # (1, T, C)
        probs = torch.softmax(logits, dim=-1).squeeze(0)   # (T, C)
        preds = logits.argmax(dim=-1).squeeze(0)           # (T,)

        T = batch["mask"][0].sum().item()
        preds  = preds[:T].cpu().tolist()
        probs  = probs[:T].cpu().numpy()
        labels = sample.labels[:T].tolist()

    return {
        "conv_id":    sample.conv_id,
        "labels":     labels,
        "preds":      preds,
        "label_names": [label_names[l] for l in labels],
        "pred_names":  [label_names[p] for p in preds],
        "probs":       probs,
    } 
# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation runner
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    checkpoint_path: str,
    cfg:             Config,
    test_samples:    List[ConversationSample],
    save_report:     bool = True,
) -> Dict:
    """
    Load a trained checkpoint, run inference on test set, print & save report.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_checkpoint(checkpoint_path, cfg, device)
    label_names = get_label_names(cfg.data.dataset)

    ds     = MultimodalDataset(test_samples, max_seq_len=cfg.train.max_seq_len)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, collate_fn=collate_fn)

    all_labels, all_preds = predict_loader(model, loader, device)

    metrics = compute_metrics(all_labels, all_preds, label_names)
    report  = full_evaluation_report(all_labels, all_preds, label_names,
                                      dataset=cfg.data.dataset.upper())
    print(report)

    if save_report:
        report_path = os.path.join(cfg.train.log_dir, "test_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"[Inference] Report saved to {report_path}")
    return metrics
# ─────────────────────────────────────────────────────────────────────────────
# Sample prediction display
# ─────────────────────────────────────────────────────────────────────────────
def print_sample_predictions(
    checkpoint_path: str,
    cfg:             Config,
    test_samples:    List[ConversationSample],
    n_conversations: int = 3,
) -> None:
    """Print per-utterance predictions for a few test conversations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_checkpoint(checkpoint_path, cfg, device)
    label_names = get_label_names(cfg.data.dataset)
    print("\n" + "="*60)
    print("Added work by darshini")
    print(f"  Sample Predictions — {cfg.data.dataset.upper()}")
    print("="*60)
    for i, sample in enumerate(test_samples[:n_conversations]):
        result = predict_conversation(model, sample, cfg, device)
        print(f"\nConversation: {result['conv_id']}")
        print(f"  {'Utterance':>10}  {'True Label':>18}  {'Predicted':>18}")
        print(f"  {'-'*10}  {'-'*18}  {'-'*18}")
        for j, (lbl, pred) in enumerate(zip(result["label_names"],
                                             result["pred_names"])):
            mark = "✓" if lbl == pred else "✗"
            print(f"  {j+1:>10}  {lbl:>18}  {pred:>18}  {mark}")

    print("="*60 + "\n")
