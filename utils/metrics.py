"""
Evaluation metrics for Multimodal ERC.

Computes:
  - Accuracy (utterance-level)
  - Weighted F1-score
  - Per-class F1
  - Confusion matrix
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    labels: List[int],
    preds:  List[int],
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute accuracy and weighted F1-score at the utterance level.

    Args:
        labels:      Ground-truth class indices
        preds:       Predicted class indices
        label_names: Optional list of class names for the report

    Returns:
        dict with 'accuracy', 'weighted_f1', and per-class F1 scores
    """
    labels = np.array(labels)
    preds  = np.array(preds)

    # Filter out padding labels (value = -1)
    valid  = labels != -1
    labels = labels[valid]
    preds  = preds[valid]

    if len(labels) == 0:
        return {"accuracy": 0.0, "weighted_f1": 0.0}

    acc  = accuracy_score(labels, preds)
    wf1  = f1_score(labels, preds, average="weighted", zero_division=0)
    mf1  = f1_score(labels, preds, average="macro",    zero_division=0)

    classes = sorted(set(labels.tolist()))
    per_class_f1 = f1_score(labels, preds, labels=classes,
                             average=None, zero_division=0)

    result = {
        "accuracy":    float(acc),
        "weighted_f1": float(wf1),
        "macro_f1":    float(mf1),
    }

    for i, cls in enumerate(classes):
        name = label_names[cls] if label_names and cls < len(label_names) else f"class_{cls}"
        result[f"f1_{name}"] = float(per_class_f1[i])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Detailed evaluation report
# ─────────────────────────────────────────────────────────────────────────────
def full_evaluation_report(
    labels: List[int],
    preds:  List[int],
    label_names: Optional[List[str]] = None,
    dataset: str = "IEMOCAP",
) -> str:
    """
    Returns a formatted evaluation report string (classification_report + confusion matrix).
    """
    labels = np.array(labels)
    preds  = np.array(preds)

    valid  = labels != -1
    labels = labels[valid]
    preds  = preds[valid]

    target_names = label_names if label_names else None
    report = classification_report(
        labels, preds,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )
    cm = confusion_matrix(labels, preds)

    lines = [
        f"\n{'='*60}",
        f"  Evaluation Report — {dataset}",
        f"{'='*60}",
        report,
        "\nConfusion Matrix:",
        str(cm),
        f"\nAccuracy    : {accuracy_score(labels, preds):.4f}",
        f"Weighted F1 : {f1_score(labels, preds, average='weighted', zero_division=0):.4f}",
        f"{'='*60}\n",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific label name helpers
# ─────────────────────────────────────────────────────────────────────────────
IEMOCAP_LABEL_NAMES = ["joy/excited", "sadness", "anger/frustration",
                        "neutral", "surprise", "fear/disgust"]

MELD_LABEL_NAMES    = ["neutral", "surprise", "fear",
                        "sadness", "joy", "disgust", "anger"]


def get_label_names(dataset: str) -> List[str]:
    return IEMOCAP_LABEL_NAMES if dataset == "iemocap" else MELD_LABEL_NAMES

