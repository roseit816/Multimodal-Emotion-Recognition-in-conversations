"""
arc_analysis.py  —  Emotional Arc Analysis (Novelty Feature)
=============================================================
Analyses the SEQUENCE of emotion predictions across a conversation
to reveal patterns that per-utterance prediction alone cannot capture.

Three functions:
  detect_shifts      — every consecutive utterance where emotion changed
  classify_arc       — labels the overall conversation arc type
  build_arc_summary  — single call returning everything for the frontend
"""

from collections import Counter
from typing import List, Dict, Any

# ── Emotion intensity weights (Russell circumplex model of affect) ─────────────
INTENSITY = {
    "joy/excited": 0.85, "joy": 0.80,
    "sadness":     0.30, "anger":    0.90,
    "neutral":     0.10, "surprise": 0.75,
    "fear":        0.70, "disgust":  0.40,
}

def _intensity(emotion: str) -> float:
    return INTENSITY.get(emotion.lower(), 0.5)


def detect_shifts(predictions: List[Dict]) -> List[Dict]:
    """
    Return every utterance where the predicted emotion changed
    compared to the previous utterance.

    Each entry:
        idx          : utterance number (1-based)
        from_emotion : previous emotion
        to_emotion   : new emotion
        confidence   : confidence at this utterance (%)
        ts           : timestamp
        input_type   : audio_upload / video_upload / text / etc.
    """
    shifts = []
    for k in range(1, len(predictions)):
        prev = predictions[k - 1]
        curr = predictions[k]
        if curr["emotion"] != prev["emotion"]:
            shifts.append({
                "idx":          curr["idx"],
                "from_emotion": prev["emotion"],
                "to_emotion":   curr["emotion"],
                "confidence":   curr["confidence"],
                "ts":           curr.get("ts", ""),
                "input_type":   curr.get("input_type", ""),
            })
    return shifts


def classify_arc(predictions: List[Dict]) -> Dict[str, Any]:
    """
    Classify the emotional arc into one of four types:
      Stable        — one emotion dominates (>= 65% of utterances)
      Volatile      — emotion changes very often (>= 55% of consecutive pairs)
      Escalation    — average intensity rises in the second half
      De-escalation — average intensity falls in the second half
    """
    if not predictions:
        return {
            "arc_type": "Unknown", "arc_emoji": "❓", "arc_color": "#6b7280",
            "description": "Not enough predictions to classify the arc.",
            "dominant_emotion": "—", "intensity_trend": [], "shift_rate": 0.0,
        }

    emotions    = [p["emotion"]  for p in predictions]
    intensities = [_intensity(e) for e in emotions]
    n           = len(predictions)

    dominant = Counter(emotions).most_common(1)[0][0]
    dom_frac = Counter(emotions)[dominant] / n

    mid      = max(n // 2, 1)
    first_h  = sum(intensities[:mid])        / mid
    second_h = sum(intensities[mid:]) / max(n - mid, 1)
    delta    = second_h - first_h

    changes    = sum(1 for k in range(1, n) if emotions[k] != emotions[k - 1])
    shift_rate = changes / max(n - 1, 1)

    if dom_frac >= 0.65:
        arc_type  = "Stable"
        arc_emoji = "😐"
        arc_color = "#6b7280"
        desc = (f"The conversation was emotionally stable. "
                f"'{dominant}' appeared in {int(dom_frac*100)}% of utterances.")

    elif shift_rate >= 0.55:
        arc_type  = "Volatile"
        arc_emoji = "⚡"
        arc_color = "#d97706"
        desc = (f"Emotions shifted in {int(shift_rate*100)}% of consecutive utterances — "
                f"a highly volatile conversation. Most frequent: '{dominant}'.")

    elif delta >= 0.12:
        arc_type  = "Escalation"
        arc_emoji = "📈"
        arc_color = "#dc2626"
        desc = (f"Emotional intensity escalated across the conversation "
                f"(Δ intensity = {delta:+.2f}). Dominant emotion: '{dominant}'.")

    elif delta <= -0.12:
        arc_type  = "De-escalation"
        arc_emoji = "📉"
        arc_color = "#16a34a"
        desc = (f"The conversation became calmer toward the end "
                f"(Δ intensity = {delta:+.2f}). Dominant emotion: '{dominant}'.")

    else:
        arc_type  = "Stable"
        arc_emoji = "😐"
        arc_color = "#6b7280"
        desc = (f"No strong emotional trend detected. "
                f"Balanced conversation; most common emotion: '{dominant}'.")

    return {
        "arc_type":         arc_type,
        "arc_emoji":        arc_emoji,
        "arc_color":        arc_color,
        "description":      desc,
        "dominant_emotion": dominant,
        "intensity_trend":  [round(v, 3) for v in intensities],
        "shift_rate":       round(shift_rate, 3),
    }


def build_arc_summary(predictions: List[Dict]) -> Dict[str, Any]:
    """Single entry-point — returns arc + shifts + total."""
    return {
        "arc":    classify_arc(predictions),
        "shifts": detect_shifts(predictions),
        "total":  len(predictions),
    }
print(".")