"""
predict.py — Run predictions using a trained SDT model.

Three modes:
  1. quick   — train on synthetic data for 3 epochs, then predict (no real data needed)
  2. test    — load checkpoint, predict on synthetic test set, show metrics + per-utterance output
  3. single  — predict on one specific conversation and show emotion probabilities

Usage:
  python predict.py --mode quick
  python predict.py --mode test   --checkpoint experiments/checkpoints/best_model.pt
  python predict.py --mode single --checkpoint experiments/checkpoints/best_model.pt --conv_index 0
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from configs.config import Config
from models.sdt_model import build_model
from data.dataset import make_synthetic_dataset, get_dataloaders, MultimodalDataset, collate_fn
from training.trainer import set_seed, train_epoch, eval_epoch
from utils.metrics import compute_metrics, full_evaluation_report, get_label_names
from torch.utils.data import DataLoader
from torch.optim import AdamW


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = {
    "iemocap": ["joy/excited", "sadness", "anger", "neutral", "surprise", "fear"],
    "meld":    ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
}

BAR_CHARS = "█▉▊▋▌▍▎▏"

def prob_bar(p, width=20):
    """Render a simple ASCII probability bar."""
    filled = int(p * width)
    frac   = p * width - filled
    bar    = "█" * filled
    if frac > 0.5 and filled < width:
        bar += "▌"
    bar = bar.ljust(width)
    return bar


def print_header(title):
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)


def print_conversation_result(result, label_names, show_probs=True):
    """Pretty-print one conversation's predictions."""
    print(f"\n  Conversation ID : {result['conv_id']}")
    print(f"  Utterances      : {len(result['preds'])}")
    correct = sum(l == p for l, p in zip(result['labels'], result['preds']))
    print(f"  Correct         : {correct}/{len(result['preds'])}")
    print()

    for i, (true_id, pred_id, true_name, pred_name) in enumerate(zip(
        result['labels'], result['preds'],
        result['label_names'], result['pred_names']
    )):
        mark = "✓" if true_id == pred_id else "✗"
        print(f"  Utterance {i+1:>2}  [{mark}]  true={true_name:<16}  pred={pred_name:<16}")

        if show_probs and result.get('probs') is not None:
            probs = result['probs'][i]           # shape (num_classes,)
            top2  = np.argsort(probs)[::-1][:3]  # top-3 classes
            for cls_id in top2:
                p    = probs[cls_id]
                name = label_names[cls_id] if cls_id < len(label_names) else f"cls_{cls_id}"
                bar  = prob_bar(p, width=18)
                print(f"            {name:<16} {bar} {p*100:5.1f}%")
            print()


# ─────────────────────────────────────────────────────────────────────────────
# Quick train + predict (no checkpoint needed)
# ─────────────────────────────────────────────────────────────────────────────
def run_quick_mode(cfg, device):
    """Train for a few epochs on synthetic data, then predict immediately."""
    print_header("QUICK MODE — train on synthetic data, then predict")

    set_seed(cfg.train.seed)

    # Small model for speed
    cfg.model.hidden_dim       = 64
    cfg.model.num_heads        = 4
    cfg.model.ffn_dim          = 128
    cfg.model.num_intra_layers = 1
    cfg.model.num_inter_layers = 1
    cfg.train.num_epochs       = 5
    cfg.train.device           = device

    label_names = LABEL_NAMES[cfg.data.dataset]
    num_classes = len(label_names)
    cfg.model.num_classes = num_classes

    # Synthetic data
    print("\n  Generating synthetic conversations...")
    train_s = make_synthetic_dataset(40, num_classes=num_classes, seed=0)
    valid_s = make_synthetic_dataset(10, num_classes=num_classes, seed=1)
    test_s  = make_synthetic_dataset(10, num_classes=num_classes, seed=2)

    train_loader, valid_loader, test_loader = get_dataloaders(
        train_s, valid_s, test_s, batch_size=8
    )

    model = build_model(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)

    # Train
    print(f"\n  Training for {cfg.train.num_epochs} epochs...")
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Acc':<12} {'Val wF1'}")
    print(f"  {'-'*8} {'-'*14} {'-'*12} {'-'*10}")

    for epoch in range(1, cfg.train.num_epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, device)
        valid_m = eval_epoch(model, valid_loader, device)
        print(f"  {epoch:<8} {train_m['loss']:<14.4f} {valid_m['accuracy']:<12.4f} {valid_m['weighted_f1']:.4f}")

    # Predict on test set
    print("\n  Running predictions on test set...")
    run_prediction_display(model, test_s, cfg, device, label_names, n_convs=3)


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint + predict on test set
# ─────────────────────────────────────────────────────────────────────────────
def run_test_mode(cfg, device, checkpoint_path, n_convs=3):
    """Load a saved checkpoint and predict on synthetic test conversations."""
    print_header(f"TEST MODE — checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"\n  ERROR: Checkpoint not found at: {checkpoint_path}")
        print("  Run  python main.py --mode dryrun  first to generate a checkpoint.")
        sys.exit(1)

    label_names = LABEL_NAMES[cfg.data.dataset]
    num_classes = len(label_names)
    cfg.model.num_classes      = num_classes
    cfg.model.hidden_dim       = 64
    cfg.model.num_heads        = 4
    cfg.model.ffn_dim          = 128
    cfg.model.num_intra_layers = 1
    cfg.model.num_inter_layers = 1

    # Load model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"\n  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Generate test data
    test_s = make_synthetic_dataset(10, num_classes=num_classes, seed=99)

    # Overall metrics
    ds     = MultimodalDataset(test_s, max_seq_len=cfg.train.max_seq_len)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            text     = batch["text"].to(device)
            audio    = batch["audio"].to(device)
            visual   = batch["visual"].to(device)
            speakers = batch["speakers"].to(device)
            mask     = batch["mask"].to(device)
            labels   = batch["labels"]

            preds = model.predict(text, audio, visual, speakers, mask)
            valid = mask.bool()
            all_preds.extend(preds[valid].cpu().tolist())
            all_labels.extend(labels[valid].tolist())

    metrics = compute_metrics(all_labels, all_preds, label_names)
    print(f"\n  ── Overall Test Metrics ──────────────────")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1    : {metrics['macro_f1']:.4f}")

    # Per-conversation predictions
    run_prediction_display(model, test_s, cfg, device, label_names, n_convs=n_convs)


# ─────────────────────────────────────────────────────────────────────────────
# Single conversation deep-dive
# ─────────────────────────────────────────────────────────────────────────────
def run_single_mode(cfg, device, checkpoint_path, conv_index=0):
    """Deep-dive prediction for one conversation — shows full probability table."""
    print_header(f"SINGLE CONVERSATION MODE — conv index {conv_index}")

    if not os.path.exists(checkpoint_path):
        print(f"\n  ERROR: Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    label_names = LABEL_NAMES[cfg.data.dataset]
    num_classes = len(label_names)
    cfg.model.num_classes      = num_classes
    cfg.model.hidden_dim       = 64
    cfg.model.num_heads        = 4
    cfg.model.ffn_dim          = 128
    cfg.model.num_intra_layers = 1
    cfg.model.num_inter_layers = 1

    model = build_model(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_s  = make_synthetic_dataset(10, num_classes=num_classes, seed=99)
    sample  = test_s[min(conv_index, len(test_s)-1)]

    ds     = MultimodalDataset([sample], max_seq_len=cfg.train.max_seq_len)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in loader:
            text     = batch["text"].to(device)
            audio    = batch["audio"].to(device)
            visual   = batch["visual"].to(device)
            speakers = batch["speakers"].to(device)
            mask     = batch["mask"].to(device)

            out    = model(text, audio, visual, speakers, mask)
            logits = out["logits"]                              # (1, T, C)
            probs  = torch.softmax(logits, dim=-1).squeeze(0)  # (T, C)
            preds  = logits.argmax(dim=-1).squeeze(0)          # (T,)
            T      = int(batch["mask"][0].sum().item())

    preds_list  = preds[:T].cpu().tolist()
    probs_array = probs[:T].cpu().numpy()
    labels_list = sample.labels[:T].tolist()

    result = {
        "conv_id":     sample.conv_id,
        "labels":      labels_list,
        "preds":       preds_list,
        "label_names": [label_names[l] for l in labels_list],
        "pred_names":  [label_names[p] for p in preds_list],
        "probs":       probs_array,
    }

    print_conversation_result(result, label_names, show_probs=True)

    # Full probability table
    print(f"\n  ── Full Probability Table ────────────────────────────────")
    header = f"  {'Utt':>4}  " + "  ".join(f"{n[:8]:<8}" for n in label_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(T):
        row = f"  {i+1:>4}  "
        for c in range(num_classes):
            p = probs_array[i, c]
            marker = "*" if preds_list[i] == c else " "
            row += f"{p*100:6.1f}%{marker} "
        row += f"  → pred={label_names[preds_list[i]]}"
        print(row)
    print("  (* = predicted class)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Shared display helper
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction_display(model, test_samples, cfg, device, label_names, n_convs=3):
    """Run prediction on n_convs conversations and display results."""
    print(f"\n  ── Per-Conversation Predictions (showing {n_convs}) ──────────")

    for sample in test_samples[:n_convs]:
        ds     = MultimodalDataset([sample], max_seq_len=cfg.train.max_seq_len)
        loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

        with torch.no_grad():
            for batch in loader:
                text     = batch["text"].to(device)
                audio    = batch["audio"].to(device)
                visual   = batch["visual"].to(device)
                speakers = batch["speakers"].to(device)
                mask     = batch["mask"].to(device)

                out    = model(text, audio, visual, speakers, mask)
                logits = out["logits"]
                probs  = torch.softmax(logits, dim=-1).squeeze(0)
                preds  = logits.argmax(dim=-1).squeeze(0)
                T      = int(batch["mask"][0].sum().item())

        preds_list  = preds[:T].cpu().tolist()
        probs_array = probs[:T].cpu().numpy()
        labels_list = sample.labels[:T].tolist()

        result = {
            "conv_id":     sample.conv_id,
            "labels":      labels_list,
            "preds":       preds_list,
            "label_names": [label_names[l] for l in labels_list],
            "pred_names":  [label_names[p] for p in preds_list],
            "probs":       probs_array,
        }
        print_conversation_result(result, label_names, show_probs=True)

    print("═" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SDT Prediction Script")
    p.add_argument("--mode",       default="quick",
                   choices=["quick", "test", "single"],
                   help="quick=train+predict, test=load ckpt+predict, single=one conv deep-dive")
    p.add_argument("--checkpoint", default="experiments/checkpoints/best_model.pt",
                   help="Path to saved .pt checkpoint (used in test/single modes)")
    p.add_argument("--dataset",    default="iemocap", choices=["iemocap", "meld"])
    p.add_argument("--conv_index", type=int, default=0,
                   help="Which test conversation to inspect in single mode")
    p.add_argument("--device",     default="cpu",
                   help="cpu or cuda")
    p.add_argument("--n_convs",    type=int, default=3,
                   help="How many conversations to display in test mode")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = Config(dataset=args.dataset)
    device = args.device

    if args.mode == "quick":
        run_quick_mode(cfg, device)

    elif args.mode == "test":
        run_test_mode(cfg, device, args.checkpoint, n_convs=args.n_convs)

    elif args.mode == "single":
        run_single_mode(cfg, device, args.checkpoint, conv_index=args.conv_index)
