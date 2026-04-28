"""
run_webapp.py
=============
One-command launcher for EmotiSense.

  python run_webapp.py

What it does
------------
1. Trains SDT on synthetic data with REAL feature dims (1024/300/342)
   — only if no checkpoint exists yet.
2. Starts the Flask server on port 5000.
3. Opens your browser automatically.

To use real IEMOCAP / MELD data instead of synthetic data, train first:
  python main.py --dataset iemocap --mode train
Then run_webapp.py will find the checkpoint and skip training.
"""

import os, sys, threading, time, webbrowser

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_WEBAPP_DIR   = os.path.join(_PROJECT_ROOT, "webapp")
os.chdir(_PROJECT_ROOT)
for _p in [_PROJECT_ROOT, _WEBAPP_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHECKPOINT  = os.path.join("experiments", "checkpoints", "best_model.pt")
PORT        = 5000
# These dims MUST match real_features.py exactly
TEXT_DIM    = 1024
AUDIO_DIM   = 300
VISUAL_DIM  = 342
NUM_CLASSES = 6     # IEMOCAP


def train_quick():
    """Train a lightweight SDT on synthetic data with correct feature dims."""
    import torch
    from configs.config import Config
    from models.sdt_model import build_model
    from data.dataset import make_synthetic_dataset, get_dataloaders
    from training.trainer import train_epoch, eval_epoch, set_seed
    from torch.optim import AdamW

    set_seed(42)
    cfg = Config(dataset="iemocap")
    cfg.model.hidden_dim       = 64
    cfg.model.num_heads        = 4
    cfg.model.ffn_dim          = 128
    cfg.model.num_intra_layers = 1
    cfg.model.num_inter_layers = 1
    cfg.model.num_classes      = NUM_CLASSES
    cfg.model.text_dim         = TEXT_DIM
    cfg.model.audio_dim        = AUDIO_DIM
    cfg.model.visual_dim       = VISUAL_DIM

    print(f"[Train] Feature dims: text={TEXT_DIM} audio={AUDIO_DIM} visual={VISUAL_DIM}")
    train_s = make_synthetic_dataset(80,  num_classes=NUM_CLASSES, seed=0,
                                     text_dim=TEXT_DIM, audio_dim=AUDIO_DIM, visual_dim=VISUAL_DIM)
    valid_s = make_synthetic_dataset(20,  num_classes=NUM_CLASSES, seed=1,
                                     text_dim=TEXT_DIM, audio_dim=AUDIO_DIM, visual_dim=VISUAL_DIM)
    train_l, valid_l, _ = get_dataloaders(train_s, valid_s, valid_s, batch_size=8)

    model = build_model(cfg)
    opt   = AdamW(model.parameters(), lr=5e-4)

    best_wf1 = -1.0
    print(f"\n  {'Ep':<4} {'Loss':<10} {'Acc':<10} {'wF1'}")
    print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*8}")

    for ep in range(1, 12):
        tm = train_epoch(model, train_l, opt, "cpu")
        vm = eval_epoch(model, valid_l, "cpu")
        print(f"  {ep:<4} {tm['loss']:<10.4f} {vm['accuracy']:<10.4f} {vm['weighted_f1']:.4f}")
        if vm["weighted_f1"] >= best_wf1:
            best_wf1 = vm["weighted_f1"]
            os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
            torch.save({"epoch": ep, "model_state": model.state_dict()}, CHECKPOINT)

    print(f"\n[Train] Best wF1={best_wf1:.4f}  saved → {CHECKPOINT}")
    print("[Train] NOTE: trained on SYNTHETIC data.")
    print("[Train] For paper-level accuracy, train on IEMOCAP:")
    print("[Train]   python main.py --dataset iemocap --mode train\n")


def open_browser():
    time.sleep(2.8)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    print("=" * 60)
    print("  EmotiSense — SDT Multimodal Emotion Recognition")
    print("  Paper: IEEE TMM 2024  |  Equations: 1–21")
    print("=" * 60 + "\n")
    print("feature added")

    if os.path.exists(CHECKPOINT):
        print(f"[Launcher] Found checkpoint: {CHECKPOINT}\n")
    else:
        print("[Launcher] No checkpoint — running quick training (~60s on CPU)…\n")
        train_quick()

    threading.Thread(target=open_browser, daemon=True).start()
    print(f"[Launcher] Server → http://localhost:{PORT}")
    print("[Launcher] Ctrl+C to stop\n")

    from webapp.app import app, load_model
    load_model(CHECKPOINT)
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
