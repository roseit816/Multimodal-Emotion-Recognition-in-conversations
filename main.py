"""Usage examples:
  # Train on IEMOCAP with real data
  python main.py --dataset iemocap --mode train

  # Train on MELD
  python main.py --dataset meld --mode train

  # Evaluate from checkpoint
  python main.py --dataset iemocap --mode eval --checkpoint experiments/checkpoints/best_model.pt

  # Dry-run with synthetic data (no real data needed)
  python main.py --dataset iemocap --mode dryrun
"""

import argparse
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from configs.config import Config
from training.trainer import run_training, set_seed
from utils.inference import evaluate, print_sample_predictions


def parse_args():
    p = argparse.ArgumentParser(description="SDT Multimodal ERC")
    p.add_argument("--dataset",    default="iemocap", choices=["iemocap", "meld"])
    p.add_argument("--mode",       default="train",   choices=["train", "eval", "dryrun"])
    p.add_argument("--checkpoint", default=None,      help="Path to .pt checkpoint for eval mode")
    p.add_argument("--epochs",     type=int,  default=None)
    p.add_argument("--lr",         type=float,default=None)
    p.add_argument("--batch_size", type=int,  default=None)
    p.add_argument("--hidden_dim", type=int,  default=None)
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--data_dir",   default="data")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Build config ─────────────────────────────────────────────────────────
    cfg = Config(dataset=args.dataset)
    cfg.train.seed   = args.seed
    cfg.train.device = args.device
    cfg.train.data_dir = args.data_dir

    if args.epochs:     cfg.train.num_epochs = args.epochs
    if args.lr:         cfg.train.lr         = args.lr
    if args.batch_size: cfg.train.batch_size = args.batch_size
    if args.hidden_dim: cfg.model.hidden_dim = args.hidden_dim

    set_seed(cfg.train.seed)

    # ── Dry run with synthetic data ───────────────────────────────────────────
    if args.mode == "dryrun":
        print("[DryRun] Generating synthetic data and running a training smoke-test…")
        from data.dataset import make_synthetic_dataset, get_dataloaders

        train_s = make_synthetic_dataset(30, num_classes=cfg.model.num_classes, seed=0)
        valid_s = make_synthetic_dataset(10, num_classes=cfg.model.num_classes, seed=1)
        test_s  = make_synthetic_dataset(10, num_classes=cfg.model.num_classes, seed=2)

        # Speed-up: use small model config for dry run
        cfg.model.hidden_dim     = 64
        cfg.model.num_heads      = 4
        cfg.model.ffn_dim        = 128
        cfg.model.num_intra_layers = 1
        cfg.model.num_inter_layers = 1
        cfg.train.num_epochs     = 3
        cfg.train.patience       = 99

        train_loader, valid_loader, test_loader = get_dataloaders(
            train_s, valid_s, test_s,
            batch_size=cfg.train.batch_size,
            max_seq_len=cfg.train.max_seq_len,
        )
        results = run_training(cfg, train_loader, valid_loader, test_loader)
        print(f"\n[DryRun] Smoke-test results: acc={results['accuracy']:.4f} wF1={results['weighted_f1']:.4f}")
        print("[DryRun] ✓ All modules ran successfully.")
        return

    # ── Load real data ────────────────────────────────────────────────────────
    from data.dataset import load_dataset, get_dataloaders

    if args.dataset == "iemocap":
        feature_file = os.path.join(cfg.train.data_dir, cfg.data.iemocap_feature_file)
    else:
        feature_file = os.path.join(cfg.train.data_dir, cfg.data.meld_feature_file)

    train_s, valid_s, test_s = load_dataset(
        feature_file=feature_file,
        dataset=args.dataset,
        text_dim=cfg.model.text_dim,
        audio_dim=cfg.model.audio_dim,
        visual_dim=cfg.model.visual_dim,
    )

    train_loader, valid_loader, test_loader = get_dataloaders(
        train_s, valid_s, test_s,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.train.max_seq_len,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.mode == "train":
        results = run_training(cfg, train_loader, valid_loader, test_loader)
        print(f"\nFinal Test Results:")
        print(f"  Accuracy    : {results['accuracy']:.4f}")
        print(f"  Weighted F1 : {results['weighted_f1']:.4f}")

    # ── Evaluate from checkpoint ──────────────────────────────────────────────
    elif args.mode == "eval":
        ckpt = args.checkpoint or os.path.join(cfg.train.checkpoint_dir, "best_model.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        results = evaluate(ckpt, cfg, test_s)
        print_sample_predictions(ckpt, cfg, test_s, n_conversations=3)


if __name__ == "__main__":
    print("Added feature")
    main()

