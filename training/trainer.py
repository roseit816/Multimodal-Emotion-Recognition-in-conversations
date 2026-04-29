"""
Training loop for SDT multimodal ERC model.

Implements:
  - End-to-end training with all loss components (task + distillation)
  - Proper conversation-level batching
  - Gradient clipping, early stopping
  - Checkpointing and logging
"""

import os
import time
import json
import random
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.sdt_model import SDTModel, build_model
from utils.metrics import compute_metrics
from configs.config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Logger setup
# ─────────────────────────────────────────────────────────────────────────────
def get_logger(log_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    ch = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(
    model:     SDTModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = total_task = total_student = total_kl = 0.0
    all_preds, all_labels = [], []
    n_batches = 0

    for batch in loader:
        text     = batch["text"].to(device)
        audio    = batch["audio"].to(device)
        visual   = batch["visual"].to(device)
        speakers = batch["speakers"].to(device)
        labels   = batch["labels"].to(device)
        mask     = batch["mask"].to(device)

        optimizer.zero_grad()

        out = model(text, audio, visual, speakers, mask, labels)

        loss = out["loss"]
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Accumulate metrics
        total_loss    += loss.item()
        total_task    += out["L_task"].item()
        total_student += out["L_student"].item()
        total_kl      += out["L_kl"].item()
        n_batches     += 1

        # Collect predictions for accuracy/F1
        with torch.no_grad():
            preds = out["logits"].argmax(dim=-1)          # (B, T)
            valid = mask.bool()
            all_preds.extend(preds[valid].cpu().tolist())
            all_labels.extend(labels[valid].cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics.update({
        "loss":       total_loss    / n_batches,
        "L_task":     total_task    / n_batches,
        "L_student":  total_student / n_batches,
        "L_kl":       total_kl      / n_batches,
    })
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# One evaluation epoch
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(
    model:  SDTModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n_batches = 0

    for batch in loader:
        text     = batch["text"].to(device)
        audio    = batch["audio"].to(device)
        visual   = batch["visual"].to(device)
        speakers = batch["speakers"].to(device)
        labels   = batch["labels"].to(device)
        mask     = batch["mask"].to(device)

        out = model(text, audio, visual, speakers, mask, labels)

        total_loss += out["loss"].item()
        n_batches  += 1

        preds = out["logits"].argmax(dim=-1)
        valid = mask.bool()
        all_preds.extend(preds[valid].cpu().tolist())
        all_labels.extend(labels[valid].cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / n_batches
    return metrics
    print("Added feature.")


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(
        self,
        cfg:        Config,
        model:      SDTModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader:  DataLoader,
        logger:     Optional[logging.Logger] = None,
    ):
        self.cfg     = cfg
        self.model   = model
        self.device  = torch.device(
            cfg.train.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader  = test_loader

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.train.num_epochs,
            eta_min=1e-6,
        )

        self.logger = logger or get_logger(cfg.train.log_dir)

        # Best model tracking
        self.best_wf1    = -1.0
        self.best_epoch  = 0
        self.best_ckpt   = os.path.join(cfg.train.checkpoint_dir, "best_model.pt")
        self.final_ckpt  = os.path.join(cfg.train.checkpoint_dir, "final_model.pt")
        self.history: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    def _log(self, epoch: int, split: str, metrics: Dict) -> None:
        acc  = metrics.get("accuracy", 0.0)
        wf1  = metrics.get("weighted_f1", 0.0)
        loss = metrics.get("loss", 0.0)
        self.logger.info(
            f"[Epoch {epoch:03d}] {split:>5s} | "
            f"loss={loss:.4f} | acc={acc:.4f} | wF1={wf1:.4f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def train(self) -> Dict:
        """Full training loop with early stopping."""
        self.logger.info("=" * 60)
        self.logger.info("Starting SDT Training")
        self.logger.info(f"  Device   : {self.device}")
        self.logger.info(f"  Dataset  : {self.cfg.data.dataset.upper()}")
        self.logger.info(f"  Epochs   : {self.cfg.train.num_epochs}")
        self.logger.info(f"  Patience : {self.cfg.train.patience}")
        self.logger.info("=" * 60)

        patience_counter = 0

        for epoch in range(1, self.cfg.train.num_epochs + 1):
            t0 = time.time()

            train_m = train_epoch(
                self.model, self.train_loader,
                self.optimizer, self.device, self.cfg.train.grad_clip,
            )
            valid_m = eval_epoch(self.model, self.valid_loader, self.device)

            self.scheduler.step()

            self._log(epoch, "TRAIN", train_m)
            self._log(epoch, "VALID", valid_m)

            self.history.append({
                "epoch": epoch, "train": train_m, "valid": valid_m,
            })

            # Save best checkpoint
            if valid_m["weighted_f1"] > self.best_wf1:
                self.best_wf1   = valid_m["weighted_f1"]
                self.best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "metrics":     valid_m,
                }, self.best_ckpt)
                self.logger.info(f"  ↑ New best wF1={self.best_wf1:.4f} — checkpoint saved")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.train.patience:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(best was epoch {self.best_epoch})"
                    )
                    break

            elapsed = time.time() - t0
            self.logger.info(f"  ↳ Epoch time: {elapsed:.1f}s\n")

        # Save final model
        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
        }, self.final_ckpt)

        # ── Test evaluation using BEST checkpoint ──────────────────────────
        self.logger.info(f"\nLoading best model from epoch {self.best_epoch} for test evaluation…")
        ckpt = torch.load(self.best_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        test_m = eval_epoch(self.model, self.test_loader, self.device)
        self._log(self.best_epoch, "TEST", test_m)

        # Save history
        hist_path = os.path.join(self.cfg.train.log_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        self.logger.info("Training complete.")
        self.logger.info(f"  Best epoch : {self.best_epoch}")
        self.logger.info(f"  Test Acc   : {test_m['accuracy']:.4f}")
        self.logger.info(f"  Test wF1   : {test_m['weighted_f1']:.4f}")

        return test_m


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_training(
    cfg:          Config,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader:  DataLoader,
) -> Dict:
    set_seed(cfg.train.seed)
    logger = get_logger(cfg.train.log_dir)
    model  = build_model(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"SDT model parameters: {n_params:,}")

    trainer = Trainer(cfg, model, train_loader, valid_loader, test_loader, logger)
    return trainer.train()
