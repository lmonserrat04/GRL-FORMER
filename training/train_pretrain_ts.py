"""
Pretrain TST1 — ROI-level masking + reconstrucción.

Config óptima (paper Table 2):
    epochs=100, mask_ratio ∈ [0.25, 0.5] aleatorio, Adam lr=1e-4 wd=1e-4, bs=32.
"""

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data.augmentation.mask_utils import mask_roi_level
from training.context import ExperimentContext
from training.tasks.reconstruction import ReconstructionTask
from training.callbacks import EarlyStopping
from training.setup import build_experiment


def train_one_epoch(ctx: ExperimentContext, mask_ratio=None) -> float:
    model = ctx.model
    task: ReconstructionTask = ctx.task
    optimizer = ctx.optimizer
    train_loader = ctx.train_loader
    device = ctx.device

    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        masked_batch, mask, target, _ = mask_roi_level(batch, mask_ratio)

        optimizer.zero_grad()
        loss = task.execution_step(model, masked_batch, mask, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def validate(ctx: ExperimentContext, mask_ratio=None) -> float:
    model = ctx.model
    task: ReconstructionTask = ctx.task
    val_loader = ctx.val_loader
    device = ctx.device

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            masked_batch, mask, target, _ = mask_roi_level(batch, mask_ratio)
            loss = task.execution_step(model, masked_batch, mask, target)
            total_loss += loss.item()

    return total_loss


def run_pretrain_ts(config: dict, fold_idx: int = 0, save_dir: str | None = None):
    config["EXPERIMENT_TYPE"] = "pretrain_ts"
    exp = build_experiment(config, fold_idx=fold_idx)

    phase = config["PT_TST1"]
    epochs = phase["N_EPOCHS"]
    mask_ratio = phase.get("MASK_RATIO")

    es_config = {
        "PATIENCE":  phase.get("PATIENCE", 20),
        "MIN_DELTA": phase.get("MIN_DELTA", 1e-4),
    }
    early_stopping = EarlyStopping(exp.model, es_config)

    train_losses, val_losses = [], []

    with tqdm(range(1, epochs + 1), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Pretrain TST1 | Epoch {epoch}")

            train_loss = train_one_epoch(exp, mask_ratio)
            val_loss = validate(exp, mask_ratio)
            exp.scheduler.step()

            avg_train = train_loss / len(exp.train_loader)
            avg_val = val_loss / len(exp.val_loader)
            train_losses.append(avg_train)
            val_losses.append(avg_val)

            tepoch.set_postfix(train=f"{avg_train:.4f}", val=f"{avg_val:.4f}")

            if early_stopping(exp.model, avg_val):
                tqdm.write(f"  ⏹ Early stopping en epoch {epoch} "
                           f"(best val={early_stopping.min_val_loss:.4f})")
                break

    early_stopping.restore(exp.model)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"best_pt_ts_fold_{fold_idx}.pt"
        torch.save(exp.model.state_dict(), path)
        tqdm.write(f"  💾 {path}")

    return train_losses, val_losses
