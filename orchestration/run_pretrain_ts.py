"""Wrapper fino: pretrain TST1. Delega en training.train_pretrain_ts."""
import time
from pathlib import Path

from training.train_pretrain_ts import run_pretrain_ts as _run


def run_pretrain_ts(config: dict, fold_idx: int = 0):
    """
    Fase 1: Pretrain TST1 (ROI-level masking).
    Guarda best_pt_ts_fold_{fold_idx}.pt en config['CHECKPOINTS_PATH'].
    """
    save_dir = Path(config["CHECKPOINTS_PATH"])

    print(f"\n{'='*60}")
    print(f"[orchestration] FASE 1: Pretrain TST1")
    print(f"  epochs   : {config['PT_TST1']['N_EPOCHS']}")
    print(f"  batch    : {config['PT_TST1']['BATCH_SIZE']}")
    print(f"  save_dir : {save_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    train_losses, val_losses = _run(config, fold_idx=fold_idx, save_dir=save_dir)
    elapsed = time.time() - t0

    ckpt = save_dir / f"best_pt_ts_fold_{fold_idx}.pt"
    print(f"[orchestration] Pretrain TST1 completado en {elapsed/60:.1f} min")
    print(f"  epochs ejecutados : {len(train_losses)}")
    print(f"  best val loss     : {min(val_losses):.4f}")
    print(f"  checkpoint        : {ckpt}")
    return train_losses, val_losses
