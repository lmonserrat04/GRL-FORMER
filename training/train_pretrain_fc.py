"""
Pretrain TST2 — element-level masking + reconstrucción del vector PCC.

Config óptima (paper Table 2):
    epochs=100, mask_ratio=0.15 fijo, Adam lr=1e-4 wd=1e-4, bs=32.
"""

from pathlib import Path

import torch

from data.augmentation.mask_utils import mask_pcc_level
from training.context import ExperimentContext
from training.tasks.reconstruction import ReconstructionTask
from training.callbacks import EarlyStopping
from training.setup import build_experiment


# ──────────────────────────────────────────────────────────────────────
# Train / validate
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(ctx: ExperimentContext, mask_ratio: float = 0.15) -> float:
    """Un epoch de pretrain TST2. Devuelve la suma de losses."""
    model = ctx.model
    task: ReconstructionTask = ctx.task
    optimizer = ctx.optimizer
    train_loader = ctx.train_loader
    device = ctx.device

    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)

        # Máscara element-level sobre el vector PCC
        masked_batch, mask, target = mask_pcc_level(batch, mask_ratio)

        optimizer.zero_grad()
        loss = task.execution_step(model, masked_batch, mask, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def validate(ctx: ExperimentContext, mask_ratio: float = 0.15) -> float:
    """Validación. Devuelve la suma de losses sobre el val set."""
    model = ctx.model
    task: ReconstructionTask = ctx.task
    val_loader = ctx.val_loader
    device = ctx.device

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            masked_batch, mask, target = mask_pcc_level(batch, mask_ratio)
            loss = task.execution_step(model, masked_batch, mask, target)
            total_loss += loss.item()

    return total_loss


# ──────────────────────────────────────────────────────────────────────
# Loop completo
# ──────────────────────────────────────────────────────────────────────

def run_pretrain_fc(config: dict, fold_idx: int = 0, save_dir: str | None = None):
    """
    Entrena TST2 con early stopping sobre val loss.

    Returns:
        (train_losses, val_losses) — listas por epoch de las losses medias.
    """
    config["EXPERIMENT_TYPE"] = "pretrain_fc"
    exp = build_experiment(config, fold_idx=fold_idx)

    phase = config["PT_TST2"]
    epochs = phase["N_EPOCHS"]
    mask_ratio = phase.get("MASK_RATIO", 0.15)

    es_config = {
        "PATIENCE":  phase.get("PATIENCE", 20),
        "MIN_DELTA": phase.get("MIN_DELTA", 1e-4),
    }
    early_stopping = EarlyStopping(exp.model, es_config)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(exp, mask_ratio)
        val_loss = validate(exp, mask_ratio)
        exp.scheduler.step()

        avg_train = train_loss / len(exp.train_loader)
        avg_val = val_loss / len(exp.val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={avg_train:.4f} | val={avg_val:.4f}")

        if early_stopping(exp.model, avg_val):
            print(f"Early stopping en epoch {epoch} "
                  f"(best val={early_stopping.min_val_loss:.4f})")
            break

    early_stopping.restore(exp.model)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"best_pt_fc_fold_{fold_idx}.pt"
        torch.save(exp.model.state_dict(), path)
        print(f"Checkpoint guardado en {path}")

    return train_losses, val_losses


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import tempfile
    import shutil
    from pathlib import Path

    torch.manual_seed(0)

    # ─── Estructura sintética ─────────────────────────────────────────
    tmp = Path(tempfile.mkdtemp())
    interp_dir = tmp / "interp"
    interp_dir.mkdir()
    save_dir = tmp / "ckpts"

    N, T, R = 30, 100, 200
    D = R * (R - 1) // 2
    sites = ["SITE_0", "SITE_1", "SITE_2"]

    rng = np.random.default_rng(0)
    rows = []
    for i in range(N):
        fid = f"S{1000 + i}"
        arr = rng.standard_normal((T, R)).astype(np.float32)
        np.savetxt(interp_dir / f"interp_{fid}_rois_cc200.1D", arr)
        rows.append({
            "FILE_ID": fid, "SUB_ID": i,
            "SITE_ID": sites[i % 3],
            "DX_GROUP": int(rng.integers(0, 2)),
        })
    csv_path = tmp / "meta.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    config = {
        "DEVICE": "cpu", "NUM_WORKERS": 0, "SEED": 0,
        "RAW_PATH": str(interp_dir), "INTERP_PATH": str(interp_dir),
        "CSV_PATH": str(csv_path), "ATLAS": "cc200", "PREFIX": "interp_",
        "N_ROIS": R, "MAX_SEQ_LEN": T, "LABEL_COL": "DX_GROUP",
        "EVAL_PROTOCOL": "kfold", "N_FOLDS": 3,
        "TST1": {
            "D_MODEL": 64, "DIM_FEEDFORWARD": 128,
            "NUM_ENCODER_LAYERS": 2, "N_HEADS": 4,
            "ENC_DROP": 0.1, "USE_CLS_TOKEN": True,
        },
        "TST2": {
            "PCC_DIM": D, "D_MODEL": 64, "N_HEADS": 4,
            "NUM_ENCODER_LAYERS": 2, "DIM_FEEDFORWARD": 128, "ENC_DROP": 0.1,
        },
        "DUAL_STREAM": {
            "FUSION_TYPE": "attention_pooling", "NUM_CLASSES": 2,
            "CLASSIFIER_DROPOUT": 0.3, "MLP_DIMS": [256, 64, 2],
        },
        "FUSION": {"ATTENTION_POOLING": {"HIDDEN_DIM": 128}},
        "PT_TST1": {
            "N_EPOCHS": 3, "BATCH_SIZE": 4, "LR": 1e-4,
            "WEIGHT_DECAY": 1e-4, "MASK_RATIO": None,
            "OPTIMIZER": "Adam", "SCHEDULER": "CosineAnnealingLR",
            "SCHEDULER_PARAMS": {"T_max": 3, "eta_min": 1e-6},
        },
        "PT_TST2": {
            "N_EPOCHS": 3, "BATCH_SIZE": 4, "LR": 1e-4,
            "WEIGHT_DECAY": 1e-4, "MASK_RATIO": 0.15,
            "OPTIMIZER": "Adam", "SCHEDULER": "CosineAnnealingLR",
            "SCHEDULER_PARAMS": {"T_max": 3, "eta_min": 1e-6},
            "PATIENCE": 5, "MIN_DELTA": 1e-4,
        },
        "FINETUNING": {
            "N_EPOCHS": 2, "BATCH_SIZE": 4, "LR": 5e-5,
            "WEIGHT_DECAY": 1e-4, "OPTIMIZER": "Adam",
            "SCHEDULER": "CosineAnnealingLR",
            "SCHEDULER_PARAMS": {"T_max": 2, "eta_min": 5e-7},
        },
    }

    # ─── TEST 1: entrenamiento completo ───────────────────────────────
    print("── TEST 1: run_pretrain_fc (3 epochs, mask=0.15) ────────────")
    train_losses, val_losses = run_pretrain_fc(config, fold_idx=0, save_dir=save_dir)
    assert len(train_losses) == 3
    assert len(val_losses) == 3
    assert all(isinstance(x, float) and x >= 0 for x in train_losses)
    print(f"  ✓ train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}\n")

    # ─── TEST 2: checkpoint guardado ──────────────────────────────────
    print("── TEST 2: checkpoint guardado ──────────────────────────────")
    ckpt = save_dir / "best_pt_fc_fold_0.pt"
    assert ckpt.exists(), f"No existe {ckpt}"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and len(state) > 0
    print(f"  ✓ {ckpt.name} ({len(state)} tensores)\n")

    # ─── TEST 3: mask_ratio se lee del config ─────────────────────────
    print("── TEST 3: mask_ratio del config ────────────────────────────")
    cfg = {**config, "PT_TST2": {**config["PT_TST2"], "MASK_RATIO": 0.30, "N_EPOCHS": 1}}
    tl, vl = run_pretrain_fc(cfg, fold_idx=0, save_dir=None)
    assert len(tl) == 1
    print(f"  ✓ 1 epoch con mask=0.30 → train={tl[0]:.4f}\n")

    # ─── TEST 4: early stopping dispara ───────────────────────────────
    print("── TEST 4: early stopping con patience=1 ────────────────────")
    cfg_es = {
        **config,
        "PT_TST2": {**config["PT_TST2"], "N_EPOCHS": 10, "PATIENCE": 1,
                    "MIN_DELTA": 1e3},
    }
    tl_es, vl_es = run_pretrain_fc(cfg_es, fold_idx=0, save_dir=None)
    assert len(tl_es) < 10
    print(f"  ✓ paró en epoch {len(tl_es)}/10\n")

    shutil.rmtree(tmp)
    print("✅ Todos los tests de train_pretrain_fc.py pasaron.")