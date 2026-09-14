"""
Fase contrastive — alinea las representaciones de TST1 y TST2 con InfoNCE.

Config óptima (paper Sec. 3.3 + Sec. 4.3.2 + Table 2):
    epochs=50, InfoNCE τ=0.07, proj 256→128,
    unfreeze BOTH encoders (Sec. 4.3.2), Adam lr=1e-4 wd=1e-4, bs=32.
"""

from pathlib import Path

import torch

from training.context import ExperimentContext
from training.tasks.contrastive import ContrastiveTask
from training.callbacks import EarlyStopping
from training.setup import build_experiment


# ──────────────────────────────────────────────────────────────────────
# Train / validate
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(ctx: ExperimentContext) -> float:
    """Un epoch de contraste. Devuelve la suma de losses + align acumulado."""
    model = ctx.model
    task: ContrastiveTask = ctx.task
    optimizer = ctx.optimizer
    train_loader = ctx.train_loader
    device = ctx.device

    model.train()
    task.contrastive_module.train()

    total_loss = 0.0
    total_align = 0.0

    for batch in train_loader:
        ts = batch["timeseries"].to(device)
        pcc = batch["pcc_vector"].to(device)

        optimizer.zero_grad()
        h_ts, h_fc = model.get_features(ts, pcc)
        loss, _, align = task.contrastive_module(h_ts, h_fc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_align += align.item()

    return total_loss, total_align


def validate(ctx: ExperimentContext) -> tuple[float, float]:
    """Val loss + align, sin grad."""
    model = ctx.model
    task: ContrastiveTask = ctx.task
    val_loader = ctx.val_loader
    device = ctx.device

    model.eval()
    task.contrastive_module.eval()

    total_loss = 0.0
    total_align = 0.0

    with torch.no_grad():
        for batch in val_loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            h_ts, h_fc = model.get_features(ts, pcc)
            loss, _, align = task.contrastive_module(h_ts, h_fc)
            total_loss += loss.item()
            total_align += align.item()

    return total_loss, total_align


# ──────────────────────────────────────────────────────────────────────
# Loop completo
# ──────────────────────────────────────────────────────────────────────

def run_contrastive(config: dict, fold_idx: int = 0, save_dir: str | None = None):
    """
    Entrena la fase contrastive con early stopping sobre val loss.

    Returns:
        (train_losses, val_losses, train_aligns, val_aligns) — listas por epoch.
    """
    config["EXPERIMENT_TYPE"] = "contrastive"
    exp = build_experiment(config, fold_idx=fold_idx)

    phase = config["T_CONTRASTIVE"]
    epochs = phase["N_EPOCHS"]

    es_config = {
        "PATIENCE":  phase.get("PATIENCE", 20),
        "MIN_DELTA": phase.get("MIN_DELTA", 1e-4),
    }
    early_stopping = EarlyStopping(exp.model, es_config)

    train_losses, val_losses = [], []
    train_aligns, val_aligns = [], []

    for epoch in range(1, epochs + 1):
        t_loss, t_align = train_one_epoch(exp)
        v_loss, v_align = validate(exp)
        exp.scheduler.step()

        n_train = len(exp.train_loader)
        n_val = len(exp.val_loader)
        avg_tl, avg_vl = t_loss / n_train, v_loss / n_val
        avg_ta, avg_va = t_align / n_train, v_align / n_val

        train_losses.append(avg_tl); val_losses.append(avg_vl)
        train_aligns.append(avg_ta); val_aligns.append(avg_va)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={avg_tl:.4f} align={avg_ta:.4f} | "
              f"val={avg_vl:.4f} align={avg_va:.4f}")

        if early_stopping(exp.model, avg_vl):
            print(f"Early stopping en epoch {epoch} "
                  f"(best val={early_stopping.min_val_loss:.4f})")
            break

    early_stopping.restore(exp.model)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"best_contrastive_fold_{fold_idx}.pt"
        torch.save(exp.model.state_dict(), path)
        print(f"Checkpoint guardado en {path}")

    return train_losses, val_losses, train_aligns, val_aligns

def run_contrastive_global(config: dict, save_dir: str | None = None):
    """
    Fase contrastive GLOBAL (paper Sec. 3.3).
    Corre UNA sola vez sobre el split 70/10/20, ANTES de folds.
    Guarda checkpoint con proj heads para que finetune lo cargue.

    Estrategia de freezing (repo OPTIMAL_CONFIGURATION.md):
        - freeze TST1
        - unfreeze TST2
        - projection heads siempre entrenables
    """
    from pathlib import Path
    from tqdm import tqdm
    from models.transformer_ts import create_transformer_ts
    from models.transformer_fc import create_transformer_fc
    from training.tasks.contrastive import ContrastiveWrapper
    from data.loaders.dataloader import get_single_split_loaders

    device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    phase = config["T_CONTRASTIVE"]

    # ─── Encoders ─────────────────────────────────────────────────────
    tst1_cfg = {
        "n_rois":          config["N_ROIS"],
        "emb_dim":         config["TST1"]["D_MODEL"],
        "n_layers":        config["TST1"]["NUM_ENCODER_LAYERS"],
        "n_heads":         config["TST1"]["N_HEADS"],
        "dim_feedforward": config["TST1"]["DIM_FEEDFORWARD"],
        "dropout":         config["TST1"]["ENC_DROP"],
        "max_seq_len":     config["MAX_SEQ_LEN"],
        "use_cls_token":   config["TST1"]["USE_CLS_TOKEN"],
    }
    tst2_cfg = {
        "pcc_dim":         config["TST2"]["PCC_DIM"],
        "d_model":         config["TST2"]["D_MODEL"],
        "n_layers":        config["TST2"]["NUM_ENCODER_LAYERS"],
        "n_heads":         config["TST2"]["N_HEADS"],
        "dim_feedforward": config["TST2"]["DIM_FEEDFORWARD"],
        "dropout":         config["TST2"]["ENC_DROP"],
    }

    tst1 = create_transformer_ts(tst1_cfg).to(device)
    tst2 = create_transformer_fc(tst2_cfg).to(device)

    if config.get("CKPT_TST1"):
        tst1.load_pretrained(config["CKPT_TST1"], strict=False)
    if config.get("CKPT_TST2"):
        tst2.load_pretrained(config["CKPT_TST2"], strict=False)

    # ─── Projection heads ────────────────────────────────────────────
    contrastive_module = ContrastiveWrapper(
        dim_ts=tst1.emb_dim,
        dim_fc=tst2.d_model,
        hidden_dim=phase["PROJ_HIDDEN_DIM"],
        output_dim=phase["PROJ_OUTPUT_DIM"],
        temperature=phase["TEMPERATURE"],
    ).to(device)

    # ─── Freezing: TST1 congelado, TST2 entrenable ───────────────────
    for p in tst1.parameters():
        p.requires_grad = False
    for p in tst2.parameters():
        p.requires_grad = True

    trainable = [p for p in tst2.parameters() if p.requires_grad] + \
                list(contrastive_module.parameters())

    optimizer = torch.optim.Adam(
        trainable,
        lr=float(phase["LR"]),
        weight_decay=float(phase["WEIGHT_DECAY"]),
    )

    # ─── Loaders (split global 70/10/20) ─────────────────────────────
    train_loader, _, _, _ = get_single_split_loaders(
        config,
        batch_size=phase["BATCH_SIZE"],
        num_workers=config.get("NUM_WORKERS", 0),
        seed=config.get("SEED", 42),
    )

    # ─── Loop ────────────────────────────────────────────────────────
    epochs = phase["N_EPOCHS"]

    with tqdm(range(1, epochs + 1), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Contrastive GLOBAL | Epoch {epoch}")

            tst1.eval()   # frozen
            tst2.train()
            contrastive_module.train()

            total_loss, total_align = 0.0, 0.0
            for batch in train_loader:
                ts_b = batch["timeseries"].to(device)
                pcc_b = batch["pcc_vector"].to(device)

                with torch.no_grad():
                    h_ts = tst1(ts_b, mode='finetune')
                h_fc = tst2(pcc_b, mode='finetune')

                loss, _, align = contrastive_module(h_ts, h_fc)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_align += align.item()

            n = len(train_loader)
            tepoch.set_postfix(loss=f"{total_loss/n:.4f}",
                               align=f"{total_align/n:.4f}")

    # ─── Guardar checkpoint completo ─────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "contrastive_global.pt"
        torch.save({
            "tst1_state_dict":        tst1.state_dict(),
            "tst2_state_dict":        tst2.state_dict(),
            "proj_head_1_state_dict": contrastive_module.proj_ts.state_dict(),
            "proj_head_2_state_dict": contrastive_module.proj_fc.state_dict(),
        }, path)
        print(f"  💾 {path}")

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

    tmp = Path(tempfile.mkdtemp())
    interp_dir = tmp / "interp"; interp_dir.mkdir()
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
        rows.append({"FILE_ID": fid, "SUB_ID": i,
                     "SITE_ID": sites[i % 3],
                     "DX_GROUP": int(rng.integers(0, 2))})
    pd.DataFrame(rows).to_csv(tmp / "meta.csv", index=False)

    config = {
        "DEVICE": "cpu", "NUM_WORKERS": 0, "SEED": 0,
        "RAW_PATH": str(interp_dir), "INTERP_PATH": str(interp_dir),
        "CSV_PATH": str(tmp / "meta.csv"), "ATLAS": "cc200", "PREFIX": "interp_",
        "N_ROIS": R, "MAX_SEQ_LEN": T, "LABEL_COL": "DX_GROUP",
        "EVAL_PROTOCOL": "kfold", "N_FOLDS": 3,
        "TST1": {"D_MODEL": 64, "DIM_FEEDFORWARD": 128, "NUM_ENCODER_LAYERS": 2,
                 "N_HEADS": 4, "ENC_DROP": 0.1, "USE_CLS_TOKEN": True},
        "TST2": {"PCC_DIM": D, "D_MODEL": 64, "N_HEADS": 4,
                 "NUM_ENCODER_LAYERS": 2, "DIM_FEEDFORWARD": 128, "ENC_DROP": 0.1},
        "DUAL_STREAM": {"FUSION_TYPE": "attention_pooling", "NUM_CLASSES": 2,
                        "CLASSIFIER_DROPOUT": 0.3, "MLP_DIMS": [256, 64, 2]},
        "FUSION": {"ATTENTION_POOLING": {"HIDDEN_DIM": 128}},
        "T_CONTRASTIVE": {
            "N_EPOCHS": 3, "BATCH_SIZE": 4, "LR": 1e-4, "WEIGHT_DECAY": 1e-4,
            "TEMPERATURE": 0.07, "PROJ_HIDDEN_DIM": 256, "PROJ_OUTPUT_DIM": 128,
            "OPTIMIZER": "Adam",
            "SCHEDULER": "CosineAnnealingLR",
            "SCHEDULER_PARAMS": {"T_max": 3, "eta_min": 1e-6},
            "PATIENCE": 5, "MIN_DELTA": 1e-4,
        },
        "FINETUNING": {"N_EPOCHS": 2, "BATCH_SIZE": 4, "LR": 5e-5,
                       "WEIGHT_DECAY": 1e-4, "OPTIMIZER": "Adam",
                       "SCHEDULER": "CosineAnnealingLR",
                       "SCHEDULER_PARAMS": {"T_max": 2, "eta_min": 5e-7}},
        "CKPT_TST1": None, "CKPT_TST2": None,
    }

    print("── TEST 1: run_contrastive (3 epochs) ──────────────────────")
    tl, vl, ta, va = run_contrastive(config, fold_idx=0, save_dir=save_dir)
    assert len(tl) == 3 and len(vl) == 3
    assert len(ta) == 3 and len(va) == 3
    assert all(-1.0 <= a <= 1.0 for a in ta + va)
    print(f"  ✓ train loss {tl[0]:.4f}→{tl[-1]:.4f}  align {ta[0]:.4f}→{ta[-1]:.4f}\n")

    print("── TEST 2: checkpoint guardado ──────────────────────────────")
    ckpt = save_dir / "best_contrastive_fold_0.pt"
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and len(state) > 0
    print(f"  ✓ {ckpt.name} ({len(state)} tensores)\n")

    print("── TEST 3: unfreeze both (encoders entrenables) ────────────")
    from training.setup import build_experiment
    cfg = {**config, "EXPERIMENT_TYPE": "contrastive"}
    exp = build_experiment(cfg, fold_idx=0)
    n_trainable = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in exp.model.parameters())
    assert n_trainable == n_total, f"{n_trainable}/{n_total} entrenables"
    print(f"  ✓ {n_trainable:,}/{n_total:,} params entrenables\n")

    shutil.rmtree(tmp)
    print("✅ Todos los tests de train_contrastive.py pasaron.")