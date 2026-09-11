"""
Finetune — clasificación ASD/TC con la configuración óptima del paper.

Pipeline por fold:
  1. Cargar dual stream con pesos pretrain TST1 + TST2.
  2. Fase contrastive (50 epochs, unfreeze both, InfoNCE τ=0.07).
  3. Finetune con attention_pooling, encoders descongelados, lr=5e-5, dropout=0.3,
     early stopping patience=20 sobre val AUC.
  4. Evaluación test con agregación a nivel de sujeto (majority_vote).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.loaders.dataloader import get_finetune_loaders
from models.dual_stream import create_dual_stream_model
from training.tasks.contrastive import ContrastiveTask, ContrastiveWrapper
from training.tasks.classification import ClassificationTask
from utils.metrics import (
    compute_metrics,
    aggregate_window_predictions_to_subject_level,
)


# ──────────────────────────────────────────────────────────────────────
# Fase contrastive (opcional, previa al finetune)
# ──────────────────────────────────────────────────────────────────────

def run_contrastive_phase(model, train_loader, phase_cfg, device):
    """
    Alinea TST1 y TST2 con InfoNCE. unfreeze both encoders (Sec. 4.3.2).
    Hiperparámetros óptimos: epochs=50, lr=1e-4, wd=1e-4, τ=0.07,
    proj hidden=256, output=128.
    """
    task = ContrastiveTask(
        dim_ts=model.dim_ts,
        dim_fc=model.dim_fc,
        hidden_dim=phase_cfg["PROJ_HIDDEN_DIM"],
        output_dim=phase_cfg["PROJ_OUTPUT_DIM"],
        temperature=phase_cfg["TEMPERATURE"],
        device=device,
    ).to(device)

    model.unfreeze_encoders()

    params = list(model.parameters()) + list(task.contrastive_module.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=float(phase_cfg["LR"]),
        weight_decay=float(phase_cfg["WEIGHT_DECAY"]),
    )

    epochs = phase_cfg["N_EPOCHS"]
    for epoch in range(1, epochs + 1):
        model.train()
        task.contrastive_module.train()

        total_loss, total_align = 0.0, 0.0
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

        n = len(train_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [contrastive] epoch {epoch:3d}/{epochs} | "
                  f"loss={total_loss/n:.4f}  align={total_align/n:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Train / validate (epoch de finetune)
# ──────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds, labels = 0.0, [], []

    for batch in loader:
        ts = batch["timeseries"].to(device)
        pcc = batch["pcc_vector"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ts, pcc)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)

            logits = model(ts, pcc)
            total_loss += criterion(logits, y).item()

            p = torch.softmax(logits, dim=1)[:, 1]
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    metrics = compute_metrics(np.array(labels), np.array(preds), np.array(probs))
    return total_loss / len(loader), metrics


# ──────────────────────────────────────────────────────────────────────
# Finetune de un fold
# ──────────────────────────────────────────────────────────────────────

def finetune_fold(config, fold_idx, save_dir=None):
    """
    Entrena y evalúa un único fold con la config óptima.

    Returns:
        dict de métricas test (subject-level si hay varias ventanas por sujeto).
    """
    device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    phase = config["FINETUNING"]
    ds = config["DUAL_STREAM"]
    fusion_hidden = config.get("FUSION", {}).get("ATTENTION_POOLING", {}).get("HIDDEN_DIM")

    # 1. Loaders
    train_loader, val_loader, test_loader, split_info = get_finetune_loaders(
        config,
        batch_size=phase["BATCH_SIZE"],
        num_workers=config.get("NUM_WORKERS", 0),
        fold_idx=fold_idx,
        n_folds=config.get("N_FOLDS", 5),
        seed=config.get("SEED", 42),
        eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
    )

    # 2. Modelo dual stream con attention_pooling + MLP [256, 64, 2]
    model = create_dual_stream_model(
        n_rois=config["N_ROIS"],
        time_points=config["MAX_SEQ_LEN"],
        pcc_dim=config["TST2"]["PCC_DIM"],
        tst1_emb_dim=config["TST1"]["D_MODEL"],
        tst2_d_model=config["TST2"]["D_MODEL"],
        fusion_type=ds["FUSION_TYPE"],
        fusion_hidden_dim=fusion_hidden,
        num_classes=ds["NUM_CLASSES"],
        dropout=ds["CLASSIFIER_DROPOUT"],
        mlp_dims=ds.get("MLP_DIMS"),
    ).to(device)

    # 3. Cargar checkpoints de pretrain si están
    if config.get("CKPT_TST1"):
        model.load_pretrained_tst1(config["CKPT_TST1"], strict=False)
        print(f"  TST1 ← {config['CKPT_TST1']}")
    if config.get("CKPT_TST2"):
        model.load_pretrained_tst2(config["CKPT_TST2"], strict=False)
        print(f"  TST2 ← {config['CKPT_TST2']}")

    # 4. Fase contrastive (opcional pero óptima)
    if config.get("RUN_CONTRASTIVE", True):
        print("\n── Fase contrastive (unfreeze both, InfoNCE τ=0.07) ──")
        run_contrastive_phase(model, train_loader, config["T_CONTRASTIVE"], device)

    # 5. Finetune (encoders descongelados + attention_pooling)
    print("\n── Finetune (encoders descongelados, attention_pooling) ──")
    model.unfreeze_encoders()   # garantía

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(phase["LR"]),
        weight_decay=float(phase["WEIGHT_DECAY"]),
    )
    epochs = phase["N_EPOCHS"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(phase["LR"]) * 0.01
    )
    criterion = nn.CrossEntropyLoss()

    patience = phase.get("PATIENCE", 20)
    best_val_auc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        val_auc = val_metrics["auc"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"val_auc={val_auc:.4f}")

        # Criterio de selección: AUC (paper: "AUC was used as the primary metric")
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping en epoch {epoch} (best val AUC={best_val_auc:.4f})")
                break

    # 6. Restaurar el mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state)

    # 7. Evaluación test (recolección de predicciones)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)

            logits = model(ts, pcc)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 8. Agregación a nivel de sujeto (mayority_vote, Sec. 4.3.4)
    subj_indices = split_info["subject_indices"]
    test_idx = split_info["test_idx"]
    n_test_subjects = len(np.unique(subj_indices[test_idx]))

    if len(test_idx) > n_test_subjects:
        yt, yp, yprob = aggregate_window_predictions_to_subject_level(
            all_labels, all_preds, all_probs,
            test_idx, subj_indices,
            strategy=config.get("SUBJECT_AGG", "majority_vote"),
        )
        print(f"\n  Fold {fold_idx}: evaluación subject-level "
              f"({len(test_idx)} ventanas → {n_test_subjects} sujetos)")
    else:
        yt, yp, yprob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
        print(f"\n  Fold {fold_idx}: evaluación window-level")

    test_metrics = compute_metrics(yt, yp, yprob)
    print(f"  AUC={test_metrics['auc']:.4f}  ACC={test_metrics['accuracy']:.4f}  "
          f"Sens={test_metrics['sensitivity']:.4f}  Spec={test_metrics['specificity']:.4f}  "
          f"F1={test_metrics['f1']:.4f}")

    # 9. Guardar
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"best_finetune_fold_{fold_idx}.pt"
        torch.save({"model_state_dict": model.state_dict(),
                    "metrics": test_metrics}, path)
        print(f"  Checkpoint: {path}")

    return test_metrics


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    import tempfile
    import shutil

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
        "EVAL_PROTOCOL": "kfold", "N_FOLDS": 3, "SUBJECT_AGG": "majority_vote",
        "TST1": {"D_MODEL": 64, "DIM_FEEDFORWARD": 128, "NUM_ENCODER_LAYERS": 2,
                 "N_HEADS": 4, "ENC_DROP": 0.1, "USE_CLS_TOKEN": True},
        "TST2": {"PCC_DIM": D, "D_MODEL": 64, "N_HEADS": 4,
                 "NUM_ENCODER_LAYERS": 2, "DIM_FEEDFORWARD": 128, "ENC_DROP": 0.1},
        "DUAL_STREAM": {"FUSION_TYPE": "attention_pooling", "NUM_CLASSES": 2,
                        "CLASSIFIER_DROPOUT": 0.3, "MLP_DIMS": [256, 64, 2]},
        "FUSION": {"ATTENTION_POOLING": {"HIDDEN_DIM": 128}},
        "T_CONTRASTIVE": {
            "N_EPOCHS": 2, "BATCH_SIZE": 4, "LR": 1e-4, "WEIGHT_DECAY": 1e-4,
            "TEMPERATURE": 0.07, "PROJ_HIDDEN_DIM": 256, "PROJ_OUTPUT_DIM": 128,
            "OPTIMIZER": "Adam",
        },
        "FINETUNING": {
            "N_EPOCHS": 4, "BATCH_SIZE": 4, "LR": 5e-5, "WEIGHT_DECAY": 1e-4,
            "OPTIMIZER": "Adam", "PATIENCE": 3,
        },
        "RUN_CONTRASTIVE": True,
        "CKPT_TST1": None, "CKPT_TST2": None,
    }

    print("── TEST 1: finetune_fold (contrastive + finetune) ───────────")
    m = finetune_fold(config, fold_idx=0, save_dir=save_dir)
    assert set(m.keys()) >= {"auc", "accuracy", "sensitivity", "specificity", "f1"}
    assert 0.0 <= m["accuracy"] <= 1.0
    print(f"  ✓ métricas: {m}\n")

    print("── TEST 2: checkpoint guardado ──────────────────────────────")
    ckpt = save_dir / "best_finetune_fold_0.pt"
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in blob and "metrics" in blob
    print(f"  ✓ {ckpt.name}\n")

    print("── TEST 3: sin fase contrastive (RUN_CONTRASTIVE=False) ────")
    cfg2 = {**config, "RUN_CONTRASTIVE": False}
    m2 = finetune_fold(cfg2, fold_idx=0, save_dir=None)
    assert 0.0 <= m2["accuracy"] <= 1.0
    print(f"  ✓ ACC={m2['accuracy']:.4f}\n")

    shutil.rmtree(tmp)
    print("✅ Todos los tests de train_finetune.py pasaron.")