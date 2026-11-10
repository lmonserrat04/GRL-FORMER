"""Entrenamiento y evaluación LOSO de la GNN temporal.

Contiene únicamente lógica de training/eval/artefactos. La construcción de
modelo, loaders y optimizador vive en `training.gnn_setup`; el entrypoint CLI
vive en `scripts/train/run_gnn.py`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from tqdm import tqdm

from training.gnn_setup import build_gnn_loaders, build_gnn_model, build_gnn_optimizer
from training.tasks.gnn_classification import GNNClassificationTask
from utils.metrics import (
    bootstrap_confidence_interval,
    compute_metrics,
    get_reproducibility_info,
)
from utils.seed import set_seed


# ──────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────
def timesteps_to_use(cfg: dict, available: int) -> int | None:
    """Recorta a GNN.T sin fallar si el sujeto tiene menos pasos."""
    t_cfg = int(cfg["GNN"].get("T", 0) or 0)
    if t_cfg <= 0:
        return None
    return min(t_cfg, available)


def _slice_ts(ts: torch.Tensor, t_use: int | None) -> torch.Tensor:
    if t_use is None or t_use >= ts.size(1):
        return ts
    return ts[:, :t_use]


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return float(thresholds[(tpr - fpr).argmax()])


# ──────────────────────────────────────────────────────────────────────
# Una epoch
# ──────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, task, device, t_use=None) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        ts = _slice_ts(batch["timeseries"], t_use).to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = task.execution_step(model, ts, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, task, device, t_use=None):
    """Devuelve (loss, labels, probs_positiva)."""
    model.eval()
    total, n = 0.0, 0
    labels, probs = [], []
    for batch in loader:
        ts = _slice_ts(batch["timeseries"], t_use).to(device)
        y = batch["label"].to(device)

        logits, loss = task.execution_step(model, ts, y, return_logits=True)
        total += loss.item() * len(y)
        n += len(y)
        probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.extend(y.cpu().numpy())
    return total / max(n, 1), np.asarray(labels), np.asarray(probs)


# ──────────────────────────────────────────────────────────────────────
# Artefactos por fold
# ──────────────────────────────────────────────────────────────────────
def save_fold_artifacts(
    ckpt_dir: Path,
    fold_idx: int,
    model,
    cfg: dict,
    y_val,
    p_val,
    y_test,
    p_test,
    split_info: dict,
    optimal_thr: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "gnn_cfg": cfg["GNN"],
            "fold_idx": fold_idx,
        },
        ckpt_dir / f"best_gnn_fold_{fold_idx}.pt",
    )

    test_idx = split_info["test_idx"]
    subject_indices = split_info["subject_indices"]
    np.savez(
        ckpt_dir / f"preds_fold_{fold_idx}.npz",
        labels=y_test,
        probs=p_test,
        preds=(p_test >= 0.5).astype(int),
        test_idx=test_idx,
        subject_indices=subject_indices,
        test_subject_ids=np.asarray(subject_indices)[np.asarray(test_idx)],
        val_labels=y_val,
        val_probs=p_val,
        val_preds=(p_val >= 0.5).astype(int),
        optimal_thr=np.array([optimal_thr]),
    )


# ──────────────────────────────────────────────────────────────────────
# Un fold LOSO
# ──────────────────────────────────────────────────────────────────────
def fit_fold(cfg: dict, fold_idx: int, ckpt_dir: Path) -> dict:
    g = cfg["GNN"]
    device = cfg["DEVICE"]

    set_seed(int(cfg["SEED"]) + fold_idx)
    train_loader, val_loader, test_loader, split_info = build_gnn_loaders(cfg, fold_idx)

    model = build_gnn_model(cfg)
    task = GNNClassificationTask(device)
    optimizer, scheduler = build_gnn_optimizer(model, cfg)

    t_use = timesteps_to_use(cfg, int(train_loader.dataset.timeseries.shape[1]))
    site = split_info.get("test_site", f"fold{fold_idx}")
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"\n{'─' * 74}")
    print(f"Fold {fold_idx:2d} | {site:>10s} | train={n_train} val={n_val} test={n_test}")
    print(f"  T={t_use} | d={g['D_MODEL']} | k={g['KNN']} | W={g['WINDOW']} | "
          f"layers={g['NUM_LAYERS']} | stride={g['STRIDE']}")

    best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.time()
    epochs = int(g["N_EPOCHS"])

    with tqdm(range(1, epochs + 1), unit="epoch", leave=False) as pbar:
        for epoch in pbar:
            train_loss = train_epoch(model, train_loader, optimizer, task, device, t_use)
            _, y_val, p_val = evaluate(model, val_loader, task, device, t_use)
            val_auc = compute_metrics(y_val, (p_val >= 0.5).astype(int), p_val)["auc"]
            scheduler.step()

            pbar.set_postfix(train=f"{train_loss:.4f}", val_auc=f"{val_auc:.4f}")
            if val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
                if bad >= int(g["PATIENCE"]):
                    tqdm.write(
                        f"  Early stop epoch {epoch} | best epoch {best_epoch} "
                        f"| val AUC {best_auc:.4f}"
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, y_val, p_val = evaluate(model, val_loader, task, device, t_use)
    optimal_thr = youden_threshold(y_val, p_val)
    _, y_test, p_test = evaluate(model, test_loader, task, device, t_use)

    metrics = compute_metrics(y_test, (p_test >= 0.5).astype(int), p_test)
    metrics_youden = compute_metrics(
        y_test, (p_test >= optimal_thr).astype(int), p_test
    )

    save_fold_artifacts(
        ckpt_dir, fold_idx, model, cfg,
        y_val, p_val, y_test, p_test, split_info, optimal_thr,
    )

    elapsed = (time.time() - t0) / 60.0
    print(f"  -> AUC {metrics['auc']:.4f} | ACC {metrics['accuracy']:.4f} | "
          f"Sens {metrics['sensitivity']:.4f} | Spec {metrics['specificity']:.4f} | "
          f"F1 {metrics['f1']:.4f}")
    print(f"  -> Youden thr {optimal_thr:.4f} | "
          f"Sens {metrics_youden['sensitivity']:.4f} | "
          f"Spec {metrics_youden['specificity']:.4f} | "
          f"F1 {metrics_youden['f1']:.4f} | {elapsed:.1f} min")

    metrics.update(
        {f"youden_{k}": v for k, v in metrics_youden.items() if k not in metrics}
    )
    metrics["fold_idx"] = fold_idx
    metrics["test_site"] = site
    metrics["best_epoch"] = best_epoch
    metrics["optimal_thr"] = optimal_thr
    metrics["elapsed_min"] = elapsed
    return metrics


# ──────────────────────────────────────────────────────────────────────
# LOSO completo + resumen
# ──────────────────────────────────────────────────────────────────────
def run_loso(cfg: dict, folds: list[int], ckpt_dir: Path) -> list[dict]:
    return [fit_fold(cfg, fold_idx, ckpt_dir) for fold_idx in folds]


def summarize(metrics_list: list[dict], cfg: dict, folds: list[int]) -> dict:
    names = [
        "auc", "accuracy", "sensitivity", "specificity", "f1",
        "youden_sensitivity", "youden_specificity", "youden_f1",
    ]
    summary = {
        "protocol": "loso",
        "n_folds": len(metrics_list),
        "seed": cfg["SEED"],
        "run_name": cfg["RUN_NAME"],
        "model": "TemporalGNN",
        "gnn": cfg["GNN"],
        "folds_run": folds,
    }
    for name in names:
        values = [m[name] for m in metrics_list if name in m]
        if not values:
            continue
        mean, std, lo, hi = bootstrap_confidence_interval(
            values, n_bootstrap=1000, ci=0.95, seed=int(cfg["SEED"])
        )
        summary[name] = {
            "mean": mean, "std": std, "ci95_lower": lo, "ci95_upper": hi,
        }

    summary["all_folds"] = metrics_list
    summary["reproducibility"] = get_reproducibility_info()
    return summary


def save_summary(summary: dict, exp_dir: Path) -> Path:
    path = exp_dir / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    return path
