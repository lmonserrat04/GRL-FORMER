"""Entrenamiento y evaluación LOSO de la GNN temporal.

Contiene únicamente lógica de training/eval/artefactos. La construcción de
modelo, loaders y optimizador vive en `training.gnn_setup`; el entrypoint CLI
vive en `scripts/train/run_gnn.py`.

Artefactos por semilla (dentro del directorio del experimento):
    - `preds_seed{seed}.npz`: probs/preds out-of-fold de todos los sujetos.
    - `site_auc.csv`, `site_auc.json`, `site_auc.md`: AUC por sitio.
    - `training_curves.json`: train/val loss y val AUC por época y fold.
    - `checkpoints/loss_history_fold_{i}.csv`: la misma curva, por fold.
    - `checkpoints/preds_fold_{i}.npz`: predicciones y umbral de cada fold.
"""

from __future__ import annotations

import csv
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
    site: str,
    history: list[dict],
    best_epoch: int,
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
    test_subject_ids = np.asarray(subject_indices)[np.asarray(test_idx)]
    test_sites = np.asarray([site] * len(test_subject_ids), dtype=str)
    per_fold_preds = (p_test >= 0.5).astype(int)
    per_fold_preds_youden = (p_test >= optimal_thr).astype(int)

    np.savez(
        ckpt_dir / f"preds_fold_{fold_idx}.npz",
        labels=y_test,
        probs=p_test,
        probs_asd=p_test,
        preds=per_fold_preds,
        preds_youden=per_fold_preds_youden,
        test_idx=test_idx,
        subject_indices=subject_indices,
        subject_ids=test_subject_ids,
        test_subject_ids=test_subject_ids,
        test_sites=test_sites,
        fold_idx=np.array([fold_idx]),
        seed=np.array([int(cfg["SEED"])]),
        val_labels=y_val,
        val_probs=p_val,
        val_preds=(p_val >= 0.5).astype(int),
        optimal_thr=np.array([optimal_thr]),
        best_epoch=np.array([best_epoch]),
        n_epochs_run=np.array([len(history)]),
    )

    # Historial de entrenamiento para curvas de pérdida por fold.
    history_path = ckpt_dir / f"loss_history_fold_{fold_idx}.csv"
    with open(history_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold_idx", "site", "epoch", "train_loss", "val_loss",
                "val_auc", "val_acc", "lr", "is_best",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow({
                "fold_idx": fold_idx,
                "site": site,
                "epoch": row["epoch"],
                "train_loss": f"{row['train_loss']:.8f}",
                "val_loss": f"{row['val_loss']:.8f}",
                "val_auc": f"{row['val_auc']:.8f}",
                "val_acc": f"{row['val_acc']:.8f}",
                "lr": f"{row['lr']:.10g}",
                "is_best": int(row["is_best"]),
            })


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
    print(f"  T={t_use} | d={g['D_MODEL']} | k={g['KNN']} | "
          f"layers={g['NUM_LAYERS']} | stride={g['STRIDE']} | "
          f"chunk={g['TIME_CHUNK']}")

    best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
    history: list[dict] = []
    t0 = time.time()
    epochs = int(g["N_EPOCHS"])

    with tqdm(range(1, epochs + 1), unit="epoch", leave=False) as pbar:
        for epoch in pbar:
            train_loss = train_epoch(model, train_loader, optimizer, task, device, t_use)
            val_loss, y_val, p_val = evaluate(model, val_loader, task, device, t_use)
            val_metrics = compute_metrics(
                y_val, (p_val >= 0.5).astype(int), p_val
            )
            val_auc = val_metrics["auc"]
            lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()

            improved = val_auc > best_auc + 1e-4
            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "val_acc": float(val_metrics["accuracy"]),
                "lr": lr,
                "is_best": bool(improved),
            })

            pbar.set_postfix(train=f"{train_loss:.4f}", val_auc=f"{val_auc:.4f}")
            if improved:
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
        site=site, history=history, best_epoch=best_epoch,
    )

    elapsed = (time.time() - t0) / 60.0
    print(f"  -> AUC {metrics['auc']:.4f} | ACC {metrics['accuracy']:.4f} | "
          f"Sens {metrics['sensitivity']:.4f} | Spec {metrics['specificity']:.4f} | "
          f"F1 {metrics['f1']:.4f}")
    print(f"  -> Youden thr {optimal_thr:.4f} | "
          f"Sens {metrics_youden['sensitivity']:.4f} | "
          f"Spec {metrics_youden['specificity']:.4f} | "
          f"F1 {metrics_youden['f1']:.4f} | {elapsed:.1f} min")

    metrics.update({f"youden_{k}": v for k, v in metrics_youden.items()})
    metrics["fold_idx"] = fold_idx
    metrics["test_site"] = site
    metrics["n_test"] = int(len(y_test))
    metrics["n_asd"] = int((y_test == 1).sum())
    metrics["n_td"] = int((y_test == 0).sum())
    metrics["best_epoch"] = best_epoch
    metrics["epochs_run"] = int(len(history))
    metrics["optimal_thr"] = optimal_thr
    metrics["elapsed_min"] = elapsed
    return metrics


# ──────────────────────────────────────────────────────────────────────
# LOSO completo + resumen
# ──────────────────────────────────────────────────────────────────────
def run_loso(cfg: dict, folds: list[int], ckpt_dir: Path) -> list[dict]:
    return [fit_fold(cfg, fold_idx, ckpt_dir) for fold_idx in folds]


def site_records(metrics_list: list[dict]) -> list[dict]:
    """Una fila por sitio/fold con AUC y métricas umbralizadas."""
    records = []
    for m in metrics_list:
        records.append({
            "site": str(m["test_site"]),
            "fold_idx": int(m["fold_idx"]),
            "n": int(m["n_test"]),
            "n_asd": int(m["n_asd"]),
            "n_td": int(m["n_td"]),
            "auc": float(m["auc"]),
            "auc_valid": bool(m["n_asd"] > 0 and m["n_td"] > 0),
            "accuracy": float(m["accuracy"]),
            "sensitivity": float(m["sensitivity"]),
            "specificity": float(m["specificity"]),
            "f1": float(m["f1"]),
            "youden_sensitivity": float(m.get("youden_sensitivity", 0.0)),
            "youden_specificity": float(m.get("youden_specificity", 0.0)),
            "youden_f1": float(m.get("youden_f1", 0.0)),
            "optimal_thr": float(m["optimal_thr"]),
            "best_epoch": int(m["best_epoch"]),
            "epochs_run": int(m["epochs_run"]),
            "elapsed_min": float(m["elapsed_min"]),
        })
    return records


def save_seed_artifacts(
    metrics_list: list[dict],
    cfg: dict,
    exp_dir: Path,
    ckpt_dir: Path,
) -> dict:
    """Guarda al final de la semilla:

    - `preds_seed{seed}.npz`: probs/preds de todos los sujetos out-of-fold.
    - `site_auc.csv`, `site_auc.json`, `site_auc.md`: AUC por sitio.
    - `training_curves.json`: historial train/val de cada fold.
    """
    exp_dir = Path(exp_dir)
    ckpt_dir = Path(ckpt_dir)
    seed = int(cfg["SEED"])
    records = site_records(metrics_list)

    # ── AUC por sitio: CSV, JSON y Markdown ──────────────────────────
    site_csv = exp_dir / "site_auc.csv"
    if records:
        with open(site_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    site_json = exp_dir / "site_auc.json"
    with open(site_json, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "run_name": cfg.get("RUN_NAME"),
            "protocol": "loso",
            "n_sites": len(records),
            "per_site": records,
        }, f, indent=2, ensure_ascii=False)

    # ── Predicciones agregadas de todos los sujetos ───────────────────
    subjects: dict[int, dict] = {}
    for path in sorted(ckpt_dir.glob("preds_fold_*.npz")):
        with np.load(path, allow_pickle=True) as z:
            if "test_subject_ids" in z.files:
                subject_ids = z["test_subject_ids"]
            else:
                subject_ids = np.asarray(z["subject_indices"])[np.asarray(z["test_idx"])]

            if "test_sites" in z.files:
                site = str(z["test_sites"][0])
            else:
                site = ""
            fold_idx = int(z["fold_idx"][0]) if "fold_idx" in z.files else int(
                path.stem.split("_")[-1]
            )
            thr = float(z["optimal_thr"][0])

            labels = np.asarray(z["labels"])
            probs = np.asarray(z["probs"])
            for i, sid in enumerate(subject_ids):
                sid_int = int(sid)
                subjects[sid_int] = {
                    "label": int(labels[i]),
                    "prob": float(probs[i]),
                    "pred": int(probs[i] >= 0.5),
                    "pred_youden": int(probs[i] >= thr),
                    "site": site,
                    "fold_idx": fold_idx,
                    "optimal_thr": thr,
                }

    preds_path = exp_dir / f"preds_seed{seed}.npz"
    if subjects:
        order = sorted(subjects)
        rows = [subjects[sid] for sid in order]
        np.savez(
            preds_path,
            subject_ids=np.asarray(order, dtype=np.int64),
            labels=np.asarray([r["label"] for r in rows], dtype=np.int64),
            probs=np.asarray([r["prob"] for r in rows], dtype=np.float64),
            probs_asd=np.asarray([r["prob"] for r in rows], dtype=np.float64),
            preds=np.asarray([r["pred"] for r in rows], dtype=np.int64),
            preds_youden=np.asarray(
                [r["pred_youden"] for r in rows], dtype=np.int64
            ),
            sites=np.asarray([r["site"] for r in rows], dtype=str),
            folds=np.asarray([r["fold_idx"] for r in rows], dtype=np.int64),
            optimal_thr=np.asarray(
                [r["optimal_thr"] for r in rows], dtype=np.float64
            ),
            seed=np.array([seed], dtype=np.int64),
        )

    # ── Curvas de pérdida por fold ────────────────────────────────────
    curves = []
    for path in sorted(ckpt_dir.glob("loss_history_fold_*.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            history = list(csv.DictReader(f))
        for row in history:
            for key in ["epoch", "fold_idx", "is_best"]:
                row[key] = int(row[key])
            for key in ["train_loss", "val_loss", "val_auc", "val_acc", "lr"]:
                row[key] = float(row[key])
        fold_idx = int(path.stem.split("_")[-1])
        site = history[0]["site"] if history else ""
        curves.append({
            "fold_idx": fold_idx,
            "site": site,
            "history": history,
        })

    curves_path = exp_dir / "training_curves.json"
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "run_name": cfg.get("RUN_NAME"),
            "folds": curves,
        }, f, indent=2, ensure_ascii=False)

    # ── Informe legible de AUC por sitio ──────────────────────────────
    md_lines = [
        f"# AUC por sitio — seed {seed}",
        "",
        "| Sitio | Fold | n | ASD | TD | AUC | ACC | Sens | Spec | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(records, key=lambda x: x["site"]):
        auc_str = f"{r['auc']:.4f}" if r["auc_valid"] else "n/d"
        md_lines.append(
            f"| {r['site']} | {r['fold_idx']} | {r['n']} | {r['n_asd']} | "
            f"{r['n_td']} | {auc_str} | {r['accuracy']:.4f} | "
            f"{r['sensitivity']:.4f} | {r['specificity']:.4f} | {r['f1']:.4f} |"
        )
    valid_aucs = [r["auc"] for r in records if r["auc_valid"]]
    if valid_aucs:
        md_lines.extend([
            "",
            f"**AUC media por sitio:** {np.mean(valid_aucs):.4f}",
        ])
    (exp_dir / "site_auc.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "preds": preds_path if subjects else None,
        "site_auc_csv": site_csv,
        "site_auc_json": site_json,
        "site_auc_md": exp_dir / "site_auc.md",
        "curves": curves_path,
    }


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
        "per_site": site_records(metrics_list),
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
