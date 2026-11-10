"""Factorías y configuración de la GNN temporal.

Centraliza:
    - defaults del bloque `GNN:` de config.yaml
    - resolución de overrides de CLI
    - construcción de modelo, optimizador, scheduler y loaders LOSO
"""

from __future__ import annotations

from typing import Any

import torch

from data.loaders.dataloader import get_finetune_loaders
from models.gnn_temporal import TemporalGNN


GNN_DEFAULTS: dict[str, Any] = {
    "T": 116,
    "D_MODEL": 16,
    "KNN": 5,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.3,
    "STRIDE": 1,
    "CLF_HIDDEN": [32, 16],
    "CLF_DROPOUT": 0.5,
    "N_EPOCHS": 500,
    "BATCH_SIZE": 32,
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "PATIENCE": 30,
}


def resolve_gnn_config(cfg: dict, overrides: dict | None = None) -> dict:
    """Fusiona defaults → config.yaml → CLI y normaliza las claves comunes."""
    gnn = dict(GNN_DEFAULTS)
    gnn.update(cfg.get("GNN", {}) or {})
    for key, value in (overrides or {}).items():
        if value is not None:
            gnn[key] = value

    if int(gnn["D_MODEL"]) < 1:
        raise ValueError("GNN.D_MODEL debe ser >= 1")
    if int(gnn["KNN"]) < 1:
        raise ValueError("GNN.KNN debe ser >= 1")

    cfg["GNN"] = gnn
    cfg["EVAL_PROTOCOL"] = "loso"
    cfg["DEVICE"] = cfg.get("DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return cfg


def build_gnn_model(cfg: dict) -> TemporalGNN:
    g = cfg["GNN"]
    return TemporalGNN(
        n_rois=int(cfg["N_ROIS"]),
        d_model=int(g["D_MODEL"]),
        knn=int(g["KNN"]),
        num_layers=int(g["NUM_LAYERS"]),
        dropout=float(g["DROPOUT"]),
        stride=int(g.get("STRIDE", 1)),
        clf_hidden=tuple(int(x) for x in g.get("CLF_HIDDEN", (32, 16))),
        clf_dropout=float(g.get("CLF_DROPOUT", 0.5)),
    ).to(cfg["DEVICE"])


def build_gnn_optimizer(model: torch.nn.Module, cfg: dict):
    g = cfg["GNN"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(g["LR"]),
        weight_decay=float(g["WEIGHT_DECAY"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(g["N_EPOCHS"]), 1),
        eta_min=float(g["LR"]) * 0.01,
    )
    return optimizer, scheduler


def build_gnn_loaders(cfg: dict, fold_idx: int):
    g = cfg["GNN"]
    return get_finetune_loaders(
        cfg,
        batch_size=int(g["BATCH_SIZE"]),
        num_workers=int(cfg.get("NUM_WORKERS", 0)),
        fold_idx=fold_idx,
        n_folds=int(cfg.get("N_FOLDS", 5)),
        seed=int(cfg["SEED"]),
        eval_protocol="loso",
    )
