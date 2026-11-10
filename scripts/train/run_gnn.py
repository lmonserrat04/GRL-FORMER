#!/usr/bin/env python
"""Entrypoint CLI para entrenar la GNN temporal con LOSO.

Toda la lógica vive en `training/`:
    - training.gnn_setup      → configuración y factorías
    - training.train_gnn     → train/eval/folds/resumen
    - models.gnn_temporal    → arquitectura

Uso:
    python scripts/train/run_gnn.py --seed 42 --run-name gnn_k8_d32
    python scripts/train/run_gnn.py --folds 0 --n-epochs 2 \
        --run-name gnn_smoke --out-root /tmp/gnn_smoke
"""

# --- sys.path bootstrap (permite ejecutar el script desde cualquier cwd) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

import numpy as np
import yaml

from data.loaders.dataloader import load_raw_data
from training.gnn_setup import resolve_gnn_config
from training.train_gnn import run_loso, save_summary, summarize
from utils.experiment import create_experiment_dir
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser("GNN temporal LOSO para ABIDE I")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="gnn_temporal")
    p.add_argument("--out-root", default="experiments_gnn")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="Folds a correr (default: todos los sitios LOSO)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--t", type=int, default=None, help="Timesteps (default: GNN.T)")
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--knn", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--time-chunk", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    overrides = {
        "N_EPOCHS": args.n_epochs,
        "PATIENCE": args.patience,
        "BATCH_SIZE": args.batch_size,
        "LR": args.lr,
        "WEIGHT_DECAY": args.weight_decay,
        "T": args.t,
        "D_MODEL": args.d_model,
        "KNN": args.knn,
        "WINDOW": args.window,
        "NUM_LAYERS": args.layers,
        "DROPOUT": args.dropout,
        "STRIDE": args.stride,
        "TIME_CHUNK": args.time_chunk,
    }
    cfg = resolve_gnn_config(cfg, overrides)
    cfg["RUN_NAME"] = args.run_name
    cfg["SEED"] = args.seed
    cfg["NUM_WORKERS"] = args.num_workers
    if args.device:
        cfg["DEVICE"] = args.device

    cfg = create_experiment_dir(cfg, root=args.out_root, config_path=args.config)
    exp_dir = Path(cfg["EXP_DIR"])
    ckpt_dir = Path(cfg["CHECKPOINTS_PATH"])

    set_seed(args.seed)
    data = load_raw_data(cfg)
    n_folds = len(np.unique(data["site_ids"]))
    folds = list(range(n_folds)) if args.folds is None else [
        f for f in args.folds if 0 <= f < n_folds
    ]
    if not folds:
        raise SystemExit(f"No hay folds válidos (0..{n_folds - 1})")

    g = cfg["GNN"]
    print("=" * 78)
    print(f"GNN temporal | run={args.run_name} | seed={args.seed} | device={cfg['DEVICE']}")
    print(f"Sujetos={len(data['labels'])} | sitios={n_folds} | T={g['T']} | "
          f"d={g['D_MODEL']} | k={g['KNN']} | W={g['WINDOW']} | "
          f"layers={g['NUM_LAYERS']} | stride={g['STRIDE']} | "
          f"batch={g['BATCH_SIZE']} | epochs={g['N_EPOCHS']}")
    print(f"Folds: {folds}")
    print(f"Salida: {exp_dir}")
    print("=" * 78)

    metrics = run_loso(cfg, folds, ckpt_dir)
    summary = summarize(metrics, cfg, folds)
    results_path = save_summary(summary, exp_dir)

    print("\n" + "=" * 78)
    print(f"RESUMEN ({args.run_name}, LOSO, seed={args.seed})")
    print("=" * 78)
    for name in ["auc", "accuracy", "sensitivity", "specificity", "f1"]:
        if name in summary:
            s = summary[name]
            print(f"  {name:<12s}: {s['mean']:.4f} +/- {s['std']:.4f}  "
                  f"[{s['ci95_lower']:.4f}, {s['ci95_upper']:.4f}]")
    print(f"\nResultados: {results_path}")


if __name__ == "__main__":
    main()
