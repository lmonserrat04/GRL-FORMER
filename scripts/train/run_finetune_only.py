# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Corre SOLO la fase de finetune LOSO, reutilizando checkpoints de
pretrain + contrastive ya existentes.

Uso:
    python scripts/train/run_finetune_only.py \\
        --ckpt-dir experiments/20260928_160330_transformer_baseline/checkpoints \\
        --seed 42 \\
        --run-name twotst_original_loso_preds
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from training.train_finetune import finetune_fold
from utils.metrics import bootstrap_confidence_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True,
                        help="Directorio con best_pt_ts_fold_0.pt, best_pt_fc_fold_0.pt, contrastive_global.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="finetune_only")
    parser.add_argument("--out-root", default="experiments_finetune_only")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override FINETUNING.N_EPOCHS (opcional, para test rápido)")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise SystemExit(f"No existe {ckpt_dir}")
    required = ["best_pt_ts_fold_0.pt", "best_pt_fc_fold_0.pt", "contrastive_global.pt"]
    for f in required:
        p = ckpt_dir / f
        if not p.exists():
            raise SystemExit(f"Falta {p}")

    # Cargar config base del repo
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ajustes
    cfg["CKPT_TST1"] = str(ckpt_dir / "best_pt_ts_fold_0.pt")
    cfg["CKPT_TST2"] = str(ckpt_dir / "best_pt_fc_fold_0.pt")
    cfg["CKPT_CONTRASTIVE"] = str(ckpt_dir / "contrastive_global.pt")
    cfg["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["EVAL_PROTOCOL"] = "loso"
    cfg["SEED"] = args.seed
    cfg["RUN_NAME"] = args.run_name
    if args.n_epochs is not None:
        cfg["FINETUNING"]["N_EPOCHS"] = args.n_epochs

    # Semilla
    try:
        from utils.seed import set_seed
        set_seed(args.seed)
    except ImportError:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Directorio de salida
    out_root = Path(args.out_root)
    out_root.mkdir(exist_ok=True)
    exp_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    out_dir = out_root / exp_id
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["EXP_DIR"] = str(out_dir)
    cfg["CHECKPOINTS_PATH"] = str(out_dir / "checkpoints")
    cfg["EXP_ID"] = exp_id

    # Copiar config usado
    shutil.copy("config/config.yaml", out_dir / "config.yaml")

    print(f"Seed:        {args.seed}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Run name:    {args.run_name}")
    print(f"Output:      {out_dir}\n")

    # LOSO folds
    from data.loaders.dataloader import load_raw_data
    data = load_raw_data(cfg)
    n_folds = len(np.unique(data["site_ids"]))
    print(f"LOSO folds: {n_folds}\n")

    fold_metrics = []
    for fold_idx in range(n_folds):
        print(f"\n─── Fold {fold_idx+1}/{n_folds} ───")
        m = finetune_fold(cfg, fold_idx=fold_idx,
                          save_dir=Path(cfg["CHECKPOINTS_PATH"]))
        m["fold_idx"] = fold_idx
        fold_metrics.append(m)

    # Resumen
    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {
        "protocol": "loso",
        "n_folds": n_folds,
        "seed": args.seed,
        "exp_id": exp_id,
        "run_name": args.run_name,
        "ckpt_dir": str(ckpt_dir),
    }
    for metric in metric_names:
        vals = [m[metric] for m in fold_metrics]
        mean_v, std_v, lo, hi = bootstrap_confidence_interval(
            vals, n_bootstrap=1000, ci=0.95, seed=args.seed)
        summary[metric] = {"mean": mean_v, "std": std_v,
                           "ci95_lower": lo, "ci95_upper": hi}

    # Guardar results.json
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print(f"RESUMEN ({args.run_name}, LOSO, seed={args.seed})")
    print(f"{'='*60}")
    for metric in metric_names:
        s = summary[metric]
        print(f"  {metric:<12s}: {s['mean']:.4f} +/- {s['std']:.4f}  "
              f"[{s['ci95_lower']:.4f}, {s['ci95_upper']:.4f}]")
    print(f"\nGuardado en: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
