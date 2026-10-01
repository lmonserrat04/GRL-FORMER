# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Corre SOLO la fase de finetune LOSO con local attention en TST1.

Reutiliza checkpoints del 20260928_160330_transformer_baseline.
La atención local solo se aplica en finetune (mode='finetune').

Uso:
    python scripts/train/run_local_attn.py --k 8 --seed 42
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from training.train_finetune import finetune_fold
from utils.metrics import bootstrap_confidence_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True,
                        help="Ventana local +-k (None para atencion global)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir",
                        default="experiments/20260928_160330_transformer_baseline/checkpoints",
                        help="Directorio con best_pt_ts/fc y contrastive_global")
    parser.add_argument("--out-root", default="experiments_local_attn")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise SystemExit(f"No existe {ckpt_dir}")
    for f in ["best_pt_ts_fold_0.pt", "best_pt_fc_fold_0.pt", "contrastive_global.pt"]:
        if not (ckpt_dir / f).exists():
            raise SystemExit(f"Falta {ckpt_dir / f}")

    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["CKPT_TST1"] = str(ckpt_dir / "best_pt_ts_fold_0.pt")
    cfg["CKPT_TST2"] = str(ckpt_dir / "best_pt_fc_fold_0.pt")
    cfg["CKPT_CONTRASTIVE"] = str(ckpt_dir / "contrastive_global.pt")
    cfg["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["EVAL_PROTOCOL"] = "loso"
    cfg["SEED"] = args.seed
    cfg["TST1"]["LOCAL_ATTN_WINDOW"] = args.k
    cfg["RUN_NAME"] = f"local_attn_k{args.k}_seed{args.seed}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.out_root)
    out_root.mkdir(exist_ok=True)
    exp_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg['RUN_NAME']}"
    out_dir = out_root / exp_id
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["EXP_DIR"] = str(out_dir)
    cfg["CHECKPOINTS_PATH"] = str(out_dir / "checkpoints")
    cfg["EXP_ID"] = exp_id

    import shutil
    shutil.copy("config/config.yaml", out_dir / "config.yaml")

    print(f"Local attention: k={args.k}  seed={args.seed}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Output: {out_dir}\n")

    from data.loaders.dataloader import load_raw_data
    data = load_raw_data(cfg)
    n_folds = len(np.unique(data["site_ids"]))
    print(f"LOSO folds: {n_folds}\n")

    fold_metrics = []
    for fold_idx in range(n_folds):
        print(f"\n--- Fold {fold_idx+1}/{n_folds} ---")
        m = finetune_fold(cfg, fold_idx=fold_idx,
                          save_dir=Path(cfg["CHECKPOINTS_PATH"]))
        m["fold_idx"] = fold_idx
        fold_metrics.append(m)

    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {
        "protocol": "loso", "n_folds": n_folds, "seed": args.seed,
        "local_attn_window": args.k, "exp_id": exp_id,
    }
    for metric in metric_names:
        vals = [m[metric] for m in fold_metrics]
        mean_v, std_v, lo, hi = bootstrap_confidence_interval(
            vals, n_bootstrap=1000, ci=0.95, seed=args.seed)
        summary[metric] = {"mean": mean_v, "std": std_v,
                           "ci95_lower": lo, "ci95_upper": hi}

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print(f"RESUMEN (local attn k={args.k}, LOSO, seed={args.seed})")
    print(f"{'='*60}")
    for metric in metric_names:
        s = summary[metric]
        print(f"  {metric:<12s}: {s['mean']:.4f} +/- {s['std']:.4f}  "
              f"[{s['ci95_lower']:.4f}, {s['ci95_upper']:.4f}]")
    print(f"\nGuardado en: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
