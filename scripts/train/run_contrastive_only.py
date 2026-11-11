# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Corre SOLO la fase contrastiva, reutilizando los checkpoints de pretrain.

Uso:
    python scripts/train/run_contrastive_only.py \
        --ckpt-dir experiments/20260928_160330_transformer_baseline/checkpoints \
        --seed 42 \
        --run-name dx_site_contrastive_seed42

Salida (por defecto en experiments_contrastive_only/<timestamp>_<run-name>/):
    config.yaml
    results.json
    checkpoints/contrastive_global.pt
    checkpoints/contrastive_history.json
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from training.train_contrastive import run_contrastive_global


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True,
                        help="Directorio con best_pt_ts_fold_0.pt y best_pt_fc_fold_0.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="contrastive_only")
    parser.add_argument("--out-root", default="experiments_contrastive_only")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override T_CONTRASTIVE.N_EPOCHS (opcional)")
    parser.add_argument("--objective", default=None,
                        choices=("subject", "diagnosis_site"),
                        help="Override T_CONTRASTIVE.OBJECTIVE")
    parser.add_argument("--min-sites-per-class", type=int, default=None,
                        help="Override T_CONTRASTIVE.MIN_SITES_PER_CLASS")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise SystemExit(f"No existe {ckpt_dir}")
    for name in ("best_pt_ts_fold_0.pt", "best_pt_fc_fold_0.pt"):
        if not (ckpt_dir / name).exists():
            raise SystemExit(f"Falta {ckpt_dir / name}")

    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["CKPT_TST1"] = str(ckpt_dir / "best_pt_ts_fold_0.pt")
    cfg["CKPT_TST2"] = str(ckpt_dir / "best_pt_fc_fold_0.pt")
    cfg["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["SEED"] = args.seed
    cfg["RUN_NAME"] = args.run_name
    if args.n_epochs is not None:
        cfg["T_CONTRASTIVE"]["N_EPOCHS"] = args.n_epochs
    if args.objective is not None:
        cfg["T_CONTRASTIVE"]["OBJECTIVE"] = args.objective
    if args.min_sites_per_class is not None:
        cfg["T_CONTRASTIVE"]["MIN_SITES_PER_CLASS"] = args.min_sites_per_class

    try:
        from utils.seed import set_seed
        set_seed(args.seed)
    except ImportError:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    out_root = Path(args.out_root)
    out_root.mkdir(exist_ok=True)
    exp_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    out_dir = out_root / exp_id
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["EXP_DIR"] = str(out_dir)
    cfg["CHECKPOINTS_PATH"] = str(out_dir / "checkpoints")
    cfg["EXP_ID"] = exp_id

    # Guardar la config efectivamente usada (con overrides).
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    shutil.copy("config/config.yaml", out_dir / "config_repo.yaml")

    print(f"Seed:        {args.seed}")
    print(f"Pretrain:    {ckpt_dir}")
    print(f"Objetivo:    {cfg['T_CONTRASTIVE'].get('OBJECTIVE', 'subject')}")
    print(f"Run name:    {args.run_name}")
    print(f"Output:      {out_dir}\n")

    metrics = run_contrastive_global(
        cfg, save_dir=out_dir / "checkpoints"
    )

    summary = {
        "exp_id": exp_id,
        "run_name": args.run_name,
        "seed": args.seed,
        "ckpt_pretrain_dir": str(ckpt_dir),
        "ckpt_contrastive": str(out_dir / "checkpoints" / "contrastive_global.pt"),
        **metrics,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    print(f"\n{'='*60}")
    print(f"RESUMEN CONTRASTIVE ({args.run_name}, seed={args.seed})")
    print(f"{'='*60}")
    print(f"  objetivo      : {summary['objective']}")
    print(f"  best epoch    : {summary['best_epoch']}")
    print(f"  best val loss : {summary['best_val_loss']:.4f}")
    print(f"  épocas        : {summary['n_epochs_run']}/{summary['n_epochs_requested']}")
    print(f"\nGuardado en: {out_dir / 'results.json'}")
    return summary


if __name__ == "__main__":
    main()
