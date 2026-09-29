"""
Sweep comparativo: single-layer vs multi-layer GRL en LOSO.

Ambas configs usan γ=1, w=0.1, warmup=10 (config C ganadora).
La única diferencia es GRL_MULTILAYER=True/False.

Uso:
    python sweep_multilayer.py
"""
# --- sys.path bootstrap (scripts movidos a subcarpetas) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

import copy
import glob
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from training.train_finetune import finetune_fold
from utils.metrics import bootstrap_confidence_interval


# ─── Localizar checkpoints base ─────────────────────────────────────
exp_dirs = sorted(glob.glob("experiments/*_transformer_baseline"))
src = None
for d in reversed(exp_dirs):
    if (Path(d) / "checkpoints" / "contrastive_global.pt").exists():
        src = Path(d); break
if src is None:
    raise SystemExit("No hay experimento base")

print(f"Reutilizando checkpoints de: {src}")

with open(src / "config.yaml") as f:
    base_config = yaml.safe_load(f)

base_config["CKPT_TST1"] = str(src / "checkpoints" / "best_pt_ts_fold_0.pt")
base_config["CKPT_TST2"] = str(src / "checkpoints" / "best_pt_fc_fold_0.pt")
base_config["CKPT_CONTRASTIVE"] = str(src / "checkpoints" / "contrastive_global.pt")
base_config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
base_config["EVAL_PROTOCOL"] = "loso"


# ─── Dos configs a comparar ────────────────────────────────────────
CONFIGS = [
    {"name": "multi_C",  "multilayer": True},
]


out_root = Path("experiments_sweep_multilayer")
out_root.mkdir(exist_ok=True)


def run_config(spec):
    name = spec["name"]
    print(f"\n{'='*70}")
    print(f"CONFIG: {name}   (multilayer={spec['multilayer']})")
    print(f"{'='*70}")

    cfg = copy.deepcopy(base_config)

    # Config C: γ=1, w=0.1, warmup=10
    cfg["FINETUNING"]["GRL_SCHEDULE"] = True
    cfg["FINETUNING"]["GRL_LAMBDA"] = 1.0
    cfg["FINETUNING"]["GRL_GAMMA"] = 1.0
    cfg["FINETUNING"]["GRL_WARMUP"] = 10
    cfg["FINETUNING"]["DOMAIN_WEIGHT"] = 0.1
    cfg["FINETUNING"]["GRL_MULTILAYER"] = spec["multilayer"]

    cfg_dir = out_root / name
    (cfg_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["CHECKPOINTS_PATH"] = str(cfg_dir / "checkpoints")

    from data.loaders.dataloader import load_raw_data
    data = load_raw_data(cfg)
    n_folds = len(np.unique(data["site_ids"]))
    print(f"  folds = {n_folds}")

    fold_metrics = []
    for fold_idx in range(n_folds):
        m = finetune_fold(cfg, fold_idx=fold_idx,
                          save_dir=Path(cfg["CHECKPOINTS_PATH"]))
        fold_metrics.append(m)

    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {"config": name, "multilayer": spec["multilayer"],
               "n_folds": n_folds}
    for metric in metric_names:
        vals = [m[metric] for m in fold_metrics]
        mean_v, std_v, lo, hi = bootstrap_confidence_interval(
            vals, n_bootstrap=1000, ci=0.95, seed=cfg["SEED"])
        summary[metric] = {"mean": mean_v, "std": std_v, "lo": lo, "hi": hi}

    with open(cfg_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


all_summaries = []
for spec in CONFIGS:
    s = run_config(spec)
    all_summaries.append(s)

# ─── Reporte ────────────────────────────────────────────────────────
print("\n" + "="*80)
print("COMPARATIVA single vs multi (LOSO, config C)")
print("="*80)
print(f"{'config':<15s}  {'AUC':<18s}  {'Sens':<18s}  {'Spec':<18s}")
print("-"*80)
for s in all_summaries:
    auc = s["auc"]; sens = s["sensitivity"]; spec = s["specificity"]
    print(f"{s['config']:<15s}  "
          f"{auc['mean']:.4f}±{auc['std']:.4f}  "
          f"{sens['mean']:.4f}±{sens['std']:.4f}  "
          f"{spec['mean']:.4f}±{spec['std']:.4f}")

with open(out_root / "all_summaries.json", "w") as f:
    json.dump(all_summaries, f, indent=2)
print(f"\n[OK] {out_root / 'all_summaries.json'}")
