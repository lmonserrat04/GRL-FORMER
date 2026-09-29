"""
Sweep: dropout=0.3, K-fold vs LOSO, no_grl vs grl_g3_w1.0.
Reutiliza pretrain + contrastive de un experimento previo.
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


# ─── Localizar checkpoints ──────────────────────────────────────────
exp_dirs = sorted(glob.glob("experiments/*_transformer_baseline"))
src = None
for d in reversed(exp_dirs):
    if (Path(d) / "checkpoints" / "contrastive_global.pt").exists():
        src = Path(d)
        break
if src is None:
    raise SystemExit("No hay experimento con contrastive_global.pt")

print(f"Reutilizando checkpoints de: {src}")

with open(src / "config.yaml") as f:
    base_config = yaml.safe_load(f)

base_config["CKPT_TST1"] = str(src / "checkpoints" / "best_pt_ts_fold_0.pt")
base_config["CKPT_TST2"] = str(src / "checkpoints" / "best_pt_fc_fold_0.pt")
base_config["CKPT_CONTRASTIVE"] = str(src / "checkpoints" / "contrastive_global.pt")
base_config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

# Forzar dropout 0.3
base_config["MLP_HEAD"]["DROPOUT"] = 0.3


# ─── Configuraciones a correr ───────────────────────────────────────
CONFIGS = [
    {"name": "kfold_no_grl_dp0.3",     "protocol": "kfold", "use_grl": False},
    {"name": "kfold_grl_g3_w1_dp0.3",  "protocol": "kfold", "use_grl": True, "gamma": 3.0, "w": 1.0},
    {"name": "loso_no_grl_dp0.3",      "protocol": "loso",  "use_grl": False},
    {"name": "loso_grl_g3_w1_dp0.3",   "protocol": "loso",  "use_grl": True, "gamma": 3.0, "w": 1.0},
]


out_root = Path("experiments_sweep_dropout_loso")
out_root.mkdir(exist_ok=True)


def run_config(spec):
    name = spec["name"]
    print(f"\n{'='*70}")
    print(f"CONFIG: {name}")
    print(f"{'='*70}")

    cfg = copy.deepcopy(base_config)
    cfg["EVAL_PROTOCOL"] = spec["protocol"]
    cfg["N_FOLDS"] = 5    # solo se usa si protocol=kfold

    if spec["use_grl"]:
        cfg["FINETUNING"]["GRL_SCHEDULE"] = True
        cfg["FINETUNING"]["GRL_LAMBDA"] = 1.0
        cfg["FINETUNING"]["GRL_GAMMA"] = spec["gamma"]
        cfg["FINETUNING"]["DOMAIN_WEIGHT"] = spec["w"]
    else:
        cfg["FINETUNING"]["GRL_SCHEDULE"] = False
        cfg["FINETUNING"]["GRL_LAMBDA"] = 0.0
        cfg["FINETUNING"]["DOMAIN_WEIGHT"] = 0.0

    cfg_dir = out_root / name
    (cfg_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["CHECKPOINTS_PATH"] = str(cfg_dir / "checkpoints")

    # Determinar n_folds según protocolo
    from data.loaders.dataloader import load_raw_data
    data = load_raw_data(cfg)
    if spec["protocol"] == "loso":
        n_folds = len(np.unique(data["site_ids"]))
    else:
        n_folds = 5
    print(f"  protocolo={spec['protocol']}, folds={n_folds}")

    fold_metrics = []
    for fold_idx in range(n_folds):
        m = finetune_fold(cfg, fold_idx=fold_idx,
                          save_dir=Path(cfg["CHECKPOINTS_PATH"]))
        m["fold_idx"] = fold_idx
        fold_metrics.append(m)

    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {"config": name, "protocol": spec["protocol"], "n_folds": n_folds,
               "use_grl": spec["use_grl"]}
    for metric in metric_names:
        vals = [m[metric] for m in fold_metrics]
        mean_v, std_v, lo, hi = bootstrap_confidence_interval(
            vals, n_bootstrap=1000, ci=0.95, seed=cfg["SEED"]
        )
        summary[metric] = {"mean": mean_v, "std": std_v, "lo": lo, "hi": hi}

    with open(cfg_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


all_summaries = []
for spec in CONFIGS:
    try:
        s = run_config(spec)
        all_summaries.append(s)
    except Exception as e:
        print(f"\n[ERROR] {spec['name']}: {e}")
        all_summaries.append({"config": spec["name"], "error": str(e)})


# ─── Tabla comparativa ──────────────────────────────────────────────
print("\n" + "="*100)
print("COMPARATIVA FINAL")
print("="*100)
print(f"{'config':<30s}  {'AUC':<18s}  {'Sens':<18s}  {'Spec':<18s}")
print("-"*100)
for s in all_summaries:
    if "error" in s:
        print(f"{s['config']:<30s}  ERROR: {s['error'][:60]}")
        continue
    auc = s["auc"]; sens = s["sensitivity"]; spec = s["specificity"]
    print(f"{s['config']:<30s}  "
          f"{auc['mean']:.4f}±{auc['std']:.4f}  "
          f"{sens['mean']:.4f}±{sens['std']:.4f}  "
          f"{spec['mean']:.4f}±{spec['std']:.4f}")

with open(out_root / "all_summaries.json", "w") as f:
    json.dump(all_summaries, f, indent=2)
print(f"\nGuardado en: {out_root / 'all_summaries.json'}")
