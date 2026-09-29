"""
Sweep GRL tuning: 4 configs (γ × w × warmup).
Reutiliza pretrain + contrastive. Solo re-corre finetune K-fold.
Guarda preds para analyze_predictions.py.
"""
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
base_config["N_FOLDS"] = 5
base_config["EVAL_PROTOCOL"] = "kfold"


# ─── 4 configs a probar ─────────────────────────────────────────────
CONFIGS = [
    {"name": "A_g3_w0.1_warm0",  "gamma": 3.0, "w": 0.1, "warmup": 0},
    {"name": "B_g1_w0.1_warm0",  "gamma": 1.0, "w": 0.1, "warmup": 0},
    {"name": "C_g1_w0.1_warm10", "gamma": 1.0, "w": 0.1, "warmup": 10},
    {"name": "D_g3_w0.1_warm10", "gamma": 3.0, "w": 0.1, "warmup": 10},
]


out_root = Path("experiments_sweep_grl_tuning")
out_root.mkdir(exist_ok=True)


def run_config(spec):
    name = spec["name"]
    print(f"\n{'='*70}")
    print(f"CONFIG: {name}")
    print(f"  gamma={spec['gamma']}  w={spec['w']}  warmup={spec['warmup']}")
    print(f"{'='*70}")

    cfg = copy.deepcopy(base_config)
    cfg["FINETUNING"]["GRL_SCHEDULE"] = True
    cfg["FINETUNING"]["GRL_LAMBDA"] = 1.0
    cfg["FINETUNING"]["GRL_GAMMA"] = spec["gamma"]
    cfg["FINETUNING"]["GRL_WARMUP"] = spec["warmup"]
    cfg["FINETUNING"]["DOMAIN_WEIGHT"] = spec["w"]

    cfg_dir = out_root / name
    (cfg_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["CHECKPOINTS_PATH"] = str(cfg_dir / "checkpoints")

    fold_metrics = []
    for fold_idx in range(cfg["N_FOLDS"]):
        print(f"\n  Fold {fold_idx+1}/{cfg['N_FOLDS']}")
        m = finetune_fold(cfg, fold_idx=fold_idx,
                          save_dir=Path(cfg["CHECKPOINTS_PATH"]))
        fold_metrics.append(m)

    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {"config": name, "n_folds": cfg["N_FOLDS"],
               "gamma": spec["gamma"], "w": spec["w"], "warmup": spec["warmup"]}
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
    s = run_config(spec)
    all_summaries.append(s)


print("\n" + "="*100)
print("COMPARATIVA FINAL")
print("="*100)
print(f"{'config':<25s}  {'AUC':<18s}  {'Sens':<18s}  {'Spec':<18s}")
print("-"*100)
for s in all_summaries:
    auc = s["auc"]; sens = s["sensitivity"]; spec = s["specificity"]
    print(f"{s['config']:<25s}  "
          f"{auc['mean']:.4f}±{auc['std']:.4f}  "
          f"{sens['mean']:.4f}±{sens['std']:.4f}  "
          f"{spec['mean']:.4f}±{spec['std']:.4f}")

with open(out_root / "all_summaries.json", "w") as f:
    json.dump(all_summaries, f, indent=2)
print(f"\nGuardado en: {out_root / 'all_summaries.json'}")
