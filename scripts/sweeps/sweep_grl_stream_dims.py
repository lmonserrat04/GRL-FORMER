"""
Sweep: comparar capacidad de los domain classifiers de los streams en LOSO.
Configs:
  1. multi_stream_full   (GRL_STREAM_HIDDEN_DIMS = null)
  2. multi_stream_weak64 (GRL_STREAM_HIDDEN_DIMS = [64])
  3. multi_stream_none   (GRL_STREAM_HIDDEN_DIMS = [])
Reutiliza checkpoints pretrain + contrastive.
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

# Hiperparametros optimos (ganadores de sweeps previos)
base_config["FINETUNING"]["GRL_SCHEDULE"] = True
base_config["FINETUNING"]["GRL_GAMMA"] = 3.0
base_config["FINETUNING"]["DOMAIN_WEIGHT"] = 1.0
base_config["MLP_HEAD"]["DROPOUT"] = 0

base_config["CKPT_TST1"] = str(src / "checkpoints" / "best_pt_ts_fold_0.pt")
base_config["CKPT_TST2"] = str(src / "checkpoints" / "best_pt_fc_fold_0.pt")
base_config["CKPT_CONTRASTIVE"] = str(src / "checkpoints" / "contrastive_global.pt")
base_config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
base_config["EVAL_PROTOCOL"] = "loso"


CONFIGS = [
    {"name": "multi_stream_full",   "multilayer": True, "stream_dims": None},
    {"name": "multi_stream_weak64", "multilayer": True, "stream_dims": [64]},
    {"name": "multi_stream_none",   "multilayer": True, "stream_dims": []},
]


out_root = Path("experiments_sweep_stream_dims")
out_root.mkdir(exist_ok=True)


def run_config(spec):
    name = spec["name"]
    print(f"\n{'='*70}")
    print(f"CONFIG: {name}")
    print(f"{'='*70}")

    cfg = copy.deepcopy(base_config)
    cfg["FINETUNING"]["GRL_MULTILAYER"] = spec["multilayer"]
    cfg["FINETUNING"]["GRL_STREAM_HIDDEN_DIMS"] = spec["stream_dims"]

    cfg_dir = out_root / name
    (cfg_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["CHECKPOINTS_PATH"] = str(cfg_dir / "checkpoints")

    from data.loaders.dataloader import load_raw_data
    data = load_raw_data(cfg)
    n_folds = len(np.unique(data["site_ids"]))
    print(f"  LOSO folds (sitios unicos): {n_folds}")

    fold_metrics = []
    for fold_idx in range(n_folds):
        m = finetune_fold(cfg, fold_idx=fold_idx,
                          save_dir=Path(cfg["CHECKPOINTS_PATH"]))
        m["fold_idx"] = fold_idx
        fold_metrics.append(m)

    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {"config": name, "n_folds": n_folds,
               "multilayer": spec["multilayer"],
               "stream_dims": spec["stream_dims"]}
    for metric in metric_names:
        vals = [m[metric] for m in fold_metrics]
        mean_v, std_v, lo, hi = bootstrap_confidence_interval(
            vals, n_bootstrap=1000, ci=0.95, seed=cfg["SEED"]
        )
        summary[metric] = {"mean": mean_v, "std": std_v, "lo": lo, "hi": hi}

    with open(cfg_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  RESUMEN {name}:")
    for metric in metric_names:
        s = summary[metric]
        print(f"    {metric:12s}: {s['mean']:.4f} +/- {s['std']:.4f} "
              f"[{s['lo']:.4f}, {s['hi']:.4f}]")

    return summary


all_summaries = []
for spec in CONFIGS:
    try:
        s = run_config(spec)
        all_summaries.append(s)
    except Exception as e:
        print(f"\n[ERROR] {spec['name']}: {e}")
        all_summaries.append({"config": spec["name"], "error": str(e)})


print("\n" + "="*100)
print("COMPARATIVA FINAL (LOSO)")
print("="*100)
print(f"{'config':<28s}  {'AUC':<18s}  {'Sens':<18s}  {'Spec':<18s}")
print("-"*100)
for s in all_summaries:
    if "error" in s:
        print(f"{s['config']:<28s}  ERROR: {s['error'][:60]}")
        continue
    auc = s["auc"]; sens = s["sensitivity"]; spec = s["specificity"]
    print(f"{s['config']:<28s}  "
          f"{auc['mean']:.4f}+/-{auc['std']:.4f}  "
          f"{sens['mean']:.4f}+/-{sens['std']:.4f}  "
          f"{spec['mean']:.4f}+/-{spec['std']:.4f}")

with open(out_root / "all_summaries.json", "w") as f:
    json.dump(all_summaries, f, indent=2)
print(f"\nGuardado en: {out_root / 'all_summaries.json'}")
