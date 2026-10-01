# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Multi-seed del TwoTST original (sin GRL, sin CDAN, sin weighted).

3 seeds x 19 folds = 57 finetunes.
Reutiliza los checkpoints de pretrain + contrastive del experimento
20260928_160330_transformer_baseline (el mismo que uso el run LOSO
del tst original).
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from training.train_finetune import finetune_fold
from utils.metrics import bootstrap_confidence_interval


CKPT_DIR = Path("experiments/20260928_160330_transformer_baseline/checkpoints")
if not CKPT_DIR.exists():
    raise SystemExit(f"No existe {CKPT_DIR}")

for f in ["best_pt_ts_fold_0.pt", "best_pt_fc_fold_0.pt", "contrastive_global.pt"]:
    if not (CKPT_DIR / f).exists():
        raise SystemExit(f"Falta {CKPT_DIR / f}")

with open("config/config.yaml", encoding="utf-8") as f:
    base_config = yaml.safe_load(f)

base_config["CKPT_TST1"] = str(CKPT_DIR / "best_pt_ts_fold_0.pt")
base_config["CKPT_TST2"] = str(CKPT_DIR / "best_pt_fc_fold_0.pt")
base_config["CKPT_CONTRASTIVE"] = str(CKPT_DIR / "contrastive_global.pt")
base_config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
base_config["EVAL_PROTOCOL"] = "loso"

# Limpiar todo lo de GRL/CDAN/weighted (TwoTST original puro)
fin = base_config.get("FINETUNING", {})
for k in ["DOMAIN_CLASSIFIER_TYPE", "GRL_LOCATIONS", "GRL_LAMBDA", "GRL_SCHEDULE",
          "GRL_GAMMA", "GRL_WARMUP", "DOMAIN_WEIGHT", "GRL_MULTILAYER",
          "GRL_STREAM_HIDDEN_DIMS", "USE_SITE_CLASS_WEIGHTS",
          "SITE_CLASS_WEIGHT_POWER", "SITE_CLASS_WEIGHT_NORMALIZE",
          "CDAN_ENTROPY_WEIGHT", "CDAN_LINEAR_REDUCE"]:
    fin.pop(k, None)


SEEDS = [42, 123, 2024]

out_root = Path("experiments_multiseed_twotst")
out_root.mkdir(exist_ok=True)

all_summaries = []

for seed in SEEDS:
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")

    cfg = copy.deepcopy(base_config)
    cfg["SEED"] = seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg_dir = out_root / f"seed_{seed}"
    (cfg_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["CHECKPOINTS_PATH"] = str(cfg_dir / "checkpoints")

    from data.loaders.dataloader import load_raw_data
    data = load_raw_data(cfg)
    n_folds = len(np.unique(data["site_ids"]))
    print(f"  LOSO folds: {n_folds}")

    fold_metrics = []
    for fold_idx in range(n_folds):
        preds_path = Path(cfg["CHECKPOINTS_PATH"]) / f"preds_fold_{fold_idx}.npz"
        if preds_path.exists():
            print(f"  [SKIP] seed={seed} fold={fold_idx} (ya existe)")
            continue

        torch.manual_seed(seed + fold_idx)
        m = finetune_fold(cfg, fold_idx=fold_idx,
                          save_dir=Path(cfg["CHECKPOINTS_PATH"]))
        m["fold_idx"] = fold_idx
        fold_metrics.append(m)

    # Resumen
    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {"seed": seed, "n_folds": n_folds}
    for metric in metric_names:
        vals = []
        for f in sorted(Path(cfg["CHECKPOINTS_PATH"]).glob("preds_fold_*.npz")):
            npz = np.load(f, allow_pickle=True)
            y, p = npz['labels'], npz['probs']
            if len(np.unique(y)) < 2:
                continue
            if metric == "auc":
                from sklearn.metrics import roc_auc_score
                vals.append(roc_auc_score(y, p))
            elif metric == "accuracy":
                vals.append(float(((p >= 0.5).astype(int) == y).mean()))
            elif metric == "sensitivity":
                yp = (p >= 0.5).astype(int)
                tp = ((yp==1)&(y==1)).sum(); fn = ((yp==0)&(y==1)).sum()
                vals.append(float(tp/(tp+fn)) if (tp+fn) > 0 else 0.0)
            elif metric == "specificity":
                yp = (p >= 0.5).astype(int)
                tn = ((yp==0)&(y==0)).sum(); fp = ((yp==1)&(y==0)).sum()
                vals.append(float(tn/(tn+fp)) if (tn+fp) > 0 else 0.0)
            elif metric == "f1":
                yp = (p >= 0.5).astype(int)
                tp = ((yp==1)&(y==1)).sum(); fp = ((yp==1)&(y==0)).sum(); fn = ((yp==0)&(y==1)).sum()
                prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
                rec = tp/(tp+fn) if (tp+fn) > 0 else 0.0
                vals.append(2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0)
        mean_v, std_v, lo, hi = bootstrap_confidence_interval(
            vals, n_bootstrap=1000, ci=0.95, seed=seed)
        summary[metric] = {"mean": mean_v, "std": std_v, "lo": lo, "hi": hi}

    with open(cfg_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    all_summaries.append(summary)

    print(f"  AUC: {summary['auc']['mean']:.4f} +/- {summary['auc']['std']:.4f}")

# Agregado
print(f"\n{'='*70}")
print("AGREGADO ENTRE SEEDS")
print(f"{'='*70}")
aucs = [s["auc"]["mean"] for s in all_summaries]
print(f"AUC medio: {np.mean(aucs):.4f}")
print(f"AUC std:   {np.std(aucs):.4f}")
print(f"Rango:     [{min(aucs):.4f}, {max(aucs):.4f}]")
for s in all_summaries:
    print(f"  seed {s['seed']}: {s['auc']['mean']:.4f} +/- {s['auc']['std']:.4f}")

with open(out_root / "all_summaries.json", "w") as f:
    json.dump(all_summaries, f, indent=2, default=float)
