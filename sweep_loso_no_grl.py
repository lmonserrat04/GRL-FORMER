"""Re-corre LOSO sin GRL con guardado de predicciones."""
import copy, glob, json
from pathlib import Path
import numpy as np, torch, yaml
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

# ─── Sin GRL ────────────────────────────────────────────────────────
cfg = copy.deepcopy(base_config)
cfg["FINETUNING"]["GRL_SCHEDULE"] = False
cfg["FINETUNING"]["GRL_LAMBDA"] = 0.0
cfg["FINETUNING"]["DOMAIN_WEIGHT"] = 0.0

cfg_dir = Path("experiments_loso_no_grl") / "no_grl_loso"
(cfg_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
cfg["CHECKPOINTS_PATH"] = str(cfg_dir / "checkpoints")

from data.loaders.dataloader import load_raw_data
data = load_raw_data(cfg)
n_folds = len(np.unique(data["site_ids"]))
print(f"folds = {n_folds}")

fold_metrics = []
for fold_idx in range(n_folds):
    print(f"\n  Fold {fold_idx+1}/{n_folds}")
    m = finetune_fold(cfg, fold_idx=fold_idx,
                      save_dir=Path(cfg["CHECKPOINTS_PATH"]))
    fold_metrics.append(m)

metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
summary = {"config": "no_grl_loso", "n_folds": n_folds}
for metric in metric_names:
    vals = [m[metric] for m in fold_metrics]
    mean_v, std_v, lo, hi = bootstrap_confidence_interval(
        vals, n_bootstrap=1000, ci=0.95, seed=cfg["SEED"])
    summary[metric] = {"mean": mean_v, "std": std_v, "lo": lo, "hi": hi}

with open(cfg_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[OK] {cfg_dir / 'summary.json'}")
