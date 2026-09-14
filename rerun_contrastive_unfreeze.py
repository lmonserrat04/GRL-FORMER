"""Reutiliza pretrains del último experimento, re-corre contrastive (unfreeze both)
y 5-fold finetune. NO re-ejecuta pretrain."""
import yaml, json, glob, numpy as np, torch, shutil
from pathlib import Path
from datetime import datetime

# Localizar pretrains del último experimento
candidates = sorted(glob.glob("experiments/*/checkpoints/best_pt_ts_fold_0.pt"))
if not candidates:
    raise SystemExit("No hay pretrains previos")
src_ckpt = Path(candidates[-1]).parent
src_exp = src_ckpt.parent
print(f"Pretrains de: {src_exp}")

with open(src_exp / "config.yaml") as f:
    config = yaml.safe_load(f)

# Crear nuevo directorio de experimento (para no sobrescribir el anterior)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
new_dir = Path(f"experiments/{ts}_transformer_baseline_unfreeze_both")
(new_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
(new_dir / "logs").mkdir(parents=True, exist_ok=True)
shutil.copy(src_exp / "config.yaml", new_dir / "config.yaml")

config["EXP_DIR"] = str(new_dir)
config["CHECKPOINTS_PATH"] = str(new_dir / "checkpoints")
config["LOGS_PATH"] = str(new_dir / "logs")
config["EXP_ID"] = new_dir.name

# Reutilizar pretrains del experimento anterior (copiándolos al nuevo dir)
for name in ["best_pt_ts_fold_0.pt", "best_pt_fc_fold_0.pt"]:
    shutil.copy(src_ckpt / name, new_dir / "checkpoints" / name)

config["CKPT_TST1"] = str(new_dir / "checkpoints" / "best_pt_ts_fold_0.pt")
config["CKPT_TST2"] = str(new_dir / "checkpoints" / "best_pt_fc_fold_0.pt")

for k in ("CKPT_TST1", "CKPT_TST2"):
    assert Path(config[k]).exists(), f"Falta {k}"
    print(f"  ✓ {k}")

torch.manual_seed(config["SEED"]); np.random.seed(config["SEED"])

from training.train_contrastive import run_contrastive_global
from training.train_finetune import finetune_fold
from data.preprocessing.splitters import get_subject_level_fold_splits
from data.loaders.dataloader import load_raw_data
from utils.metrics import bootstrap_confidence_interval, get_reproducibility_info

print(f"\n>>> CONTRASTIVE (unfreeze both)")
ckpt_dir = Path(config["CHECKPOINTS_PATH"])
run_contrastive_global(config, save_dir=ckpt_dir)
config["CKPT_CONTRASTIVE"] = str(ckpt_dir / "contrastive_global.pt")

# Folds
data = load_raw_data(config)
folds = get_subject_level_fold_splits(
    data["labels"], data["subject_indices"], site_ids=data["site_ids"],
    n_splits=config.get("N_FOLDS", 5), val_ratio=0.15, seed=config["SEED"],
)
n_folds = len(folds)

print(f"\n>>> KFOLD — {n_folds} folds")
fold_metrics = []
for f_idx in range(n_folds):
    print(f"\n──── Fold {f_idx + 1}/{n_folds} ────")
    m = finetune_fold(config, fold_idx=f_idx, save_dir=ckpt_dir)
    m["fold_idx"] = f_idx
    fold_metrics.append(m)

# Resumen
metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
summary = {
    "protocol": config.get("EVAL_PROTOCOL", "kfold"),
    "n_folds": n_folds, "seed": config["SEED"], "exp_id": new_dir.name,
    "contrastive_strategy": "unfreeze_both",
}
print(f"\n{'='*60}")
print(f"RESUMEN — unfreeze both (mean ± std [95% CI])")
print(f"{'='*60}")
for name in metric_names:
    vals = [m[name] for m in fold_metrics if name in m]
    if not vals:
        continue
    mean_v, std_v, lo, hi = bootstrap_confidence_interval(vals, n_bootstrap=1000, ci=0.95, seed=config["SEED"])
    summary[name] = {"mean": mean_v, "std": std_v, "ci95_lower": lo, "ci95_upper": hi}
    label = name.upper() if name == "auc" else name.capitalize()
    print(f"  {label:12s}: {mean_v:.4f} ± {std_v:.4f}  [{lo:.4f}, {hi:.4f}]")

summary["all_folds"] = fold_metrics
summary["reproducibility"] = get_reproducibility_info()

with open(new_dir / "results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
print(f"\n✅ {new_dir / 'results.json'}")
