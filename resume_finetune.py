"""Reanuda finetune desde folds que falten, reutilizando checkpoints ya entrenados.
No re-ejecuta pretrain ni contrastive."""
import yaml, json, glob, numpy as np, torch
from pathlib import Path

# Encontrar el último experimento con contrastive_global.pt
exp_candidates = sorted(glob.glob("experiments/*/checkpoints/contrastive_global.pt"))
if not exp_candidates:
    raise SystemExit("No se encontró ningún contraste_global.pt")
exp_ckpt_dir = Path(exp_candidates[-1]).parent
exp_dir = exp_ckpt_dir.parent
print(f"Reanudando desde: {exp_dir}")

with open(exp_dir / "config.yaml") as f:
    config = yaml.safe_load(f)

# Apuntar al directorio del experimento existente
config["EXP_DIR"] = str(exp_dir)
config["CHECKPOINTS_PATH"] = str(exp_ckpt_dir)
config["LOGS_PATH"] = str(exp_dir / "logs")

# Checkpoints existentes
config["CKPT_TST1"] = str(exp_ckpt_dir / "best_pt_ts_fold_0.pt")
config["CKPT_TST2"] = str(exp_ckpt_dir / "best_pt_fc_fold_0.pt")
config["CKPT_CONTRASTIVE"] = str(exp_ckpt_dir / "contrastive_global.pt")

for k in ("CKPT_TST1", "CKPT_TST2", "CKPT_CONTRASTIVE"):
    assert Path(config[k]).exists(), f"Falta {k}={config[k]}"
    print(f"  ✓ {k}")

torch.manual_seed(config["SEED"]); np.random.seed(config["SEED"])

from training.train_finetune import finetune_fold
from data.preprocessing.splitters import get_subject_level_fold_splits
from data.loaders.dataloader import load_raw_data
from utils.metrics import bootstrap_confidence_interval, get_reproducibility_info

# Determinar folds existentes
existing = sorted(exp_ckpt_dir.glob("best_finetune_fold_*.pt"))
done_folds = [int(p.stem.split("_")[-1]) for p in existing]
print(f"\nFolds ya hechos: {done_folds}")

# Reconstruir folds (splits)
data = load_raw_data(config)
folds = get_subject_level_fold_splits(
    data["labels"], data["subject_indices"], site_ids=data["site_ids"],
    n_splits=config.get("N_FOLDS", 5), val_ratio=0.15, seed=config["SEED"],
)
n_folds = len(folds)

# Cargar métricas existentes
fold_metrics = []
for f_idx in done_folds:
    blob = torch.load(exp_ckpt_dir / f"best_finetune_fold_{f_idx}.pt",
                       map_location="cpu", weights_only=False)
    m = blob["metrics"]
    m["fold_idx"] = f_idx
    fold_metrics.append(m)
    print(f"  fold {f_idx} (recuperado): AUC={m['auc']:.4f} ACC={m['accuracy']:.4f}")

# Correr folds faltantes
for f_idx in range(n_folds):
    if f_idx in done_folds:
        continue
    print(f"\n──── Fold {f_idx + 1}/{n_folds} (pendiente) ────")
    m = finetune_fold(config, fold_idx=f_idx, save_dir=exp_ckpt_dir)
    m["fold_idx"] = f_idx
    fold_metrics.append(m)

# Resumen
metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
summary = {
    "protocol": config.get("EVAL_PROTOCOL", "kfold"),
    "n_folds": n_folds, "seed": config["SEED"], "exp_id": exp_dir.name,
}
print(f"\n{'='*60}")
print(f"RESUMEN ({summary['protocol'].upper()}, mean ± std [95% CI])")
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

out = exp_dir / "results.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
print(f"\n✅ {out}")
