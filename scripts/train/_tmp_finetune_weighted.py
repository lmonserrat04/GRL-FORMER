# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Corre SOLO la fase de finetune con ponderacion (sitio, clase).

Reutiliza los checkpoints de pretrain + contrastive de un experimento
previo (DANN, feat/grl-locations). Los busca en experiments/.

Uso:
    python scripts/train/_tmp_finetune_weighted.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from training.train_finetune import finetune_fold
from utils.metrics import bootstrap_confidence_interval


# ─── 1. Localizar checkpoints de pretrain + contrastive ────────────
# Busca cualquier experimento que tenga los 3 checkpoints.
# Prefiere el más reciente.
candidates = []
for d in sorted(glob.glob("experiments/*"), reverse=True):
    ckpt = Path(d) / "checkpoints"
    if not ckpt.exists():
        continue
    if all((ckpt / f).exists() for f in [
        "best_pt_ts_fold_0.pt",
        "best_pt_fc_fold_0.pt",
        "contrastive_global.pt",
    ]):
        candidates.append(Path(d))

if not candidates:
    raise SystemExit(
        "No hay ningún experimento con best_pt_ts_fold_0.pt, "
        "best_pt_fc_fold_0.pt y contrastive_global.pt"
    )

src = candidates[0]
print(f"Reutilizando checkpoints de: {src}")

# ─── 2. Cargar config y fijar los CKPT_* ───────────────────────────
with open("config/config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["CKPT_TST1"] = str(src / "checkpoints" / "best_pt_ts_fold_0.pt")
config["CKPT_TST2"] = str(src / "checkpoints" / "best_pt_fc_fold_0.pt")
config["CKPT_CONTRASTIVE"] = str(src / "checkpoints" / "contrastive_global.pt")
config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 3. Directorio de salida del run ───────────────────────────────
from datetime import datetime
exp_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['RUN_NAME']}"
out_dir = Path("experiments") / exp_id
(out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

config["EXP_DIR"] = str(out_dir)
config["CHECKPOINTS_PATH"] = str(out_dir / "checkpoints")
config["EXP_ID"] = exp_id

# Copiar config para trazabilidad
import shutil
shutil.copy("config/config.yaml", out_dir / "config.yaml")

print(f"Salida en: {out_dir}")
print(f"EVAL_PROTOCOL: {config.get('EVAL_PROTOCOL')}")
print(f"GRL_LOCATIONS: {config['FINETUNING'].get('GRL_LOCATIONS')}")
print(f"DOMAIN_WEIGHT: {config['FINETUNING'].get('DOMAIN_WEIGHT')}")
print(f"USE_SITE_CLASS_WEIGHTS: {config['FINETUNING'].get('USE_SITE_CLASS_WEIGHTS')}")
print()

# ─── 4. Determinar cuántos folds corre ─────────────────────────────
from data.loaders.dataloader import load_raw_data
data = load_raw_data(config)
protocol = config.get("EVAL_PROTOCOL", "kfold")
if protocol == "loso":
    n_folds = len(np.unique(data["site_ids"]))
else:
    n_folds = config.get("N_FOLDS", 5)
print(f"Protocolo: {protocol}  →  {n_folds} folds")
print()

# ─── 5. Loop de finetune ───────────────────────────────────────────
fold_metrics = []
for fold_idx in range(n_folds):
    print(f"\n─── Fold {fold_idx+1}/{n_folds} ───")
    m = finetune_fold(
        config,
        fold_idx=fold_idx,
        save_dir=Path(config["CHECKPOINTS_PATH"]),
    )
    m["fold_idx"] = fold_idx
    fold_metrics.append(m)

# ─── 6. Resumen ────────────────────────────────────────────────────
metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
summary = {
    "protocol": protocol,
    "n_folds": n_folds,
    "seed": config["SEED"],
    "exp_id": exp_id,
    "grl_locations": config["FINETUNING"].get("GRL_LOCATIONS"),
    "domain_weight": config["FINETUNING"].get("DOMAIN_WEIGHT"),
    "use_site_class_weights": config["FINETUNING"].get("USE_SITE_CLASS_WEIGHTS", False),
}
for metric in metric_names:
    vals = [m[metric] for m in fold_metrics]
    mean_v, std_v, lo, hi = bootstrap_confidence_interval(
        vals, n_bootstrap=1000, ci=0.95, seed=config["SEED"]
    )
    summary[metric] = {"mean": mean_v, "std": std_v,
                       "ci95_lower": lo, "ci95_upper": hi}

with open(out_dir / "results.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

print()
print("=" * 60)
print(f"RESUMEN ({protocol.upper()}, weighted site-class)")
print("=" * 60)
for metric in metric_names:
    s = summary[metric]
    print(f"  {metric:<12s}: {s['mean']:.4f} +/- {s['std']:.4f}  "
          f"[{s['ci95_lower']:.4f}, {s['ci95_upper']:.4f}]")

print(f"\nGuardado en: {out_dir / 'results.json'}")
