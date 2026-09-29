"""
Sweep de hiperparámetros GRL reutilizando checkpoints guardados.
Solo re-corre FASE 4 (finetune) con distintas configuraciones.
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


# ─── Localizar experimento con checkpoints ──────────────────────────
exp_dirs = sorted(glob.glob("experiments/*_transformer_baseline"))
if not exp_dirs:
    raise SystemExit("No hay experimentos previos")

# Preferir uno con contrastive_global.pt
src = None
for d in reversed(exp_dirs):
    if (Path(d) / "checkpoints" / "contrastive_global.pt").exists():
        src = Path(d)
        break
if src is None:
    raise SystemExit("Ningún experimento tiene contrastive_global.pt")

print(f"Reutilizando checkpoints de: {src}")

with open(src / "config.yaml") as f:
    base_config = yaml.safe_load(f)

base_config["CKPT_TST1"] = str(src / "checkpoints" / "best_pt_ts_fold_0.pt")
base_config["CKPT_TST2"] = str(src / "checkpoints" / "best_pt_fc_fold_0.pt")
base_config["CKPT_CONTRASTIVE"] = str(src / "checkpoints" / "contrastive_global.pt")
base_config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
base_config["N_FOLDS"] = 5
base_config["EVAL_PROTOCOL"] = "kfold"


# ─── Configuraciones a probar ───────────────────────────────────────
CONFIGS = [
    {"name": "no_grl",          "use_grl": False, "gamma": None, "w": None},
    {"name": "grl_g10_w1.0",    "use_grl": True,  "gamma": 10.0, "w": 1.0},
    {"name": "grl_g3_w1.0",     "use_grl": True,  "gamma": 3.0,  "w": 1.0},
    {"name": "grl_g1_w1.0",     "use_grl": True,  "gamma": 1.0,  "w": 1.0},
    {"name": "grl_g3_w0.1",     "use_grl": True,  "gamma": 3.0,  "w": 0.1},
]


# ─── Directorio de salida ───────────────────────────────────────────
out_root = Path("experiments_sweep_grl")
out_root.mkdir(exist_ok=True)


# ─── Función que corre un fold ──────────────────────────────────────
def run_config(cfg_spec):
    name = cfg_spec["name"]
    print(f"\n{'='*70}")
    print(f"CONFIG: {name}")
    print(f"{'='*70}")

    cfg = copy.deepcopy(base_config)

    # Configurar finetune
    if not cfg_spec["use_grl"]:
        # Sin GRL: dominio no participa
        cfg["FINETUNING"]["GRL_SCHEDULE"] = False
        cfg["FINETUNING"]["GRL_LAMBDA"] = 0.0   # GRL anula el domain_loss
    else:
        cfg["FINETUNING"]["GRL_SCHEDULE"] = True
        cfg["FINETUNING"]["GRL_GAMMA"] = cfg_spec["gamma"]
        cfg["FINETUNING"]["DOMAIN_WEIGHT"] = cfg_spec["w"]

    # Directorio aislado por config
    cfg_dir = out_root / name
    (cfg_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["CHECKPOINTS_PATH"] = str(cfg_dir / "checkpoints")

    fold_metrics = []
    for fold_idx in range(cfg["N_FOLDS"]):
        print(f"\n  Fold {fold_idx+1}/{cfg['N_FOLDS']}")
        m = finetune_fold(cfg, fold_idx=fold_idx,
                          save_dir=Path(cfg["CHECKPOINTS_PATH"]))
        fold_metrics.append(m)

    # Resumen
    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {"config": name, "n_folds": cfg["N_FOLDS"]}
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
        print(f"    {metric:12s}: {s['mean']:.4f} ± {s['std']:.4f} "
              f"[{s['lo']:.4f}, {s['hi']:.4f}]")

    return summary


# ─── Ejecutar todos ─────────────────────────────────────────────────
all_summaries = []
for spec in CONFIGS:
    try:
        s = run_config(spec)
        all_summaries.append(s)
    except Exception as e:
        print(f"\n[ERROR] Config {spec['name']} falló: {e}")
        all_summaries.append({"config": spec["name"], "error": str(e)})


# ─── Tabla comparativa final ────────────────────────────────────────
print("\n" + "="*80)
print("COMPARATIVA FINAL")
print("="*80)
print(f"{'config':<20s}  {'AUC':<18s}  {'Sens':<18s}  {'Spec':<18s}")
print("-"*80)
for s in all_summaries:
    if "error" in s:
        print(f"{s['config']:<20s}  ERROR: {s['error'][:50]}")
        continue
    auc = s["auc"]; sens = s["sensitivity"]; spec = s["specificity"]
    print(f"{s['config']:<20s}  "
          f"{auc['mean']:.4f}±{auc['std']:.4f}  "
          f"{sens['mean']:.4f}±{sens['std']:.4f}  "
          f"{spec['mean']:.4f}±{spec['std']:.4f}")

with open(out_root / "all_summaries.json", "w") as f:
    json.dump(all_summaries, f, indent=2)
print(f"\nGuardado en: {out_root / 'all_summaries.json'}")
