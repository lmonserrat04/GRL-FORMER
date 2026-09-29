"""
Verifica el dropout real aplicado en el modelo finetuneado.
"""
# --- sys.path bootstrap (scripts movidos a subcarpetas) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

import glob
from pathlib import Path

import yaml
import torch

from training.setup import build_experiment


exp_dirs = sorted(glob.glob("experiments/*_transformer_baseline"))
src = None
for d in reversed(exp_dirs):
    if (Path(d) / "checkpoints" / "contrastive_global.pt").exists():
        src = Path(d)
        break

if src is None:
    raise SystemExit("No hay experimento con contrastive_global.pt")

print(f"Usando experimento: {src}\n")

with open(src / "config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 60)
print("CONFIG")
print("=" * 60)
print(f"MLP_HEAD.DROPOUT            = {config['MLP_HEAD']['DROPOUT']}")
print(f"TST1.ENC_DROP               = {config['TST1']['ENC_DROP']}")
print(f"TST2.ENC_DROP               = {config['TST2']['ENC_DROP']}")


config["EXP_DIR"] = str(src)
config["CHECKPOINTS_PATH"] = str(src / "checkpoints")
config["CKPT_TST1"] = str(src / "checkpoints" / "best_pt_ts_fold_0.pt")
config["CKPT_TST2"] = str(src / "checkpoints" / "best_pt_fc_fold_0.pt")
config["CKPT_CONTRASTIVE"] = str(src / "checkpoints" / "contrastive_global.pt")
config["EXPERIMENT_TYPE"] = "finetune"
config["DEVICE"] = "cpu"

exp = build_experiment(config, fold_idx=0,
                       ckpt_contrastive=config["CKPT_CONTRASTIVE"])
model = exp.model

print()
print("=" * 60)
print("MODELO — capas Dropout y su probabilidad")
print("=" * 60)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout):
        print(f"  {name:<70s}  p={module.p}")


dropouts = [(n, m.p) for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout)]
unique_p = sorted(set(p for _, p in dropouts))

print()
print("=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"Total capas Dropout: {len(dropouts)}")
print(f"Valores únicos de p: {unique_p}")

# Dropout del tag_classifier
tag_dropouts = [(n, m.p) for n, m in model.tag_classifier.named_modules()
                if isinstance(m, torch.nn.Dropout)]
print(f"tag_classifier dropout p: {sorted(set(p for _, p in tag_dropouts))}")

# Dropout del domain_classifier
dom_dropouts = [(n, m.p) for n, m in model.domain_classifier.named_modules()
                if isinstance(m, torch.nn.Dropout)]
print(f"domain_classifier dropout p: {sorted(set(p for _, p in dom_dropouts))}")
