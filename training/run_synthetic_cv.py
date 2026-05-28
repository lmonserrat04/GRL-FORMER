# run_synthetic_cv.py

import sys
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.cross_validator_template import run_cross_validation

# ------------------------------------------------------------
# 1. Cargar YAML sintético
# ------------------------------------------------------------
config_path = "config/config_synth.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# ------------------------------------------------------------
# 2. Ajustar rutas y dispositivo
# ------------------------------------------------------------
config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
config["RAW_PATH"] = str(Path("data/datasets/synthetic/raw").resolve())
config["CSV_PATH"] = str(Path("data/datasets/synthetic/data.csv").resolve())

# ------------------------------------------------------------
# 3. Ejecutar validación cruzada
# ------------------------------------------------------------
print("=" * 60)
print("Iniciando smoke test con datos sintéticos (3 folds)")
print(f"Device: {config['DEVICE']}")
print(f"RAW_PATH: {config['RAW_PATH']}")
print(f"CSV_PATH: {config['CSV_PATH']}")
print("=" * 60)

run_cross_validation(config, n_folds=config.get("N_FOLDS", 3))

print("\n✅ Validación cruzada sintética finalizada sin errores.")