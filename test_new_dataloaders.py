import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.loaders.dataloader1 import build_dataloaders

# ── 1. Parámetros sintéticos ─────────────────────────────────────────────
SYNTHETIC_DIR = Path("data/datasets/sintetic")
RAW_DIR = SYNTHETIC_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

N_SUBJECTS = 20
N_ROIS = 10
MAX_SEQ_LEN = 50
PCC_DIM = N_ROIS * (N_ROIS - 1) // 2

# ── 2. Generar series aleatorias (todas ≥ MAX_SEQ_LEN) ──────────────────
np.random.seed(42)
file_ids = []
for i in range(N_SUBJECTS):
    # Aseguramos longitud mínima = MAX_SEQ_LEN + margen para truncar
    t = np.random.randint(MAX_SEQ_LEN, MAX_SEQ_LEN + 20)
    ts = np.random.randn(t, N_ROIS).astype(np.float32)
    fname = f"sub-{i:03d}_rois_cc200.1D"
    np.savetxt(RAW_DIR / fname, ts, fmt='%.6f')
    file_ids.append(f"sub-{i:03d}")

# ── 3. DataFrame con columnas necesarias ─────────────────────────────────
df = pd.DataFrame({
    "FILE_ID": file_ids,
    "SUB_ID": [f"SUB-{i:03d}" for i in range(N_SUBJECTS)],
    "SITE_ID": np.random.choice(["site_A", "site_B", "site_C"], N_SUBJECTS),
    "DX_GROUP": np.random.randint(0, 2, N_SUBJECTS),
    "AGE_AT_SCAN": np.random.uniform(10, 60, N_SUBJECTS),
    "SEX": np.random.randint(1, 3, N_SUBJECTS),
    "EYE_STATUS_AT_SCAN": np.random.randint(1, 3, N_SUBJECTS),
})

# División simple (70/15/15)
indices = df.index.values
np.random.shuffle(indices)
n_train = int(0.7 * N_SUBJECTS)
n_val = int(0.85 * N_SUBJECTS)
train_idx = indices[:n_train]
val_idx = indices[n_train:n_val]
test_idx = indices[n_val:]

df_train = df.loc[train_idx].reset_index(drop=True)
df_val = df.loc[val_idx].reset_index(drop=True)
df_test = df.loc[test_idx].reset_index(drop=True)

# ── 4. Configuración mínima ──────────────────────────────────────────────
config = {
    "EXPERIMENT_TYPE": "pretrain_ts",
    "RAW_PATH": str(RAW_DIR),
    "ATLAS": "cc200",
    "N_ROIS": N_ROIS,
    "MAX_SEQ_LEN": MAX_SEQ_LEN,
    "LABEL_COL": "DX_GROUP",
    "NUM_WORKERS": 0,
    "SEED": 42,
    "TST2": {"PCC_DIM": PCC_DIM},
    "PT_TST1": {"BATCH_SIZE": 4},
    "PT_TST2": {"BATCH_SIZE": 4},
    "T_CONTRASTIVE": {"BATCH_SIZE": 4},
    "FINETUNING": {"BATCH_SIZE": 4},
    "USE_WEIGHTED_SAMPLER": False,
}

# ── 5. Ejecutar pruebas para cada fase ───────────────────────────────────
for phase in ["pretrain_ts", "pretrain_fc", "contrastive", "finetune"]:
    config["EXPERIMENT_TYPE"] = phase
    print(f"\n=== Test fase: {phase} ===")
    try:
        train_loader, val_loader, test_loader = build_dataloaders(
            config, df_train, df_val, df_test,
            harmonizer=None, normalizer=None
        )
        # Tomar un batch de ejemplo
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                data, labels = batch
                if isinstance(data, dict):
                    print(f"Batch OK - ts shape: {data['timeseries'].shape}, pcc shape: {data['pcc_vector'].shape}")
                else:
                    print(f"Batch OK - data shape: {data.shape}")
            else:
                print(f"Batch OK - type: {type(batch)}")
            break
    except Exception as e:
        print(f"Error en fase {phase}: {e}")

print("\n✅ Todos los tests pasaron.")