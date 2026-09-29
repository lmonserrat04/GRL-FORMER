"""
Interpolación de series temporales fMRI a un TR común.

Flujo:
  1. Cargar .1D (T, R)
  2. Excluir sujetos con T_raw < MIN_TIMESTEPS
  3. Excluir sujetos con ROIs constantes (std < 1e-8) — ANTES de interpolar
  4. Interpolar a TR_goal
  5. Crop a MAX_SEQ_LEN
  6. Guardar como .1D (T, R) en INTERP_PATH

Uso:
    python data/preprocessing/interp.py --config config/config.yaml
"""

# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.augmentation.augmentation import interpolate_timeseries


REQUIRED_COLUMNS = ["FILE_ID", "SITE_ID"]


def _load_arr(path: Path, n_rois: int) -> np.ndarray:
    """Carga un .1D y lo normaliza a (T, R)."""
    arr = np.loadtxt(path)
    if arr.shape[1] != n_rois:
        if arr.shape[0] == n_rois:
            arr = arr.T
        else:
            raise ValueError(
                f"{path.name}: shape {arr.shape}, "
                f"esperaba {n_rois} ROIs en dim 0 o 1"
            )
    return arr


def _has_constant_roi(ts: np.ndarray) -> bool:
    """ts: (R, T). True si alguna ROI tiene std < 1e-8."""
    return bool((ts.std(axis=1) < 1e-8).any())


def generate_interp(config: dict, df: pd.DataFrame) -> dict:
    """
    Genera los .1D interpolados en INTERP_PATH.

    Returns:
        dict con contadores de cada exclusión.
    """
    n_rois       = config["N_ROIS"]
    raw_path     = Path(config["RAW_PATH"])
    atlas        = config["ATLAS"]
    tr_goal      = float(config["TR"])
    output_path  = Path(config["INTERP_PATH"])
    prefix       = config["PREFIX"]
    min_ts       = int(config["MIN_TIMESTEPS"])
    max_seq_len  = int(config["MAX_SEQ_LEN"])

    output_path.mkdir(parents=True, exist_ok=True)

    counts = {
        "ok": 0,
        "missing_file": 0,
        "short_raw": 0,
        "const_roi": 0,
        "interp_failed": 0,
        "short_interp": 0,
    }

    for _, row in df.iterrows():
        filename = f"{row.FILE_ID}_rois_{atlas}.1D"
        filepath = raw_path / filename

        if not filepath.exists():
            counts["missing_file"] += 1
            continue

        # 1. Cargar (T, R)
        arr = _load_arr(filepath, n_rois)

        # 2. Filtrar por T_raw < MIN_TIMESTEPS
        if arr.shape[0] < min_ts:
            counts["short_raw"] += 1
            continue

        # 3. Filtrar ROIs constantes (ANTES de interpolar)
        ts = arr.T  # (R, T)
        if _has_constant_roi(ts):
            counts["const_roi"] += 1
            continue

        # 4. Interpolar a TR_goal
        arr_interp = interpolate_timeseries(ts, row.SITE_ID, tr_goal, min_ts)
        if arr_interp is None:
            counts["interp_failed"] += 1
            continue

        # 5. Crop a MAX_SEQ_LEN
        if arr_interp.shape[1] < max_seq_len:
            counts["short_interp"] += 1
            continue
        arr_interp = arr_interp[:, :max_seq_len]

        # 6. Guardar como (T, R) para coincidir con el formato de los .1D raw
        output_filename = prefix + filename
        np.savetxt(output_path / output_filename, arr_interp.T)
        counts["ok"] += 1

    return counts


def load_and_validate_df(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["CSV_PATH"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return df


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = load_and_validate_df(config)

    print(f"Interpolando {len(df)} sujetos → TR_goal={config['TR']}s, "
          f"crop a {config['MAX_SEQ_LEN']} timesteps")
    print(f"  RAW_PATH:    {config['RAW_PATH']}")
    print(f"  INTERP_PATH: {config['INTERP_PATH']}")

    counts = generate_interp(config, df)

    print("\nResumen:")
    for k, v in counts.items():
        print(f"  {k:<15s}: {v:>5d}")
    print(f"\nTotal escrito: {counts['ok']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpola series ROI-level a un TR común y crops a MAX_SEQ_LEN."
    )
    parser.add_argument("--config", default="./config/config.yaml")
    main(parser.parse_args())