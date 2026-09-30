"""
Precomputa y guarda los vectores de conectividad (tangent o pearson).

Lee de RAW_PATH o INTERP_PATH según USE_INTERP.

Flujo:
  1. Leer CSV
  2. Cargar .1D desde RAW_PATH (USE_INTERP=false) o INTERP_PATH (USE_INTERP=true)
  3. Filtrar T<MIN_TIMESTEPS y ROIs constantes
  4. Crop a MAX_SEQ_LEN
  5. Calcular ConnectivityMeasure (kind=PCC_KIND, vectorize, discard_diagonal)
  6. Guardar un único .npz con todos los vectores + metadatos

Uso:
    python data/preprocessing/compute_connectivity.py --config config/config.yaml

Salida:
    {CONNECTIVITY_PATH}/connectivity_{kind}_{atlas}_T{max_seq_len}_{src}.npz
    con src = "raw" | "interp"
    arrays: vectors, file_ids, subject_ids, site_ids, labels, meta
"""

# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def _load_arr(path: Path, n_rois: int) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.shape[1] != n_rois:
        if arr.shape[0] == n_rois:
            arr = arr.T
        else:
            raise ValueError(f"{path.name}: shape {arr.shape}")
    return arr


def _has_constant_roi(ts: np.ndarray) -> bool:
    return bool((ts.std(axis=1) < 1e-8).any())


def _resolve_source(config: dict) -> tuple[Path, str, str]:
    """
    Devuelve (src_path, prefix, src_tag) según USE_INTERP.

    src_tag ∈ {"raw", "interp"} se usa para nombrar el cache.
    """
    use_interp = bool(config.get("USE_INTERP", False))

    if use_interp:
        src_key = "INTERP_PATH"
        prefix = config.get("PREFIX", "interp_")
        src_tag = "interp"
    else:
        src_key = "RAW_PATH"
        prefix = ""
        src_tag = "raw"

    src_path = Path(config[src_key])
    if not src_path.exists():
        raise FileNotFoundError(f"{src_key} no existe: {src_path}")

    return src_path, prefix, src_tag


def _cache_path(config: dict, kind: str, src_tag: str) -> Path:
    out_dir = Path(config["CONNECTIVITY_PATH"])
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas = config["ATLAS"]
    max_seq_len = int(config["MAX_SEQ_LEN"])
    return out_dir / f"connectivity_{kind}_{atlas}_T{max_seq_len}_{src_tag}.npz"


def compute_all(config: dict) -> dict:
    from data.loaders.pcc_utils import (
        compute_pcc_tangent_batch,
        compute_pcc_vector,
    )

    atlas = config["ATLAS"]
    n_rois = config["N_ROIS"]
    max_seq_len = int(config["MAX_SEQ_LEN"])
    min_ts = int(config["MIN_TIMESTEPS"])
    kind = config.get("PCC_KIND", "tangent")

    src_path, prefix, src_tag = _resolve_source(config)

    print(f"Fuente:      {src_path}  (prefix={prefix!r}, tag={src_tag})")

    df = pd.read_csv(config["CSV_PATH"])
    label_col = config["LABEL_COL"]

    ts_list, file_ids, subj_ids, sites, labels = [], [], [], [], []
    counts = {"ok": 0, "missing": 0, "short": 0, "const_roi": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Leyendo .1D de {src_path.name}"):
        fname = f"{prefix}{row['FILE_ID']}_rois_{atlas}.1D"
        fpath = src_path / fname
        if not fpath.exists():
            counts["missing"] += 1
            continue

        arr = _load_arr(fpath, n_rois)             # (T, R)
        if arr.shape[0] < max_seq_len:
            counts["short"] += 1
            continue
        arr = arr[:max_seq_len]

        ts = arr.T                                 # (R, T)
        if _has_constant_roi(ts):
            counts["const_roi"] += 1
            continue

        ts_list.append(arr)                        # (T, R)
        file_ids.append(str(row["FILE_ID"]))
        subj_ids.append(int(row["SUB_ID"]))
        sites.append(str(row["SITE_ID"]))
        labels.append(int(row[label_col]))
        counts["ok"] += 1

    if not ts_list:
        raise RuntimeError("No se cargó ningún sujeto")

    all_ts = np.stack(ts_list).astype(np.float32)  # (N, T, R)
    print(f"Calculando conectividad kind={kind} para {all_ts.shape[0]} sujetos...")

    if kind == "tangent":
        vectors = compute_pcc_tangent_batch(all_ts)
    elif kind == "pearson":
        vecs = [
            compute_pcc_vector(np.asarray(x.T)).numpy()
            for x in all_ts
        ]
        vectors = np.stack(vecs).astype(np.float32)
    else:
        raise ValueError(f"PCC_KIND desconocido: {kind!r}")

    return {
        "vectors": vectors,
        "file_ids": np.array(file_ids),
        "subject_ids": np.array(subj_ids, dtype=np.int64),
        "site_ids": np.array(sites),
        "labels": np.array(labels, dtype=np.int64),
        "counts": counts,
        "src_tag": src_tag,
        "meta": {
            "atlas": atlas,
            "n_rois": n_rois,
            "kind": kind,
            "src_tag": src_tag,
            "max_seq_len": max_seq_len,
            "min_timesteps": min_ts,
            "n_subjects": len(ts_list),
            "timestamp": datetime.now().isoformat(),
        },
    }


def save_cache(result: dict, config: dict) -> Path:
    kind = config.get("PCC_KIND", "tangent")
    src_tag = result["src_tag"]
    path = _cache_path(config, kind, src_tag)

    np.savez_compressed(
        path,
        vectors=result["vectors"],
        file_ids=result["file_ids"],
        subject_ids=result["subject_ids"],
        site_ids=result["site_ids"],
        labels=result["labels"],
        meta=np.array([result["meta"]], dtype=object),
    )
    return path


def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"USE_INTERP:    {config.get('USE_INTERP')}")
    print(f"PCC_KIND:      {config.get('PCC_KIND', 'tangent')}")
    print(f"CACHE OUT:     {config['CONNECTIVITY_PATH']}")

    result = compute_all(config)
    path = save_cache(result, config)

    print("\nFiltros:")
    for k, v in result["counts"].items():
        print(f"  {k:<12s}: {v}")
    print(f"\nShape vectors: {result['vectors'].shape}")
    print(f"Guardado en:   {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/config.yaml")
    main(parser.parse_args())