"""
Data Loader Module — carga .1D + CSV en runtime, con cache en disco de
conectividad.

Expone:
    load_raw_data            → dict {timeseries, pcc_vectors, labels,
                                     subject_indices, site_ids, site_to_idx}
    TwoTSTDataset            → finetune (dict con ts, pcc, label)
    PretrainTSDataset        → pretrain TST1 (solo ts)
    PretrainFCDataset        → pretrain TST2 (solo pcc)
    get_pretrain_loaders     → (train_ts, val_ts, train_fc, val_fc)
    get_finetune_loaders     → (train, val, test, split_info)
    get_single_split_loaders → split 70/10/20 único

Conectividad:
    PCC_KIND = "tangent"  → nilearn ConnectivityMeasure, batch, cacheable
    PCC_KIND = "pearson"  → legacy, por sujeto

Fuente de .1D:
    USE_INTERP = True   → carga de INTERP_PATH con prefix=PREFIX ("interp_")
    USE_INTERP = False  → carga de RAW_PATH sin prefix

Cache en disco:
    {CONNECTIVITY_PATH}/connectivity_{kind}_{atlas}_T{max_seq_len}_{src}.npz
    donde src ∈ {"raw", "interp"}.
    Si existe y cubre todos los FILE_IDs válidos, se carga directo.
    Si no, se calcula y se guarda automáticamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data.preprocessing.splitters import (
    get_subject_level_fold_splits,
    get_subject_level_train_val_test_split,
    get_loso_fold_splits,
)
from data.loaders.pcc_utils import (
    compute_pcc_vector,
    compute_pcc_tangent_batch,
)

_DATA_CACHE = {}


# ──────────────────────────────────────────────────────────────────────
# Cache de conectividad en disco
# ──────────────────────────────────────────────────────────────────────

def _src_tag(use_interp: bool) -> str:
    return "interp" if use_interp else "raw"


def _connectivity_cache_path(config: dict, use_interp: bool) -> Path | None:
    """Devuelve el path del .npz de cache, o None si CONNECTIVITY_PATH no está."""
    cache_dir = config.get("CONNECTIVITY_PATH")
    if not cache_dir:
        return None
    kind = config.get("PCC_KIND", "tangent")
    atlas = config["ATLAS"]
    max_seq_len = int(config["MAX_SEQ_LEN"])
    src = _src_tag(use_interp)
    return Path(cache_dir) / f"connectivity_{kind}_{atlas}_T{max_seq_len}_{src}.npz"


def _load_connectivity_cache(
    config: dict, use_interp: bool, valid_file_ids: list[str]
) -> np.ndarray | None:
    """
    Intenta cargar el cache de conectividad.

    Devuelve los vectores reordenados según `valid_file_ids`, o None si:
      - No existe CONNECTIVITY_PATH
      - No existe el .npz
      - Alguno de los valid_file_ids no está en el cache
    """
    path = _connectivity_cache_path(config, use_interp)
    if path is None or not path.exists():
        return None

    data = np.load(path, allow_pickle=True)
    cached_ids = data["file_ids"].tolist()
    idx = {fid: i for i, fid in enumerate(cached_ids)}

    missing = [fid for fid in valid_file_ids if fid not in idx]
    if missing:
        print(f"⚠ Cache incompleto ({len(missing)} FILE_IDs faltantes). "
              f"Se recalculará la conectividad.")
        return None

    order = [idx[fid] for fid in valid_file_ids]
    return data["vectors"][order]


def _save_connectivity_cache(
    config: dict,
    use_interp: bool,
    file_ids: list[str],
    subject_ids: np.ndarray,
    site_ids: np.ndarray,
    labels: np.ndarray,
    vectors: np.ndarray,
) -> Path:
    """Guarda el .npz de conectividad. Devuelve el path."""
    from datetime import datetime

    path = _connectivity_cache_path(config, use_interp)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "atlas": config["ATLAS"],
        "n_rois": config["N_ROIS"],
        "kind": config.get("PCC_KIND", "tangent"),
        "src_tag": _src_tag(use_interp),
        "max_seq_len": int(config["MAX_SEQ_LEN"]),
        "min_timesteps": int(config["MIN_TIMESTEPS"]),
        "n_subjects": len(file_ids),
        "timestamp": datetime.now().isoformat(),
    }

    np.savez_compressed(
        path,
        vectors=vectors,
        file_ids=np.array(file_ids),
        subject_ids=subject_ids,
        site_ids=site_ids,
        labels=labels,
        meta=np.array([meta], dtype=object),
    )
    return path


# ──────────────────────────────────────────────────────────────────────
# Pearson legacy (por sujeto)
# ──────────────────────────────────────────────────────────────────────

def _compute_pcc_pearson(ts: np.ndarray) -> np.ndarray | None:
    """
    ts: (R, T) → vector pearson upper-triangle (D,).
    Devuelve None si alguna ROI es constante.
    """
    t = torch.from_numpy(ts).float()
    if (t.std(dim=1) < 1e-8).any():
        return None
    return compute_pcc_vector(t).numpy()


# ──────────────────────────────────────────────────────────────────────
# Carga principal
# ──────────────────────────────────────────────────────────────────────

def load_raw_data(config: dict, use_interp: bool = None) -> dict:
    """
    Carga .1D + CSV. Aplica filtros:
      1. Excluye FILE_IDs sin archivo
      2. Excluye T < MAX_SEQ_LEN
      3. Excluye ROIs constantes (std < 1e-8)
      4. Crop a MAX_SEQ_LEN

    Conectividad:
      - Si existe cache en disco válido → carga
      - Si no → calcula (tangent por batch o pearson por sujeto) y guarda cache
    """
    project_root = Path(__file__).resolve().parents[2]

    if use_interp is None:
        use_interp = config.get("USE_INTERP", False)

    raw_key = "INTERP_PATH" if use_interp else "RAW_PATH"
    cache_key = (
        config.get("RAW_PATH"),
        config.get("INTERP_PATH"),
        config.get("CSV_PATH"),
        use_interp,
        config.get("MAX_SEQ_LEN"),
        config.get("ATLAS"),
        config.get("PREFIX"),
        config.get("PCC_KIND", "tangent"),
        config.get("CONNECTIVITY_PATH"),
    )
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    root = Path(config[raw_key])
    if not root.is_absolute():
        root = (project_root / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"No existe: {root}")

    csv_path = Path(config["CSV_PATH"])
    if not csv_path.is_absolute():
        csv_path = (project_root / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe: {csv_path}")

    atlas = config["ATLAS"]
    prefix = config.get("PREFIX", "interp_") if use_interp else ""
    n_rois = config["N_ROIS"]
    label_col = config["LABEL_COL"]
    max_seq_len = int(config.get("MAX_SEQ_LEN", 200))

    df = pd.read_csv(csv_path)

    ts_list, file_ids, labels, subj_ids, sites = [], [], [], [], []
    missing = too_short = const_roi = 0

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Cargando .1D de {root.name}"
    ):
        fname = f"{prefix}{row['FILE_ID']}_rois_{atlas}.1D"
        fpath = root / fname
        if not fpath.exists():
            missing += 1
            continue

        arr = np.loadtxt(fpath)

        if arr.shape[1] != n_rois:
            if arr.shape[0] == n_rois:
                arr = arr.T
            else:
                raise ValueError(
                    f"{fname}: shape {arr.shape}, esperaba N_ROIS={n_rois}"
                )

        if arr.shape[0] < max_seq_len:
            too_short += 1
            continue

        arr = arr[:max_seq_len]

        ts = arr.T.astype(np.float32)            # (R, T)
        if (np.std(ts, axis=1) < 1e-8).any():
            const_roi += 1
            continue

        ts_list.append(arr.astype(np.float32))   # (T, R)
        file_ids.append(str(row["FILE_ID"]))
        labels.append(int(row[label_col]))
        subj_ids.append(int(row["SUB_ID"]))
        sites.append(str(row["SITE_ID"]))

    if missing:
        print(f"⚠ {missing} sujetos saltados por .1D ausente")
    if too_short:
        print(f"⚠ {too_short} sujetos saltados por T < MAX_SEQ_LEN ({max_seq_len})")
    if const_roi:
        print(f"⚠ {const_roi} sujetos saltados por ROIs constantes "
              f"(QC paper Sec. 4.1.3)")

    if not ts_list:
        raise RuntimeError(f"No se cargó ningún .1D de {root}")

    ts_array   = np.stack(ts_list).astype(np.float32)     # (N, T, R)
    labels_arr = np.array(labels, dtype=np.int64)
    subj_arr   = np.array(subj_ids, dtype=np.int64)
    sites_arr  = np.array(sites)

    # ─── Conectividad: cache → calcular → guardar ──────────────────
    kind = config.get("PCC_KIND", "tangent")
    pcc = _load_connectivity_cache(config, use_interp, file_ids)

    if pcc is not None:
        print(f"✓ Conectividad ({kind}, {_src_tag(use_interp)}) "
              f"cargada de cache: {_connectivity_cache_path(config, use_interp)}")
    else:
        print(f"Calculando conectividad ({kind}, {_src_tag(use_interp)}) "
              f"para {len(file_ids)} sujetos...")

        if kind == "tangent":
            pcc = compute_pcc_tangent_batch(ts_array)
        elif kind == "pearson":
            vecs = []
            for ts in ts_array:
                v = _compute_pcc_pearson(ts.T)
                if v is None:
                    raise RuntimeError(
                        "ROI constante detectada tras el filtro previo"
                    )
                vecs.append(v)
            pcc = np.stack(vecs).astype(np.float32)
        else:
            raise ValueError(f"PCC_KIND desconocido: {kind!r}")

        # Auto-guardar cache
        try:
            saved = _save_connectivity_cache(
                config, use_interp, file_ids, subj_arr, sites_arr, labels_arr, pcc
            )
            print(f"✓ Cache guardado en: {saved}")
        except Exception as e:
            print(f"⚠ No se pudo guardar cache: {e}")

    # ─── Mapeo de sitios ───────────────────────────────────────────
    unique_sites = sorted(set(sites))
    site_to_idx = {s: i for i, s in enumerate(unique_sites)}

    data = {
        "timeseries":      ts_array,
        "pcc_vectors":     pcc.astype(np.float32),
        "labels":          labels_arr,
        "subject_indices": subj_arr,
        "site_ids":        sites_arr,
        "site_to_idx":     site_to_idx,
    }

    print(
        f"Cargados {len(labels)} sujetos  |  "
        f"ts={ts_array.shape}  pcc={pcc.shape}"
    )
    _DATA_CACHE[cache_key] = data
    return data


# ──────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────

class TwoTSTDataset(Dataset):
    def __init__(
        self,
        timeseries,
        pcc_vectors,
        labels,
        site_ids=None,
        site_to_idx=None,
        normalize_ts=True,
        normalize_pcc=True,
    ):
        self.timeseries  = timeseries.astype(np.float32)
        self.pcc_vectors = pcc_vectors.astype(np.float32)
        self.labels      = labels.astype(np.int64)

        if site_ids is not None and site_to_idx is not None:
            self.site_ids = np.array(
                [site_to_idx[s] for s in site_ids], dtype=np.int64
            )
        else:
            self.site_ids = None

        if normalize_ts:
            mean = self.timeseries.mean(axis=(1, 2), keepdims=True)
            std = self.timeseries.std(axis=(1, 2), keepdims=True) + 1e-8
            self.timeseries = (self.timeseries - mean) / std

        if normalize_pcc:
            mean = self.pcc_vectors.mean(axis=1, keepdims=True)
            std = self.pcc_vectors.std(axis=1, keepdims=True) + 1e-8
            self.pcc_vectors = (self.pcc_vectors - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "timeseries": torch.from_numpy(self.timeseries[idx]),
            "pcc_vector": torch.from_numpy(self.pcc_vectors[idx]),
            "label":      torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.site_ids is not None:
            item["site_id"] = torch.tensor(
                self.site_ids[idx], dtype=torch.long
            )
        return item


class PretrainTSDataset(Dataset):
    def __init__(self, timeseries):
        ts = timeseries.astype(np.float32)
        mean = ts.mean(axis=(1, 2), keepdims=True)
        std = ts.std(axis=(1, 2), keepdims=True) + 1e-8
        self.timeseries = (ts - mean) / std

    def __len__(self):
        return len(self.timeseries)

    def __getitem__(self, idx):
        return torch.from_numpy(self.timeseries[idx])


class PretrainFCDataset(Dataset):
    def __init__(self, pcc_vectors):
        pcc = pcc_vectors.astype(np.float32)
        mean = pcc.mean(axis=1, keepdims=True)
        std = pcc.std(axis=1, keepdims=True) + 1e-8
        self.pcc_vectors = (pcc - mean) / std

    def __len__(self):
        return len(self.pcc_vectors)

    def __getitem__(self, idx):
        return torch.from_numpy(self.pcc_vectors[idx])


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────

def get_pretrain_loaders(
    config: dict,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
):
    data = load_raw_data(config)
    ts, pcc = data["timeseries"], data["pcc_vectors"]

    train_idx, val_idx, _ = get_subject_level_train_val_test_split(
        data["labels"], data["subject_indices"], site_ids=data["site_ids"],
        train_ratio=1.0 - val_ratio - test_ratio,
        val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
    )

    pin = torch.cuda.is_available()
    train_ts = DataLoader(PretrainTSDataset(ts[train_idx]),
                          batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
    val_ts = DataLoader(PretrainTSDataset(ts[val_idx]),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin)
    train_fc = DataLoader(PretrainFCDataset(pcc[train_idx]),
                          batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
    val_fc = DataLoader(PretrainFCDataset(pcc[val_idx]),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin)

    return train_ts, val_ts, train_fc, val_fc


def get_finetune_loaders(
    config: dict,
    batch_size: int = 32,
    num_workers: int = 4,
    fold_idx: int = 0,
    n_folds: int = 5,
    val_ratio: float = 0.15,
    seed: int = 42,
    eval_protocol: str = "kfold",
):
    data = load_raw_data(config)
    labels = data["labels"]
    subject_indices = data["subject_indices"]
    site_ids = data["site_ids"]

    if eval_protocol == "loso":
        if site_ids is None:
            raise ValueError("LOSO requiere site_ids.")
        splits = get_loso_fold_splits(
            labels, subject_indices, site_ids,
            val_ratio=val_ratio, seed=seed,
        )
    else:
        splits = get_subject_level_fold_splits(
            labels, subject_indices, site_ids=site_ids,
            n_splits=n_folds, val_ratio=val_ratio, seed=seed,
        )

    split = splits[fold_idx]
    train_idx, val_idx, test_idx = (
        split["train_idx"], split["val_idx"], split["test_idx"]
    )

    ts, pcc = data["timeseries"], data["pcc_vectors"]
    site_to_idx = data["site_to_idx"]

    train_ds = TwoTSTDataset(
        ts[train_idx], pcc[train_idx], labels[train_idx],
        site_ids=site_ids[train_idx], site_to_idx=site_to_idx,
    )
    val_ds = TwoTSTDataset(
        ts[val_idx], pcc[val_idx], labels[val_idx],
        site_ids=site_ids[val_idx], site_to_idx=site_to_idx,
    )
    test_ds = TwoTSTDataset(
        ts[test_idx], pcc[test_idx], labels[test_idx],
        site_ids=site_ids[test_idx], site_to_idx=site_to_idx,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    split_info = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "subject_indices": subject_indices,
        "fold_idx": fold_idx,
        "eval_protocol": eval_protocol,
    }
    if "test_site" in split:
        split_info["test_site"] = split["test_site"]

    return train_loader, val_loader, test_loader, split_info


def get_single_split_loaders(
    config: dict,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
):
    """
    Split único 70/10/20 a nivel de sujeto.
    Usado por la fase contrastive global (paper Sec. 3.3).
    """
    data = load_raw_data(config)
    labels = data["labels"]
    subject_indices = data["subject_indices"]
    site_ids = data["site_ids"]

    train_idx, val_idx, test_idx = get_subject_level_train_val_test_split(
        labels, subject_indices, site_ids=site_ids,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        seed=seed,
    )

    ts, pcc = data["timeseries"], data["pcc_vectors"]

    train_ds = TwoTSTDataset(ts[train_idx], pcc[train_idx], labels[train_idx])
    val_ds   = TwoTSTDataset(ts[val_idx],   pcc[val_idx],   labels[val_idx])
    test_ds  = TwoTSTDataset(ts[test_idx],  pcc[test_idx],  labels[test_idx])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    split_info = {
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "subject_indices": subject_indices,
    }
    return train_loader, val_loader, test_loader, split_info