"""
Data Loader Module — carga .1D + CSV en runtime, sin .pkl intermedio.

Expone:
    load_raw_data            → dict {timeseries, pcc_vectors, labels, subject_indices, site_ids}
    TwoTSTDataset            → finetune (dict con ts, pcc, label)
    PretrainTSDataset        → pretrain TST1 (solo ts)
    PretrainFCDataset        → pretrain TST2 (solo pcc)
    get_pretrain_loaders     → (train_ts, val_ts, train_fc, val_fc)
    get_finetune_loaders     → (train, val, test, split_info)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_DATA_CACHE = {}

from data.preprocessing.splitters import (
    get_subject_level_fold_splits,
    get_subject_level_train_val_test_split,
    get_loso_fold_splits,
)


def _compute_pcc_upper(ts: np.ndarray) -> np.ndarray | None:
    """ts: (R, T) → triángulo superior de la matriz de correlación (D,).
    Devuelve None si alguna ROI es constante (varianza 0) — el paper
    descarta estos sujetos en QC (Sec. 4.1.3)."""
    ts_t = torch.from_numpy(ts).float()
    if (ts_t.std(dim=1) < 1e-8).any():
        return None
    corr = torch.corrcoef(ts_t)
    if torch.isnan(corr).any() or torch.isinf(corr).any():
        return None
    triu = torch.triu_indices(corr.shape[0], corr.shape[1], offset=1)
    return corr[triu[0], triu[1]].numpy().astype(np.float32)


def load_raw_data(config: dict, use_interp: bool = None) -> dict:
    """Carga .1D + CSV. Trunca a MAX_SEQ_LEN y descarta T < MAX_SEQ_LEN."""
    project_root = Path(__file__).resolve().parents[2]

    if use_interp is None:
        use_interp = config.get("USE_INTERP", False)

    raw_key = "INTERP_PATH" if use_interp else "RAW_PATH"
    cache_key = (config.get("RAW_PATH"), config.get("INTERP_PATH"),
                 config.get("CSV_PATH"), use_interp,
                 config.get("MAX_SEQ_LEN"), config.get("ATLAS"),
                 config.get("PREFIX"))
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

    ts_list, pcc_list, labels, subj_ids, sites = [], [], [], [], []
    missing = too_short = const_roi = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Cargando .1D de {root.name}"):
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
                raise ValueError(f"{fname}: shape {arr.shape}, esperaba N_ROIS={n_rois}")

        if arr.shape[0] < max_seq_len:
            too_short += 1
            continue

        arr = arr[:max_seq_len]

        ts = arr.T.astype(np.float32)                    # (R, T)
        pcc = _compute_pcc_upper(ts)
        if pcc is None:
            const_roi += 1
            continue

        ts_list.append(ts.T)                             # (T, R)
        pcc_list.append(pcc)
        labels.append(int(row[label_col]))
        subj_ids.append(int(row["SUB_ID"]))
        sites.append(str(row["SITE_ID"]))

    if missing:
        print(f"⚠ {missing} sujetos saltados por .1D ausente")
    if too_short:
        print(f"⚠ {too_short} sujetos saltados por T < MAX_SEQ_LEN ({max_seq_len})")
    if const_roi:
        print(f"⚠ {const_roi} sujetos saltados por ROIs constantes (QC paper Sec. 4.1.3)")

    if not ts_list:
        raise RuntimeError(f"No se cargó ningún .1D de {root}")

    data = {
        "timeseries":      np.stack(ts_list).astype(np.float32),
        "pcc_vectors":     np.stack(pcc_list).astype(np.float32),
        "labels":          np.array(labels, dtype=np.int64),
        "subject_indices": np.array(subj_ids, dtype=np.int64),
        "site_ids":        np.array(sites),
    }
    print(f"Cargados {len(labels)} sujetos  |  "
          f"ts={data['timeseries'].shape}  pcc={data['pcc_vectors'].shape}")
    _DATA_CACHE[cache_key] = data
    return data


class TwoTSTDataset(Dataset):
    def __init__(self, timeseries, pcc_vectors, labels,
                 normalize_ts=True, normalize_pcc=True):
        self.timeseries = timeseries.astype(np.float32)
        self.pcc_vectors = pcc_vectors.astype(np.float32)
        self.labels = labels.astype(np.int64)

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
        return {
            "timeseries": torch.from_numpy(self.timeseries[idx]),
            "pcc_vector": torch.from_numpy(self.pcc_vectors[idx]),
            "label":      torch.tensor(self.labels[idx], dtype=torch.long),
        }


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

    train_ds = TwoTSTDataset(ts[train_idx], pcc[train_idx], labels[train_idx])
    val_ds   = TwoTSTDataset(ts[val_idx],   pcc[val_idx],   labels[val_idx])
    test_ds  = TwoTSTDataset(ts[test_idx],  pcc[test_idx],  labels[test_idx])

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

    Returns:
        (train_loader, val_loader, test_loader, split_info)
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
                              drop_last=True, num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    split_info = {
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "subject_indices": subject_indices,
    }
    return train_loader, val_loader, test_loader, split_info
# ──────────────────────────────────────────────────────────────────────
# Tests (usan .1D sintéticos en disco)
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    rng = np.random.default_rng(0)
    N, T, R = 30, 100, 200
    D = R * (R - 1) // 2

    # ─── Crear estructura sintética en disco ──────────────────────────
    tmp = Path(tempfile.mkdtemp())
    raw_dir = tmp / "raw"
    interp_dir = tmp / "interp"
    raw_dir.mkdir(); interp_dir.mkdir()

    sites = ["SITE_0", "SITE_1", "SITE_2"]
    rows = []
    for i in range(N):
        fid = f"S{1000 + i}"
        arr = rng.standard_normal((T, R)).astype(np.float32)
        np.savetxt(interp_dir / f"interp_{fid}_rois_cc200.1D", arr)
        np.savetxt(raw_dir / f"{fid}_rois_cc200.1D", arr)
        rows.append({
            "FILE_ID": fid, "SUB_ID": i,
            "SITE_ID": sites[i % 3],
            "DX_GROUP": int(rng.integers(0, 2)),
        })

    csv_path = tmp / "meta.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    config = {
        "RAW_PATH": str(raw_dir),
        "INTERP_PATH": str(interp_dir),
        "CSV_PATH": str(csv_path),
        "ATLAS": "cc200",
        "PREFIX": "interp_",
        "N_ROIS": R,
        "LABEL_COL": "DX_GROUP",
    }

    # ─── TEST 1: load_raw_data ────────────────────────────────────────
    print("── TEST 1: load_raw_data ────────────────────────────────────")
    data = load_raw_data(config, use_interp=True)
    assert data["timeseries"].shape == (N, T, R)
    assert data["pcc_vectors"].shape == (N, D)
    assert data["labels"].shape == (N,)
    assert len(np.unique(data["site_ids"])) == 3
    print(f"  ✓ ts={data['timeseries'].shape}  pcc={data['pcc_vectors'].shape}\n")

    # ─── TEST 2: PCC correcto ─────────────────────────────────────────
    print("── TEST 2: PCC coincide con cálculo directo ────────────────")
    sample_ts = data["timeseries"][0].T                       # (R, T)
    expected_pcc = _compute_pcc_upper(sample_ts)
    assert np.allclose(data["pcc_vectors"][0], expected_pcc, atol=1e-5)
    print(f"  ✓ primer PCC coincide\n")

    # ─── TEST 3: pretrain loaders ─────────────────────────────────────
    print("── TEST 3: get_pretrain_loaders ─────────────────────────────")
    tr_ts, va_ts, tr_fc, va_fc = get_pretrain_loaders(
        config, batch_size=4, num_workers=0, seed=0,
    )
    b_ts = next(iter(tr_ts)); b_fc = next(iter(tr_fc))
    assert b_ts.shape == (4, T, R) and b_fc.shape == (4, D)
    print(f"  ✓ batch ts {tuple(b_ts.shape)}  fc {tuple(b_fc.shape)}\n")

    # ─── TEST 4: finetune kfold ───────────────────────────────────────
    print("── TEST 4: get_finetune_loaders (kfold) ─────────────────────")
    tr, va, te, si = get_finetune_loaders(
        config, batch_size=4, num_workers=0, fold_idx=0,
        eval_protocol="kfold", n_folds=3, seed=0,
    )
    batch = next(iter(tr))
    assert batch["timeseries"].shape == (4, T, R)
    assert batch["pcc_vector"].shape == (4, D)
    assert batch["label"].shape == (4,)
    print(f"  ✓ keys={list(batch.keys())}  "
          f"train={len(si['train_idx'])} val={len(si['val_idx'])} test={len(si['test_idx'])}\n")

    # ─── TEST 5: finetune loso ────────────────────────────────────────
    print("── TEST 5: get_finetune_loaders (loso) ──────────────────────")
    for f in range(3):
        _, _, _, si = get_finetune_loaders(
            config, batch_size=4, num_workers=0, fold_idx=f,
            eval_protocol="loso", seed=0,
        )
        assert "test_site" in si
        sites_in_test = np.unique(data["site_ids"][si["test_idx"]])
        assert len(sites_in_test) == 1 and sites_in_test[0] == si["test_site"]
        print(f"  ✓ fold {f} test_site={si['test_site']}  test={len(si['test_idx'])}")
    print()

    # ─── TEST 6: use_interp=False lee RAW_PATH ────────────────────────
    print("── TEST 6: use_interp=False lee RAW_PATH ────────────────────")
    data_raw = load_raw_data(config, use_interp=False)
    assert data_raw["timeseries"].shape == (N, T, R)
    print(f"  ✓ ts raw {data_raw['timeseries'].shape}\n")

    # limpieza
    import shutil; shutil.rmtree(tmp)
    print("✅ Todos los tests de dataloader.py pasaron.")