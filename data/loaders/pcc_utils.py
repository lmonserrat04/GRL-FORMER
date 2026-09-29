"""
Utilidades de conectividad funcional.

- compute_pcc_vector: Pearson upper-triangle (legacy, por sujeto).
- compute_pcc_tangent_batch: tangent space (nilearn + joblib threads).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from joblib import Parallel, delayed
from nilearn.connectome import ConnectivityMeasure


def compute_pcc_vector(timeseries: torch.Tensor) -> torch.Tensor:
    """Vector PCC de Pearson (triangular superior). Legacy."""
    corr = torch.corrcoef(timeseries)
    triu_idx = torch.triu_indices(corr.shape[0], corr.shape[1], offset=1)
    return corr[triu_idx[0], triu_idx[1]]


def _safe_n_jobs(n_jobs: int) -> int:
    """
    Traduce n_jobs a un valor seguro para esta máquina.

    Reglas:
      - n_jobs == -1 → 4 (cap duro para 15 GiB RAM / 8 cores físicos)
      - n_jobs == 0  → 1
      - n_jobs < 0 y distinto de -1 → 1 (conservador)
      - n_jobs > 0   → cap a 6 (para no saturar RAM)
    """
    if n_jobs == -1:
        return 4
    if n_jobs <= 0:
        return 1
    return min(int(n_jobs), 6)


def compute_pcc_tangent_batch(
    timeseries_batch: np.ndarray,
    n_jobs: int = -1,
    verbose: int = 0,
) -> np.ndarray:
    """
    Vectores de conectividad en tangent space (paralelizado con threads).

    nilearn usa scipy.linalg.logm (BLAS/LAPACK) que libera el GIL, así que
    threads dan el mismo speedup que processes con mucha menos RAM.

    nilearn espera cada sujeto como (n_samples, n_features) = (T, R).

    Args:
        timeseries_batch: (N, T, R).
        n_jobs:           int. -1 se traduce a 4 (safe cap).
        verbose:          nivel de joblib.

    Returns:
        vectors: (N, R*(R-1)//2).
    """
    if timeseries_batch.ndim != 3:
        raise ValueError(
            f"Se esperaba (N, T, R), recibido {timeseries_batch.shape}"
        )

    n_jobs_eff = _safe_n_jobs(n_jobs)
    N, T, R = timeseries_batch.shape

    ts_list = [timeseries_batch[i] for i in range(N)]   # (T, R)

    measure = ConnectivityMeasure(
        kind="tangent",
        vectorize=True,
        discard_diagonal=True,
    )

    # Fit: referencia (media geométrica). Rápido, secuencial.
    measure.fit(ts_list)

    # Transform en chunks (menos overhead de scheduling)
    chunk_size = 16
    chunks = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

    def _transform_chunk(start_stop):
        start, stop = start_stop
        return [measure.transform([ts_list[i]])[0] for i in range(start, stop)]

    results = Parallel(
        n_jobs=n_jobs_eff,
        prefer="threads",          # ← clave: threads en vez de processes
        verbose=verbose,
        batch_size="auto",
    )(delayed(_transform_chunk)(c) for c in chunks)

    vectors = np.concatenate(results, axis=0)
    return vectors.astype(np.float32)
