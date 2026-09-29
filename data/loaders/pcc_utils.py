"""
Utilidades de conectividad funcional.

- compute_pcc_vector: Pearson upper-triangle vector (legacy, por sujeto).
- compute_pcc_tangent_batch: tangent-space vector (nilearn, por batch).
"""
import numpy as np
import torch
from nilearn.connectome import ConnectivityMeasure


def compute_pcc_vector(timeseries: torch.Tensor) -> torch.Tensor:
    """
    Vector PCC de Pearson (triangular superior).
    Legacy: se mantiene por compatibilidad.

    Args:
        timeseries: tensor (N_ROIS, T)

    Returns:
        pcc_vector: tensor (N_ROIS*(N_ROIS-1)//2,)
    """
    corr = torch.corrcoef(timeseries)
    triu_idx = torch.triu_indices(corr.shape[0], corr.shape[1], offset=1)
    return corr[triu_idx[0], triu_idx[1]]


def compute_pcc_tangent_batch(timeseries_batch: np.ndarray) -> np.ndarray:
    """
    Calcula vectores de conectividad en tangent space para un batch.

    Requiere nilearn >= 0.9. Usa el geometric mean del batch como referencia.

    Args:
        timeseries_batch: (N, T, R) — N sujetos, T timesteps, R ROIs.

    Returns:
        vectors: (N, D) con D = R*(R-1)//2 (19900 para R=200).
    """
    if timeseries_batch.ndim != 3:
        raise ValueError(
            f"Se esperaba (N, T, R), recibido {timeseries_batch.shape}"
        )
    n_subjects = timeseries_batch.shape[0]
    ts_list = [timeseries_batch[i].T for i in range(n_subjects)]  # cada (R, T)

    measure = ConnectivityMeasure(
        kind="tangent",
        vectorize=True,
        discard_diagonal=True,
    )
    vectors = measure.fit_transform(ts_list)  # (N, D)
    return vectors.astype(np.float32)