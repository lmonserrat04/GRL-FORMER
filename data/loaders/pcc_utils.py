import torch

def compute_pcc_vector(timeseries: torch.Tensor) -> torch.Tensor:
    """
    Calcula el vector PCC (triangular superior) a partir de una serie temporal.
    Args:
        timeseries: tensor de forma (N_ROIS, T)
    Returns:
        pcc_vector: tensor de forma (N_ROIS*(N_ROIS-1)//2,)
    """
    # Correlación de Pearson: (N_ROIS, T) -> (N_ROIS, N_ROIS)
    corr = torch.corrcoef(timeseries)  # correlación entre filas (ROIs)
    # Obtener índices del triángulo superior sin diagonal
    triu_idx = torch.triu_indices(corr.shape[0], corr.shape[1], offset=1)
    pcc_vector = corr[triu_idx[0], triu_idx[1]]
    return pcc_vector