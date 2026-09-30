"""
Ponderación de la pérdida por (sitio, clase).

Motivación:
    En ABIDE, la proporción ASD/TD varía por sitio (29.6% a 64.2%).
    El dataset global está balanceado (~54/46) pero cada sitio
    individual no. Con CrossEntropy plana, los sitios con clases
    desbalanceadas contribuyen de forma no uniforme al gradiente.

    Al ponderar por (sitio, clase) forzamos:
      1. Que cada sitio contribuya por igual al gradiente.
      2. Que dentro de cada sitio, cada clase contribuya por igual.

Fórmula:
    weight(s, c) = (1 / count(s, c)) ^ power

Normalización:
    - "site":   media de weights dentro de cada sitio = 1
    - "global": media de todos los weights > 0 = 1
    - "none":   sin normalizar

Uso:
    W = compute_site_class_weight_matrix(
        labels=train_labels,
        site_ids=train_site_ids,
        num_sites=19,
        num_classes=2,
        power=1.0,
        normalize="site",
    )
    # W.shape = (19, 2)  → W[site_id, class_id] = weight
    # En el trainer:
    #   W_tensor = torch.from_numpy(W).to(device)
    #   sample_w = W_tensor[site_batch, y_batch]   # (B,)
"""
from __future__ import annotations

import numpy as np


def compute_site_class_weight_matrix(
    labels: np.ndarray,
    site_ids: np.ndarray,
    num_sites: int,
    num_classes: int = 2,
    power: float = 1.0,
    normalize: str = "site",
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Devuelve W: (num_sites, num_classes) float32.

    Args:
        labels:       (N,) int con clase de cada muestra del split de train.
        site_ids:     (N,) int con índice de sitio (ya mapeado).
        num_sites:    nº de sitios totales (len(site_to_idx)).
        num_classes:  nº de clases (2 en tu caso).
        power:        exponente. 0 → uniforme, 1 → inversa frecuencia.
        normalize:    "site" | "global" | "none".
    """
    labels = np.asarray(labels, dtype=np.int64)
    site_ids = np.asarray(site_ids, dtype=np.int64)
    assert labels.shape == site_ids.shape, "labels y site_ids deben tener mismo shape"

    W = np.zeros((num_sites, num_classes), dtype=np.float32)

    for s in range(num_sites):
        mask_s = (site_ids == s)
        if mask_s.sum() == 0:
            continue
        for c in range(num_classes):
            n = int(((labels == c) & mask_s).sum())
            if n > 0:
                W[s, c] = (1.0 / n) ** power

    if normalize == "site":
        for s in range(num_sites):
            row = W[s, :]
            nz = row[row > 0]
            if len(nz) > 0 and nz.mean() > epsilon:
                W[s, :] = row / nz.mean()
    elif normalize == "global":
        nz = W[W > 0]
        if len(nz) > 0 and nz.mean() > epsilon:
            W = W / nz.mean()
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"normalize desconocido: {normalize!r}")

    return W


def summarize_weight_matrix(W: np.ndarray) -> str:
    """Devuelve un string legible para debugging."""
    lines = ["=== Weight matrix (site x class) ==="]
    for s in range(W.shape[0]):
        row = "  ".join(f"{W[s, c]:6.3f}" for c in range(W.shape[1]))
        lines.append(f"  site {s:>2d}: [{row}]")
    return "\n".join(lines)
