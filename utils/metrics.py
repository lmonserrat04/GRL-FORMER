# utils/metrics.py
"""
Métricas de clasificación, intervalos de confianza bootstrap y agregación
de predicciones de ventana a nivel de sujeto.

Métricas reportadas (las del paper): AUC, ACC, Sensitivity, Specificity, F1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# ──────────────────────────────────────────────────────────────────────
# Métricas de clasificación
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict:
    """
    Calcula las 5 métricas del paper.

    Args:
        y_true: etiquetas reales (n,)
        y_pred: predicciones (n,)
        y_prob: probabilidad de clase positiva (n,), necesaria para AUC

    Returns:
        dict con 'auc', 'accuracy', 'sensitivity', 'specificity', 'f1'.
        'auc' = 0.0 si y_true es de una sola clase.
    """
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
    }

        # AUC (requiere ambas clases presentes)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0

    # Sensitivity / Specificity desde la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        metrics["sensitivity"] = 0.0
        metrics["specificity"] = 0.0

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────────────────────────────

def bootstrap_confidence_interval(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> tuple[float, float, float, float]:
    """
    Bootstrap CI sobre una lista de métricas (una por fold/semilla).

    Returns:
        (mean, std, ci_lower, ci_upper)
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = values[idx].mean(axis=1)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(values.mean()), float(values.std()), lower, upper


# ──────────────────────────────────────────────────────────────────────
# Agregación ventana → sujeto
# ──────────────────────────────────────────────────────────────────────

def aggregate_window_predictions_to_subject_level(
    y_true_window: np.ndarray,
    y_pred_window: np.ndarray,
    y_prob_window: np.ndarray,
    test_sample_indices: np.ndarray,
    subject_indices: np.ndarray,
    strategy: str = "majority_vote",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce predicciones por ventana a predicciones por sujeto.

    Estrategias:
        - 'majority_vote': voto mayoritario de las etiquetas predichas.
        - 'prob_mean':     media de las probabilidades, umbral 0.5.

    Args:
        y_true_window, y_pred_window, y_prob_window: (n_windows,) alineadas
            con test_sample_indices.
        test_sample_indices: (n_windows,) índices de muestra del test set.
        subject_indices:     (n_samples,) id de sujeto por muestra.
        strategy: 'majority_vote' o 'prob_mean'.

    Returns:
        (y_true_subj, y_pred_subj, y_prob_subj)
    """
    assert strategy in ("majority_vote", "prob_mean"), \
        f"strategy desconocida: {strategy}"

    test_subjects = subject_indices[test_sample_indices]
    unique_subjects = np.unique(test_subjects)

    y_true_w = np.asarray(y_true_window)
    y_pred_w = np.asarray(y_pred_window)
    y_prob_w = np.asarray(y_prob_window)

    y_true_s, y_pred_s, y_prob_s = [], [], []
    for s in unique_subjects:
        mask = test_subjects == s
        true_label = int(y_true_w[mask][0])       # misma etiqueta en todas las ventanas
        probs = y_prob_w[mask]

        if strategy == "prob_mean":
            prob_subj = float(probs.mean())
            pred_subj = int(prob_subj >= 0.5)
        else:  # majority_vote
            pred_subj = int(round(y_pred_w[mask].mean()))
            prob_subj = float(probs.mean())

        y_true_s.append(true_label)
        y_pred_s.append(pred_subj)
        y_prob_s.append(prob_subj)

    return (
        np.array(y_true_s),
        np.array(y_pred_s),
        np.array(y_prob_s),
    )


# ──────────────────────────────────────────────────────────────────────
# Reproducibilidad
# ──────────────────────────────────────────────────────────────────────

def get_reproducibility_info() -> dict:
    """Metadata de entorno para guardar junto a los resultados."""
    info = {"numpy": np.__version__}
    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["pytorch"] = "not installed"
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except ImportError:
        info["sklearn"] = "not installed"
    return info


