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


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ─── TEST 1: compute_metrics con caso balanceado conocido ──────────
    print("── TEST 1: compute_metrics (caso balanceado) ─────────────────")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.3, 0.4, 0.7, 0.1])

    m = compute_metrics(y_true, y_pred, y_prob)
    # Manual: TN=3, FP=1, FN=1, TP=3
    #   ACC = 6/8 = 0.75
    #   Sens = 3/4 = 0.75
    #   Spec = 3/4 = 0.75
    assert abs(m["accuracy"] - 0.75) < 1e-9, m
    assert abs(m["sensitivity"] - 0.75) < 1e-9, m
    assert abs(m["specificity"] - 0.75) < 1e-9, m
    assert 0.0 < m["auc"] < 1.0, m
    assert 0.0 < m["f1"] <= 1.0, m
    print(f"  ✓ ACC={m['accuracy']:.3f}  Sens={m['sensitivity']:.3f}  "
          f"Spec={m['specificity']:.3f}  AUC={m['auc']:.3f}  F1={m['f1']:.3f}\n")

    # ─── TEST 2: compute_metrics con una sola clase en y_true ──────────
    print("── TEST 2: compute_metrics (una sola clase en y_true) ────────")
    y_true_single = np.array([1, 1, 1, 1])
    y_pred_single = np.array([1, 1, 0, 1])
    m2 = compute_metrics(y_true_single, y_pred_single,
                         y_prob=np.array([0.9, 0.8, 0.4, 0.7]))
    assert m2["auc"] == 0.0, "AUC debe ser 0.0 si solo hay una clase."
    assert m2["sensitivity"] == 0.75, m2   # TP=3, FN=1
    assert m2["specificity"] == 0.0, m2    # sin clase 0 → sin TN ni FP
    print(f"  ✓ AUC=0.0 (sin clase negativa), Sens={m2['sensitivity']:.3f}, "
          f"Spec={m2['specificity']:.3f}\n")

    # ─── TEST 3: bootstrap CI sobre datos constantes ───────────────────
    print("── TEST 3: bootstrap_confidence_interval ─────────────────────")
    vals = np.array([0.70, 0.72, 0.74, 0.76, 0.78])
    mean, std, lo, hi = bootstrap_confidence_interval(vals, n_bootstrap=2000, seed=42)
    assert abs(mean - 0.74) < 1e-9, mean
    assert lo <= mean <= hi, (lo, mean, hi)
    assert hi - lo < 0.10, "CI demasiado ancho para datos tan concentrados."
    print(f"  ✓ mean={mean:.4f}  std={std:.4f}  CI95=[{lo:.4f}, {hi:.4f}]")

    # Caso vacío
    m0, s0, l0, h0 = bootstrap_confidence_interval(np.array([]))
    assert (m0, s0, l0, h0) == (0.0, 0.0, 0.0, 0.0)
    print(f"  ✓ Caso vacío → (0,0,0,0)\n")

    # ─── TEST 4: agregación a nivel de sujeto ──────────────────────────
    print("── TEST 4: aggregate_window_predictions_to_subject_level ─────")
       # 3 sujetos, 2 ventanas cada uno
    #   Sujeto 10: true=1, preds=[1,1] → mean=1.0  → 1
    #   Sujeto 20: true=0, preds=[0,1] → mean=0.5  → round(0.5)=0 (empate cae a 0)
    #   Sujeto 30: true=1, preds=[0,0] → mean=0.0  → 0
    subject_indices = np.array([10, 10, 20, 20, 30, 30])
    test_sample_indices = np.array([0, 1, 2, 3, 4, 5])
    y_true_w = np.array([1, 1, 0, 0, 1, 1])
    y_pred_w = np.array([1, 1, 0, 1, 0, 0])
    y_prob_w = np.array([0.9, 0.8, 0.3, 0.6, 0.4, 0.3])

    yt_mv, yp_mv, yprob_mv = aggregate_window_predictions_to_subject_level(
        y_true_w, y_pred_w, y_prob_w,
        test_sample_indices, subject_indices,
        strategy="majority_vote",
    )
    assert len(yt_mv) == 3, f"Esperados 3 sujetos, hay {len(yt_mv)}"
    assert np.array_equal(yt_mv, [1, 0, 1]), yt_mv
    assert np.array_equal(yp_mv, [1, 0, 0]), yp_mv    # ← corregido
    print(f"  ✓ majority_vote: y_true={yt_mv.tolist()}  y_pred={yp_mv.tolist()}")
    
    # ─── TEST 5: reproducibilidad no rompe ─────────────────────────────
    print("── TEST 5: get_reproducibility_info ──────────────────────────")
    info = get_reproducibility_info()
    assert "numpy" in info and "pytorch" in info
    print(f"  ✓ keys: {sorted(info.keys())}\n")

    print("✅ Todos los tests de metrics.py pasaron.")