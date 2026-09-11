# utils/splitters.py
"""
Módulo de partición de datos.

Tres estrategias, todas a nivel de SUJETO, para evitar fuga de información
entre conjuntos (crítico en fMRI, donde múltiples ventanas del mismo sujeto
podrían acabar en particiones distintas):

    1. K-fold estratificado por sujeto  → get_subject_level_fold_splits
    2. Leave-One-Site-Out (LOSO)         → get_loso_fold_splits
    3. train / val / test único          → get_subject_level_train_val_test_split

API estable: todas las funciones reciben arrays a nivel de MUESTRA
(labels, subject_indices, site_ids) y devuelven índices de muestras.
La garantía que ofrecen es que TODAS las muestras de un mismo sujeto
caen siempre en la misma partición.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


# ──────────────────────────────────────────────────────────────────────
# Helpers internos
# ──────────────────────────────────────────────────────────────────────

def _subject_to_label(labels: np.ndarray, subject_indices: np.ndarray) -> dict:
    """Devuelve un dict {sujeto: etiqueta} tomando la primera muestra de cada sujeto."""
    unique_subjects = np.unique(subject_indices)
    return {int(s): int(labels[subject_indices == s][0]) for s in unique_subjects}


def _safe_stratified_split(
    indices: np.ndarray,
    strat_labels: np.ndarray,
    test_size: float,
    seed: int,
):
    """
    train_test_split estratificado con fallback a split aleatorio.

    sklearn lanza ValueError si alguna clase tiene menos de 2 miembros.
    En datasets pequeños o muy desbalanceados (p.ej. LOSO con sitios con
    pocos sujetos de una clase), caemos a un split aleatorio sin estratificar
    para no romper la ejecución.
    """
    try:
        return train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=strat_labels
        )
    except ValueError:
        return train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=None
        )


def _build_fold_dict(
    subject_indices: np.ndarray,
    train_subjects: np.ndarray,
    val_subjects: np.ndarray,
    test_subjects: np.ndarray,
    extra: Optional[dict] = None,
) -> dict:
    """Construye el dict estándar de un fold a partir de sujetos por partición."""
    fold = {
        "train_idx": np.where(np.isin(subject_indices, train_subjects))[0],
        "val_idx":   np.where(np.isin(subject_indices, val_subjects))[0],
        "test_idx":  np.where(np.isin(subject_indices, test_subjects))[0],
        "train_subjects": train_subjects,
        "val_subjects":   val_subjects,
        "test_subjects":  test_subjects,
    }
    if extra:
        fold.update(extra)
    return fold


# ──────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────

def get_subject_level_fold_splits(
    labels: np.ndarray,
    subject_indices: np.ndarray,
    site_ids: Optional[np.ndarray] = None,
    n_splits: int = 5,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> list[dict]:
    """
    K-fold a nivel de sujeto, estratificado por etiqueta.

    Notas:
        - `site_ids` se conserva para downstream (p.ej. análisis por sitio o
          WeightedRandomSampler estratificado por (label, site)), pero NO se
          usa para construir las particiones (evita fallos cuando algún sitio
          tiene muy pocos sujetos de una clase).
        - Usa `StratifiedGroupKFold` con `groups=subject_indices`, lo cual
          garantiza que todas las muestras de un sujeto viajan juntas.

    Args:
        labels:          (n_samples,)  etiquetas a nivel de muestra.
        subject_indices: (n_samples,)  id de sujeto por muestra.
        site_ids:        (n_samples,)  opcional, sitio por muestra.
        n_splits:        nº de folds.
        val_ratio:       fracción de sujetos de train_val reservados para val.
        seed:            semilla de reproducibilidad.

    Returns:
        Lista de dicts con claves:
            train_idx, val_idx, test_idx,         (índices de muestras)
            train_subjects, val_subjects, test_subjects
    """
    unique_subjects = np.unique(subject_indices)
    subject_to_label = _subject_to_label(labels, subject_indices)
    subject_labels = np.array([subject_to_label[s] for s in unique_subjects])

    groups = subject_indices  # a nivel de muestra

    try:
        kfold = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        fold_iter = kfold.split(np.arange(len(labels)), labels, groups)
    except TypeError:
        # sklearn < 1.1 no soporta shuffle en StratifiedGroupKFold
        kfold = StratifiedGroupKFold(n_splits=n_splits)
        fold_iter = kfold.split(np.arange(len(labels)), labels, groups)

    splits: list[dict] = []
    for train_val_sample_idx, test_sample_idx in fold_iter:
        train_val_subjects = np.unique(subject_indices[train_val_sample_idx])
        test_subjects = np.unique(subject_indices[test_sample_idx])

        # Sub-split train_val → train / val a nivel de sujeto
        train_val_labels = np.array([subject_to_label[s] for s in train_val_subjects])
        n_val_subjects = min(
            max(1, int(len(train_val_subjects) * val_ratio)),
            len(train_val_subjects) - 1,
        )

        if n_val_subjects < 1:
            train_subjects = train_val_subjects
            val_subjects = np.array([], dtype=np.int64)
        else:
            train_idx_rel, val_idx_rel = _safe_stratified_split(
                np.arange(len(train_val_subjects)),
                train_val_labels,
                test_size=val_ratio,
                seed=seed,
            )
            train_subjects = train_val_subjects[train_idx_rel]
            val_subjects = train_val_subjects[val_idx_rel]

        splits.append(
            _build_fold_dict(
                subject_indices,
                train_subjects,
                val_subjects,
                test_subjects,
            )
        )

    return splits


def get_loso_fold_splits(
    labels: np.ndarray,
    subject_indices: np.ndarray,
    site_ids: np.ndarray,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> list[dict]:
    """
    Leave-One-Site-Out: un sitio como test, el resto como train.

    Args:
        labels:          (n_samples,)
        subject_indices: (n_samples,)
        site_ids:        (n_samples,)  obligatorio.
        val_ratio:       fracción de sujetos de train reservados para val.
        seed:            semilla.

    Returns:
        Lista de dicts, uno por sitio reservado, con las claves estándar
        más `test_site` (nombre del sitio reservado).
    """
    if site_ids is None:
        raise ValueError("LOSO requiere site_ids")

    unique_sites = np.unique(site_ids)
    subject_to_label = _subject_to_label(labels, subject_indices)
    subject_to_site = {
        int(s): site_ids[subject_indices == s][0]
        for s in np.unique(subject_indices)
    }

    splits: list[dict] = []
    for test_site in unique_sites:
        train_subjects_pool = np.array(
            [s for s in np.unique(subject_indices) if subject_to_site[s] != test_site],
            dtype=np.int64,
        )
        test_subjects = np.array(
            [s for s in np.unique(subject_indices) if subject_to_site[s] == test_site],
            dtype=np.int64,
        )

        if len(test_subjects) == 0:
            continue
        if len(train_subjects_pool) == 0:
            continue  # dataset con un solo sitio → LOSO indefinido

        # Sub-split del pool de entrenamiento → train / val
        pool_labels = np.array([subject_to_label[s] for s in train_subjects_pool])
        n_val = min(
            max(1, int(len(train_subjects_pool) * val_ratio)),
            len(train_subjects_pool) - 1,
        )

        if n_val < 1:
            train_subjects = train_subjects_pool
            val_subjects = np.array([], dtype=np.int64)
        else:
            train_idx_rel, val_idx_rel = _safe_stratified_split(
                np.arange(len(train_subjects_pool)),
                pool_labels,
                test_size=val_ratio,
                seed=seed,
            )
            train_subjects = train_subjects_pool[train_idx_rel]
            val_subjects = train_subjects_pool[val_idx_rel]

        splits.append(
            _build_fold_dict(
                subject_indices,
                train_subjects,
                val_subjects,
                test_subjects,
                extra={"test_site": str(test_site)},
            )
        )

    return splits


def get_subject_level_train_val_test_split(
    labels: np.ndarray,
    subject_indices: np.ndarray,
    site_ids: Optional[np.ndarray] = None,  # noqa: ARG001 (reservado para futuro)
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split único train/val/test a nivel de sujeto (para pretrain, etc.).

    Returns:
        (train_idx, val_idx, test_idx) — índices de muestras.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Los ratios deben sumar 1."

    unique_subjects = np.unique(subject_indices)
    subject_to_label = _subject_to_label(labels, subject_indices)
    subject_labels = np.array([subject_to_label[s] for s in unique_subjects])

    # 1) train vs (val + test)
    train_subj, temp_subj = _safe_stratified_split(
        unique_subjects,
        subject_labels,
        test_size=(val_ratio + test_ratio),
        seed=seed,
    )

    # 2) val vs test
    temp_labels = np.array([subject_to_label[s] for s in temp_subj])
    val_subj, test_subj = _safe_stratified_split(
        temp_subj,
        temp_labels,
        test_size=test_ratio / (val_ratio + test_ratio),
        seed=seed,
    )

    train_idx = np.where(np.isin(subject_indices, train_subj))[0]
    val_idx   = np.where(np.isin(subject_indices, val_subj))[0]
    test_idx  = np.where(np.isin(subject_indices, test_subj))[0]
    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Semilla global para reproducibilidad del test
    rng = np.random.default_rng(0)

    # ─── Dataset sintético: 60 sujetos, 3 sitios, 2 clases, 5 ventanas/sujeto
    n_subjects      = 60
    windows_per_sub = 5
    sites           = ["SITE_A", "SITE_B", "SITE_C"]

    subject_ids     = np.repeat(np.arange(n_subjects), windows_per_sub)
    subject_sites   = np.array([sites[i % len(sites)] for i in range(n_subjects)])
    subject_labels  = rng.integers(0, 2, size=n_subjects)

    site_ids = subject_sites[subject_ids]
    labels   = subject_labels[subject_ids]

    n_samples = len(subject_ids)
    print(f"Synthetic dataset: {n_samples} muestras, "
          f"{n_subjects} sujetos, {len(sites)} sitios, 2 clases.\n")

    # ─── Utilidades de validación ────────────────────────────────────────
    def check_coverage(fold: dict, tag: str) -> None:
        """Todos los sujetos deben aparecer exactamente una vez."""
        all_subj = np.concatenate([
            fold["train_subjects"], fold["val_subjects"], fold["test_subjects"]
        ])
        unique, counts = np.unique(all_subj, return_counts=True)
        assert len(unique) == len(np.unique(subject_ids)), \
            f"[{tag}] Cobertura incompleta: {len(unique)} != {len(np.unique(subject_ids))}"
        assert (counts == 1).all(), \
            f"[{tag}] Sujeto duplicado entre particiones."

    def check_no_leak(fold: dict, tag: str) -> None:
        """Ningún sujeto puede estar en dos particiones a la vez."""
        tr = set(fold["train_subjects"].tolist())
        va = set(fold["val_subjects"].tolist())
        te = set(fold["test_subjects"].tolist())
        assert not (tr & va), f"[{tag}] Leak train ∩ val"
        assert not (tr & te), f"[{tag}] Leak train ∩ test"
        assert not (va & te), f"[{tag}] Leak val ∩ test"

    def check_indices_match_subjects(fold: dict, tag: str) -> None:
        """Los índices de muestra deben coincidir con los sujetos esperados."""
        assert set(np.unique(subject_ids[fold["train_idx"]]).tolist()) == \
               set(fold["train_subjects"].tolist()), f"[{tag}] train_idx inconsistente"
        assert set(np.unique(subject_ids[fold["val_idx"]]).tolist()) == \
               set(fold["val_subjects"].tolist()), f"[{tag}] val_idx inconsistente"
        assert set(np.unique(subject_ids[fold["test_idx"]]).tolist()) == \
               set(fold["test_subjects"].tolist()), f"[{tag}] test_idx inconsistente"

    # ─── TEST 1: K-fold ─────────────────────────────────────────────────
    print("── TEST 1: get_subject_level_fold_splits ─────────────────────")
    folds = get_subject_level_fold_splits(
        labels, subject_ids, site_ids=site_ids, n_splits=5, val_ratio=0.15, seed=42
    )
    assert len(folds) == 5, "Se esperaban 5 folds."
    for i, fold in enumerate(folds):
        tag = f"KFOLD fold {i}"
        check_coverage(fold, tag)
        check_no_leak(fold, tag)
        check_indices_match_subjects(fold, tag)
    print(f"  ✓ {len(folds)} folds generados")
    print(f"  ✓ Sin fuga de sujetos entre train/val/test")
    print(f"  ✓ Cobertura completa de sujetos en cada fold")
    print(f"  ✓ Índices de muestra alineados con sujetos")
    print(f"  Ej. fold 0 → train={len(folds[0]['train_subjects'])} subj, "
          f"val={len(folds[0]['val_subjects'])} subj, "
          f"test={len(folds[0]['test_subjects'])} subj\n")

    # ─── TEST 2: LOSO ───────────────────────────────────────────────────
    print("── TEST 2: get_loso_fold_splits ──────────────────────────────")
    loso = get_loso_fold_splits(labels, subject_ids, site_ids, val_ratio=0.15, seed=42)
    assert len(loso) == len(sites), f"Se esperaban {len(sites)} folds LOSO."
    for i, fold in enumerate(loso):
        tag = f"LOSO {fold['test_site']}"
        check_coverage(fold, tag)
        check_no_leak(fold, tag)
        check_indices_match_subjects(fold, tag)
        # El test debe contener exactamente un sitio
        test_sites_in_fold = np.unique(site_ids[fold["test_idx"]])
        assert len(test_sites_in_fold) == 1, \
            f"[{tag}] test_idx contiene más de un sitio: {test_sites_in_fold}"
        assert test_sites_in_fold[0] == fold["test_site"], \
            f"[{tag}] test_site inconsistente."
    print(f"  ✓ {len(loso)} folds LOSO generados")
    print(f"  ✓ Cada fold reserva exactamente un sitio como test")
    print(f"  ✓ Sin fuga de sujetos entre particiones")
    for fold in loso:
        print(f"    · test_site={fold['test_site']:>8s} "
              f"→ train={len(fold['train_subjects'])} "
              f"val={len(fold['val_subjects'])} "
              f"test={len(fold['test_subjects'])}")
    print()

    # ─── TEST 3: train/val/test único ───────────────────────────────────
    print("── TEST 3: get_subject_level_train_val_test_split ───────────")
    tr, va, te = get_subject_level_train_val_test_split(
        labels, subject_ids, site_ids=site_ids,
        train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42,
    )
    tr_s = set(np.unique(subject_ids[tr]).tolist())
    va_s = set(np.unique(subject_ids[va]).tolist())
    te_s = set(np.unique(subject_ids[te]).tolist())
    assert not (tr_s & va_s) and not (tr_s & te_s) and not (va_s & te_s), \
        "Leak entre particiones."
    assert len(tr_s | va_s | te_s) == n_subjects, "Cobertura incompleta."
    total_ratio = (len(tr) + len(va) + len(te)) / n_samples
    assert abs(total_ratio - 1.0) < 1e-9, "Los índices no cubren todas las muestras."
    print(f"  ✓ train={len(tr_s)} subj ({len(tr)} muestras)")
    print(f"  ✓ val  ={len(va_s)} subj ({len(va)} muestras)")
    print(f"  ✓ test ={len(te_s)} subj ({len(te)} muestras)")
    print(f"  ✓ Sin fuga, cobertura completa\n")

    # ─── TEST 4: caso extremo, clases muy pequeñas ──────────────────────
    print("── TEST 4: caso extremo (clase minoritaria < 2) ──────────────")
    small_labels   = np.array([0, 0, 0, 0, 0, 1])   # 1 sola muestra de clase 1
    small_subjects = np.arange(len(small_labels))
    try:
        folds_small = get_subject_level_fold_splits(
            small_labels, small_subjects, n_splits=3, val_ratio=0.3, seed=0
        )
        print(f"  ✓ No crashea: {len(folds_small)} folds generados")

        # Validaciones locales (no usar el dataset grande)
        all_small = np.concatenate([
            f["train_subjects"] for f in folds_small
        ] + [
            f["test_subjects"] for f in folds_small
        ])
        unique_small = np.unique(all_small)
        assert len(unique_small) == len(small_subjects), \
            f"[small] Cobertura incompleta: {len(unique_small)} != {len(small_subjects)}"

        for f in folds_small:
            tr = set(f["train_subjects"].tolist())
            va = set(f["val_subjects"].tolist())
            te = set(f["test_subjects"].tolist())
            assert not (tr & va) and not (tr & te) and not (va & te), \
                "[small] Leak entre particiones."
        print("  ✓ Cobertura completa y sin fuga")
    except Exception as e:
        print(f"  ✗ Falló: {e}")
        sys.exit(1)
    print()

    print("✅ Todos los tests de splitters.py pasaron.")