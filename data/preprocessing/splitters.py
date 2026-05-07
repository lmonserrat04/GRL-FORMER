"""
Módulo de partición de datos
Implementa la partición a nivel de sujeto y K-fold estratificado por sitio para evitar la fuga de información causada por ventanas deslizantes.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def get_subject_level_fold_splits(
    labels,
    subject_indices,
    site_ids=None,
    n_splits=5,
    val_ratio=0.15,
    seed=42,
):
    """
    Genera particiones K-fold a nivel de sujeto y estratificadas por sitio (opcional).
    Todas las muestras de un mismo sujeto (incluidas las ventanas deslizantes) aparecerán únicamente en uno de los conjuntos: train, val o test.

    Args:
        labels: ndarray (n_samples,), etiquetas de las muestras.
        subject_indices: ndarray (n_samples,), índice del sujeto correspondiente a cada muestra.
        site_ids: ndarray (n_samples,), ID del sitio correspondiente a cada muestra, usado para estratificación por sitio (opcional).
        n_splits: número de pliegues (folds).
        val_ratio: proporción para dividir el conjunto de validación desde el conjunto de entrenamiento.
        seed: semilla aleatoria.

    Returns:
        splits: list of dict, cada pliegue contiene:
            - train_idx: índices de muestras de entrenamiento.
            - val_idx: índices de muestras de validación.
            - test_idx: índices de muestras de prueba.
            - train_subjects: índices de sujetos en el conjunto de entrenamiento.
            - test_subjects: índices de sujetos en el conjunto de prueba.
    """
    unique_subjects = np.unique(subject_indices)
    n_subjects = len(unique_subjects)

    # Se toma una muestra por cada sujeto para obtener el mapeo sujeto -> etiqueta.
    subject_to_label = {}
    subject_to_site = {}
    for s in unique_subjects:
        mask = subject_indices == s
        subject_to_label[s] = int(labels[mask][0])
        if site_ids is not None:
            subject_to_site[s] = site_ids[mask][0]

    subject_labels = np.array([subject_to_label[s] for s in unique_subjects])
    subject_sites = np.array([subject_to_site.get(s, "unknown") for s in unique_subjects]) if site_ids is not None else None

    # Uso de StratifiedGroupKFold: groups=subject_indices equivale a dividir por sujeto.
    # Es necesario pasar los grupos a nivel de muestra; las muestras de un mismo sujeto tienen el mismo group id.
    groups = subject_indices

    try:
        kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_iter = kfold.split(np.arange(len(labels)), labels, groups)
    except TypeError:
        # Si sklearn < 0.24 no tiene StratifiedGroupKFold, se degrada a partición manual por sujeto.
        fold_iter = _fallback_subject_stratified_kfold(
            unique_subjects, subject_labels, n_splits, seed
        )
        # fold_iter genera (train_subject_indices, test_subject_indices).
        # Es necesario convertirlo a índices de muestras.
        fold_iter = _convert_subject_splits_to_sample_splits(
            fold_iter, subject_indices, unique_subjects
        )

    splits = []
    for fold_idx, (train_val_sample_idx, test_sample_idx) in enumerate(fold_iter):
        train_val_subjects = np.unique(subject_indices[train_val_sample_idx])
        test_subjects = np.unique(subject_indices[test_sample_idx])

        # Dividir train_val en train / val, manteniendo el nivel de sujeto.
        train_val_labels_subj = np.array([subject_to_label[s] for s in train_val_subjects])
        n_val_subjects = max(1, int(len(train_val_subjects) * val_ratio))
        n_val_subjects = min(n_val_subjects, len(train_val_subjects) - 1)

        if n_val_subjects < 1:
            train_subjects = train_val_subjects
            val_subjects = np.array([], dtype=np.int64)
        else:
            try:
                train_subj_idx, val_subj_idx = train_test_split(
                    np.arange(len(train_val_subjects)),
                    test_size=val_ratio,
                    random_state=seed,
                    stratify=train_val_labels_subj,
                )
                train_subjects = train_val_subjects[train_subj_idx]
                val_subjects = train_val_subjects[val_subj_idx]
            except ValueError:
                train_subjects = train_val_subjects
                val_subjects = np.array([], dtype=np.int64)

        train_idx = np.where(np.isin(subject_indices, train_subjects))[0]
        val_idx = np.where(np.isin(subject_indices, val_subjects))[0]

        splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_sample_idx,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        })

    return splits


def _fallback_subject_stratified_kfold(unique_subjects, subject_labels, n_splits, seed):
    """Realiza K-fold estratificado por sujeto cuando StratifiedGroupKFold no está disponible."""
    from sklearn.model_selection import StratifiedKFold

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(unique_subjects, subject_labels):
        yield unique_subjects[train_idx], unique_subjects[test_idx]


def _convert_subject_splits_to_sample_splits(fold_iter, subject_indices, unique_subjects):
    """Convierte las particiones a nivel de sujeto en particiones a nivel de muestra."""
    for train_subjects, test_subjects in fold_iter:
        train_val_sample_idx = np.where(np.isin(subject_indices, train_subjects))[0]
        test_sample_idx = np.where(np.isin(subject_indices, test_subjects))[0]
        yield train_val_sample_idx, test_sample_idx


def get_loso_fold_splits(
    labels,
    subject_indices,
    site_ids,
    val_ratio=0.15,
    seed=42,
):
    """
    Partición Leave-One-Site-Out (LOSO): en cada iteración se reserva un sitio como conjunto de prueba y el resto como entrenamiento.
    Se utiliza para evaluar la capacidad de generalización entre sitios.

    Args:
        labels: (n_samples,)
        subject_indices: (n_samples,)
        site_ids: (n_samples,), ID de los sitios.
        val_ratio: proporción para dividir el conjunto de validación (por sujeto).
        seed: semilla aleatoria.

    Returns:
        splits: list of dict, donde cada pliegue corresponde a un sitio reservado.
    """
    if site_ids is None:
        raise ValueError("LOSO requiere site_ids")

    unique_sites = np.unique(site_ids)
    subject_to_label = {s: int(labels[subject_indices == s][0]) for s in np.unique(subject_indices)}
    subject_to_site = {s: site_ids[subject_indices == s][0] for s in np.unique(subject_indices)}

    splits = []
    for test_site in unique_sites:
        train_val_subjects = np.array([s for s in np.unique(subject_indices) if subject_to_site[s] != test_site])
        test_subjects = np.array([s for s in np.unique(subject_indices) if subject_to_site[s] == test_site])

        if len(test_subjects) == 0:
            continue

        # Dividir train_val para obtener val.
        train_val_labels = np.array([subject_to_label[s] for s in train_val_subjects])
        n_val = max(1, int(len(train_val_subjects) * val_ratio))
        n_val = min(n_val, len(train_val_subjects) - 1)
        if n_val < 1:
            train_subjects = train_val_subjects
            val_subjects = np.array([], dtype=np.int64)
        else:
            try:
                train_idx, val_idx = train_test_split(
                    np.arange(len(train_val_subjects)),
                    test_size=val_ratio,
                    random_state=seed,
                    stratify=train_val_labels,
                )
                train_subjects = train_val_subjects[train_idx]
                val_subjects = train_val_subjects[val_idx]
            except ValueError:
                train_subjects = train_val_subjects
                val_subjects = np.array([], dtype=np.int64)

        train_idx = np.where(np.isin(subject_indices, train_subjects))[0]
        val_idx = np.where(np.isin(subject_indices, val_subjects))[0]
        test_idx = np.where(np.isin(subject_indices, test_subjects))[0]

        splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "test_site": str(test_site),
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        })

    return splits


def get_subject_level_train_val_test_split(
    labels,
    subject_indices,
    site_ids=None,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    seed=42,
):
    """
    Partición única de train/val/test a nivel de sujeto (utilizada para pre-entrenamiento, etc.).

    Args:
        labels: (n_samples,)
        subject_indices: (n_samples,)
        site_ids: (n_samples,), opcional.
        train_ratio, val_ratio, test_ratio: proporciones, deben sumar 1.
        seed: semilla aleatoria.

    Returns:
        train_idx, val_idx, test_idx: índices de las muestras.
    """
    unique_subjects = np.unique(subject_indices)
    subject_to_label = {s: int(labels[subject_indices == s][0]) for s in unique_subjects}
    subject_labels = np.array([subject_to_label[s] for s in unique_subjects])

    # Primero dividir entre train y (val+test).
    train_subj, temp_subj = train_test_split(
        unique_subjects,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=subject_labels,
    )
    temp_labels = np.array([subject_to_label[s] for s in temp_subj])
    val_subj, test_subj = train_test_split(
        temp_subj,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=temp_labels,
    )

    train_idx = np.where(np.isin(subject_indices, train_subj))[0]
    val_idx = np.where(np.isin(subject_indices, val_subj))[0]
    test_idx = np.where(np.isin(subject_indices, test_subj))[0]

    return train_idx, val_idx, test_idx