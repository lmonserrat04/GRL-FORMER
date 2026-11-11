"""Samplers para la fase contrastiva.

``DiagnosisSiteBalancedSampler`` construye batches con la regla del
experimento de alineación por diagnóstico entre sitios:

  - Cada batch tiene ``batch_size // 2`` sujetos por clase (16 ASD / 16 TD
    con batch 32).
  - Cada clase aporta sujetos de al menos ``min_sites_per_class`` sitios
    distintos (4 por defecto).
  - Los sujetos de una misma celda (diagnóstico, sitio) pueden repetirse
    entre batches, pero no dentro de un batch mientras haya suficientes
    sujetos de esa clase.

La garantía de >= k sitios por clase es lo que asegura que casi todo
ancla tenga al menos un positivo válido (mismo diagnóstico, distinto
sitio) en el batch.
"""

from __future__ import annotations

import math

import numpy as np
from torch.utils.data import Sampler


class DiagnosisSiteBalancedSampler(Sampler[list[int]]):
    """Batch sampler balanceado por diagnóstico y cobertura de sitios.

    Args:
        labels: etiquetas (n,) del split de entrenamiento.
        sites: identificadores de sitio (n,), mismo orden que ``labels``.
        batch_size: tamaño de batch (debe ser par).
        min_sites_per_class: mínimo de sitios distintos por clase y batch.
        seed: semilla base del muestreo.
        drop_last: si True, el número de batches por época es
            ``len(labels) // batch_size``; si False, ``ceil``.
        max_resample: reintentos si un batch no cumple la cobertura mínima.
    """

    def __init__(
        self,
        labels,
        sites,
        batch_size: int = 32,
        min_sites_per_class: int = 4,
        seed: int = 42,
        drop_last: bool = True,
        max_resample: int = 50,
    ):
        labels = np.asarray(labels)
        sites = np.asarray(sites)

        if len(labels) != len(sites):
            raise ValueError("labels y sites deben tener la misma longitud.")
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError("batch_size debe ser par y >= 2.")
        if min_sites_per_class < 1:
            raise ValueError("min_sites_per_class debe ser >= 1.")

        self.labels = labels
        self.sites = sites
        self.batch_size = int(batch_size)
        self.per_class = batch_size // 2
        self.min_sites_per_class = int(min_sites_per_class)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.max_resample = int(max_resample)
        self.epoch = 0

        self.classes = np.unique(labels)
        if len(self.classes) != 2:
            raise ValueError(
                f"Se esperaban 2 clases, hay {len(self.classes)}: {self.classes}"
            )

        # Índices por clase y por (clase, sitio)
        self.class_indices = {
            int(c): np.flatnonzero(labels == c) for c in self.classes
        }
        self.class_sites: dict[int, np.ndarray] = {}
        self.class_site_indices: dict[int, dict[str, np.ndarray]] = {}
        for c in self.classes:
            c = int(c)
            mask_c = labels == c
            sites_c = np.unique(sites[mask_c])
            self.class_sites[c] = sites_c
            self.class_site_indices[c] = {
                str(s): np.flatnonzero(mask_c & (sites == s)) for s in sites_c
            }

        if self.drop_last:
            n_batches = len(labels) // self.batch_size
        else:
            n_batches = math.ceil(len(labels) / self.batch_size)
        self.n_batches = max(1, int(n_batches))

    def set_epoch(self, epoch: int) -> None:
        """Cambia la semilla efectiva por época (uso con DataLoader)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def _sample_class(self, c: int, rng: np.random.Generator) -> list[int]:
        """Muestrea ``per_class`` sujetos de la clase ``c`` con >= k sitios."""
        sites_c = self.class_sites[c]
        n_available_sites = len(sites_c)
        k = min(self.min_sites_per_class, n_available_sites)

        # Sitios elegidos para garantizar cobertura. Si hay más sitios que
        # el mínimo, se añaden algunos extra de vez en cuando para no
        # congelar siempre los mismos.
        if n_available_sites > k:
            extra = int(rng.integers(0, min(3, n_available_sites - k) + 1))
            k_eff = min(k + extra, n_available_sites)
        else:
            k_eff = n_available_sites
        chosen_sites = rng.choice(sites_c, size=k_eff, replace=False)

        selected: list[int] = []
        selected_set: set[int] = set()

        # Un sujeto por sitio elegido: cobertura garantizada.
        for s in chosen_sites:
            pool = self.class_site_indices[c][str(s)]
            idx = int(rng.choice(pool))
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

        # Rellenar el resto desde toda la clase, sin duplicar en el batch.
        pool_all = self.class_indices[c]
        attempts = 0
        max_attempts = self.per_class * 50
        while len(selected) < self.per_class and attempts < max_attempts:
            attempts += 1
            idx = int(rng.choice(pool_all))
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)

        # Fallback para clases con muy pocos sujetos: permitir repetición.
        while len(selected) < self.per_class:
            selected.append(int(rng.choice(pool_all)))

        return selected

    def _make_batch(self, rng: np.random.Generator) -> list[int]:
        batch: list[int] = []
        for c in self.classes:
            batch.extend(self._sample_class(int(c), rng))
        rng.shuffle(batch)
        return batch

    def _sites_per_class(self, batch: list[int]) -> dict[int, int]:
        counts: dict[int, int] = {}
        for c in self.classes:
            c = int(c)
            mask = self.labels[batch] == c
            counts[c] = int(len(np.unique(self.sites[np.asarray(batch)[mask]])))
        return counts

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 100_003)
        for _ in range(self.n_batches):
            batch = None
            for _try in range(self.max_resample):
                candidate = self._make_batch(rng)
                counts = self._sites_per_class(candidate)
                ok = all(
                    counts[int(c)] >= min(
                        self.min_sites_per_class,
                        len(self.class_sites[int(c)]),
                    )
                    for c in self.classes
                )
                if ok:
                    batch = candidate
                    break
            if batch is None:
                # No se pudo cumplir el criterio (p. ej. clase con muy
                # pocos sitios). Se usa el último candidato; la loss
                # simplemente tendrá menos positivos para esas anclas.
                batch = candidate
            yield batch


if __name__ == "__main__":
    # Test rápido con datos sintéticos: dos clases, sitios desbalanceados.
    rng = np.random.default_rng(0)
    n = 400
    labs = np.array([0] * 200 + [1] * 200)
    sites = np.array(
        [f"S{i % 8}" for i in range(200)] + [f"S{i % 5}" for i in range(200)]
    )
    sampler = DiagnosisSiteBalancedSampler(
        labs, sites, batch_size=32, min_sites_per_class=4, seed=7
    )
    assert len(sampler) == n // 32
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for batch in sampler:
            assert len(batch) == 32
            b = np.asarray(batch)
            for c in (0, 1):
                n_sites = len(np.unique(sites[b[labs[b] == c]]))
                assert n_sites >= 4, (epoch, c, n_sites)
    print("OK: DiagnosisSiteBalancedSampler — batches 16/16 con >=4 sitios por clase.")
