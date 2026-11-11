"""Samplers para la fase contrastiva.

``DiagnosisStratifiedSampler`` construye batches de ``batch_size`` con
muestreo estratificado por diagnóstico según la prevalencia de la cohorte
(sin conteos fijos). Cada batch se valida y, si falla, se vuelve a
muestrear hasta cumplir:

  1. ambas diagnosis están presentes;
  2. cada diagnóstico aporta sujetos de al menos ``min_sites_per_class``
     sitios distintos;
  3. cada ancla tiene al menos un positivo válido (mismo diagnóstico,
     sitio distinto).

La condición 3 se cumple automáticamente si se cumple la 2, pero se
verifica explícitamente porque es la propiedad que necesita la loss.
"""

from __future__ import annotations

import math

import numpy as np
from torch.utils.data import Sampler


class DiagnosisStratifiedSampler(Sampler[list[int]]):
    """Batch sampler estratificado por diagnóstico con validación de sitios.

    Args:
        labels: etiquetas (n,) del split.
        sites: identificadores de sitio (n,), mismo orden que ``labels``.
        batch_size: tamaño de batch.
        min_sites_per_class: mínimo de sitios distintos por diagnóstico
            en cada batch.
        seed: semilla base del muestreo.
        drop_last: si True, el número de batches por época es
            ``len(labels) // batch_size``; si False, ``ceil``.
        max_resample: reintentos antes de abortar si no se logra un batch
            válido.
    """

    def __init__(
        self,
        labels,
        sites,
        batch_size: int = 32,
        min_sites_per_class: int = 4,
        seed: int = 42,
        drop_last: bool = True,
        max_resample: int = 100,
    ):
        labels = np.asarray(labels)
        sites = np.asarray(sites)

        if len(labels) != len(sites):
            raise ValueError("labels y sites deben tener la misma longitud.")
        if batch_size < 2:
            raise ValueError("batch_size debe ser >= 2.")
        if min_sites_per_class < 1:
            raise ValueError("min_sites_per_class debe ser >= 1.")

        self.labels = labels
        self.sites = sites
        self.batch_size = int(batch_size)
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

        self.class_indices = {
            int(c): np.flatnonzero(labels == c) for c in self.classes
        }
        self.class_counts = np.array(
            [len(self.class_indices[int(c)]) for c in self.classes], dtype=float
        )
        for i, c in enumerate(self.classes):
            if self.class_counts[i] == 0:
                raise ValueError(f"La clase {int(c)} no tiene sujetos.")
            n_sites = len(np.unique(sites[labels == c]))
            if n_sites < self.min_sites_per_class:
                raise ValueError(
                    f"La clase {int(c)} solo tiene {n_sites} sitios; se "
                    f"requieren >= {self.min_sites_per_class} por batch."
                )

        self.class_probs = self.class_counts / self.class_counts.sum()

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

    def _candidate(self, rng: np.random.Generator) -> list[int] | None:
        """Un batch candidato con conteos ~ Multinomial(batch_size, p_clase)."""
        counts = rng.multinomial(self.batch_size, self.class_probs)
        batch: list[int] = []
        for i, c in enumerate(self.classes):
            n_c = int(counts[i])
            if n_c == 0:
                return None
            pool = self.class_indices[int(c)]
            if n_c > len(pool):
                return None
            batch.extend(rng.choice(pool, size=n_c, replace=False).tolist())
        rng.shuffle(batch)
        return batch

    def _is_valid(self, batch: list[int]) -> bool:
        b = np.asarray(batch)
        labs = self.labels[b]
        site_b = self.sites[b]

        # 1) Ambas diagnosis presentes.
        if len(np.unique(labs)) != len(self.classes):
            return False

        # 2) Cobertura mínima de sitios por diagnóstico.
        for c in self.classes:
            mask = labs == c
            if len(np.unique(site_b[mask])) < self.min_sites_per_class:
                return False

        # 3) Cada ancla tiene al menos un positivo válido.
        for i in range(len(b)):
            has_positive = np.any((labs == labs[i]) & (site_b != site_b[i]))
            if not has_positive:
                return False

        return True

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 100_003)
        for _ in range(self.n_batches):
            batch = None
            for _try in range(self.max_resample):
                candidate = self._candidate(rng)
                if candidate is not None and self._is_valid(candidate):
                    batch = candidate
                    break
            if batch is None:
                raise RuntimeError(
                    f"No se pudo construir un batch válido tras "
                    f"{self.max_resample} intentos (epoch {self.epoch})."
                )
            yield batch


if __name__ == "__main__":
    # Test con cohorte desbalanceada: 75% clase 0, 25% clase 1.
    rng = np.random.default_rng(0)
    n = 400
    labs = np.array([0] * 300 + [1] * 100)
    sites = np.array(
        [f"S{i % 8}" for i in range(300)] + [f"S{i % 5}" for i in range(100)]
    )
    sampler = DiagnosisStratifiedSampler(
        labs, sites, batch_size=32, min_sites_per_class=4, seed=7
    )

    assert len(sampler) == n // 32
    class_counts = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for batch in sampler:
            assert len(batch) == 32
            b = np.asarray(batch)
            for c in (0, 1):
                mask = labs[b] == c
                assert mask.any(), (epoch, c, "clase ausente")
                assert len(np.unique(sites[b][mask])) >= 4, (epoch, c, "sitios")
                for i in np.flatnonzero(mask):
                    assert np.any((labs[b] == c) & (sites[b] != sites[b][i])), (
                        epoch, c, i, "ancla sin positivo"
                    )
            class_counts.append(int((labs[b] == 1).sum()))
    print(
        "OK: DiagnosisStratifiedSampler — batch 32 estratificado "
        f"(clase 1: min={min(class_counts)}, max={max(class_counts)}), "
        "ambas clases, >=4 sitios y positivos válidos por ancla."
    )
