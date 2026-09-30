"""
Utilidades de conectividad funcional.

- compute_pcc_vector: Pearson upper-triangle (legacy, por sujeto).
- compute_pcc_tangent_batch: tangent space (nilearn + joblib processes)
  con guard de RAM para evitar crashes de sistema.

Estrategia de paralelismo:
    ConnectivityMeasure NO acepta n_jobs. Se paraleliza por fuera con joblib
    usando processes (independiente del GIL). BLAS cap a 1 thread por worker
    (OMP_NUM_THREADS=1) para evitar oversubscription.

Protección de RAM:
    - Pre-flight: _safe_n_jobs reduce workers si la RAM libre es baja.
    - Runtime: _RAMGuard lanza un thread daemon que monitorea RAM cada 2 s.
      Si baja del umbral 3 checks seguidos → os._exit(2) para evitar freeze.
"""
from __future__ import annotations

# Cap de BLAS ANTES de importar numpy/scipy/nilearn.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import threading
import numpy as np
import torch
from joblib import Parallel, delayed
from nilearn.connectome import ConnectivityMeasure


# ──────────────────────────────────────────────────────────────────────
# Monitoreo de RAM
# ──────────────────────────────────────────────────────────────────────

def _available_ram_gb() -> float:
    """
    RAM disponible del sistema en GiB.

    Intenta psutil primero; fallback a /proc/meminfo (Linux) o 999 si
    no se puede determinar (asume seguro).
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass

    # Fallback Linux: /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass

    return float("inf")  # no idea, asumir ok


_WORKER_RAM_GB = 0.5  # estimación conservadora por worker (numpy + scipy + nilearn)
_RESERVE_GB    = 2.0  # margen de seguridad para el resto del sistema


def _safe_n_jobs(n_jobs: int) -> int:
    """
    Reduce n_jobs según RAM libre y cap duro de 4.

    Reglas:
      - n_jobs == -1  → min(4, max_por_ram)
      - n_jobs <= 0   → 1
      - n_jobs > 0    → min(n_jobs, 4, max_por_ram)
    """
    avail = _available_ram_gb()
    usable = max(0.0, avail - _RESERVE_GB)
    max_by_ram = max(1, int(usable / _WORKER_RAM_GB))

    if n_jobs == -1:
        return min(4, max_by_ram)
    if n_jobs <= 0:
        return 1
    return max(1, min(int(n_jobs), 4, max_by_ram))


class _RAMGuard:
    """
    Thread daemon que aborta el proceso si la RAM disponible cae por
    debajo del umbral de forma sostenida.

    Uso:
        guard = _RAMGuard(threshold_gb=1.5, interval_s=2.0, consecutive=3)
        guard.start()
        try:
            ... trabajo pesado ...
        finally:
            guard.stop()
    """

    def __init__(
        self,
        threshold_gb: float = 1.5,
        interval_s: float = 2.0,
        consecutive: int = 3,
    ):
        self.threshold_gb = threshold_gb
        self.interval_s = interval_s
        self.consecutive = consecutive
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._counter = 0
        self.breached = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            avail = _available_ram_gb()
            if avail < self.threshold_gb:
                self._counter += 1
                print(
                    f"  [RAMGuard] RAM baja: {avail:.2f} GiB "
                    f"(aviso {self._counter}/{self.consecutive})",
                    flush=True,
                )
                if self._counter >= self.consecutive:
                    self.breached = True
                    print(
                        f"  [RAMGuard] RAM crítica sostenida "
                        f"(< {self.threshold_gb} GiB) → abortando "
                        f"para evitar crash de sistema.",
                        flush=True,
                    )
                    os._exit(2)
            else:
                self._counter = 0


# ──────────────────────────────────────────────────────────────────────
# Pearson legacy
# ──────────────────────────────────────────────────────────────────────

def compute_pcc_vector(timeseries: torch.Tensor) -> torch.Tensor:
    """Vector PCC de Pearson (triangular superior). Legacy."""
    corr = torch.corrcoef(timeseries)
    triu_idx = torch.triu_indices(corr.shape[0], corr.shape[1], offset=1)
    return corr[triu_idx[0], triu_idx[1]]


# ──────────────────────────────────────────────────────────────────────
# Tangent batch
# ──────────────────────────────────────────────────────────────────────

def compute_pcc_tangent_batch(
    timeseries_batch: np.ndarray,
    n_jobs: int = 4,
    verbose: int = 0,
    ram_guard: bool = True,
    ram_threshold_gb: float = 1.5,
) -> np.ndarray:
    """
    Vectores de conectividad en tangent space (joblib processes + RAM guard).

    nilearn espera cada sujeto como (n_samples, n_features) = (T, R).

    Args:
        timeseries_batch: (N, T, R).
        n_jobs:           workers de joblib. -1 = min(4, RAM disponible).
        verbose:          nivel de logging de joblib.
        ram_guard:        activar el guard de RAM en runtime.
        ram_threshold_gb: umbral mínimo de RAM libre antes de abortar.

    Returns:
        vectors: (N, R*(R-1)//2).
    """
    if timeseries_batch.ndim != 3:
        raise ValueError(
            f"Se esperaba (N, T, R), recibido {timeseries_batch.shape}"
        )

    # Pre-flight: ajustar n_jobs según RAM disponible
    n_jobs_eff = _safe_n_jobs(n_jobs)
    avail_gb = _available_ram_gb()
    print(
        f"  [tangent] n_jobs efectivo: {n_jobs_eff}  "
        f"(RAM libre: {avail_gb:.1f} GiB, cap: 4)",
        flush=True,
    )

    N, T, R = timeseries_batch.shape
    ts_list = [timeseries_batch[i] for i in range(N)]   # (T, R)

    measure = ConnectivityMeasure(
        kind="tangent",
        vectorize=True,
        discard_diagonal=True,
    )

    # Fit: referencia (media geométrica). Rápido, secuencial.
    print(f"  [tangent] fit sobre {N} sujetos...", flush=True)
    measure.fit(ts_list)

    # Transform en chunks. Cada worker recibe el `measure` una vez (pickle).
    chunk_size = 16
    chunks = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

    def _transform_chunk(start_stop):
        start, stop = start_stop
        return [measure.transform([ts_list[i]])[0] for i in range(start, stop)]

    print(
        f"  [tangent] transform en {len(chunks)} chunks "
        f"({chunk_size} sujetos por chunk)...",
        flush=True,
    )

    guard = _RAMGuard(
        threshold_gb=ram_threshold_gb,
        interval_s=2.0,
        consecutive=3,
    ) if ram_guard else None

    if guard is not None:
        guard.start()

    try:
        results = Parallel(
            n_jobs=n_jobs_eff,
            prefer="processes",
            verbose=verbose,
            batch_size="auto",
        )(delayed(_transform_chunk)(c) for c in chunks)
    finally:
        if guard is not None:
            guard.stop()

    vectors = np.concatenate(results, axis=0)
    print(f"  [tangent] completado: {vectors.shape}", flush=True)
    return vectors.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Test inline
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=== Test tangent con RAM guard ===")
    print(f"RAM disponible: {_available_ram_gb():.2f} GiB")
    print(f"n_jobs seguro (-1): {_safe_n_jobs(-1)}")
    print(f"n_jobs seguro (4):  {_safe_n_jobs(4)}")
    print()

    X = np.random.randn(20, 116, 200).astype(np.float32)

    t0 = time.time()
    v = compute_pcc_tangent_batch(X, n_jobs=4, verbose=0)
    dt = time.time() - t0

    print()
    print(f"20 sujetos: {dt:.2f}s, shape={v.shape}")
    print(f"Extrapolación a 831: ~{dt * 831 / 20:.1f}s")
    assert v.shape == (20, 19900), f"shape inesperado: {v.shape}"
    print("OK")


def compute_pcc_vector_np(ts_np: np.ndarray) -> np.ndarray:
    """
    Wrapper de compute_pcc_vector que acepta numpy (R, T) y devuelve numpy.
    """
    return compute_pcc_vector(torch.from_numpy(ts_np).float()).numpy()
