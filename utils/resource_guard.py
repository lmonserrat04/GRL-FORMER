"""Centinela de recursos para entrenamientos largos.

Lee RAM del host/contenedor y, si está disponible, métricas de NVIDIA con
`nvidia-smi`. No necesita psutil: usa /proc/meminfo como fallback.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResourceSample:
    ram_percent: float
    gpu_mem_percent: float | None = None
    gpu_util_percent: float | None = None
    gpu_used_mb: float | None = None
    gpu_total_mb: float | None = None
    gpu_name: str | None = None


def read_ram_percent() -> float:
    """Porcentaje de RAM usada. Prefiere psutil; si no, /proc/meminfo."""
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().percent)
    except Exception:
        pass

    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return float("nan")

    values: dict[str, float] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if parts:
            values[key] = float(parts[0])

    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", values.get("MemFree", 0.0))
    if total <= 0:
        return float("nan")
    return max(0.0, min(100.0, 100.0 * (1.0 - available / total)))


def read_gpu_sample() -> tuple[float | None, float | None, float | None, float | None, str | None]:
    """Devuelve (mem_pct, util_pct, used_mb, total_mb, name) de la GPU más cargada."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None, None, None, None, None

    best = None
    for line in out.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            continue
        name, used, total, util = fields
        try:
            used_mb = float(used)
            total_mb = float(total)
            util_pct = float(util)
        except ValueError:
            continue
        mem_pct = 100.0 * used_mb / total_mb if total_mb > 0 else 0.0
        candidate = (mem_pct, util_pct, used_mb, total_mb, name)
        if best is None or candidate[0] > best[0]:
            best = candidate

    return best if best is not None else (None, None, None, None, None)


class ResourceSentinel:
    """Muestrea RAM/GPU y decide si se superó un umbral."""

    def __init__(
        self,
        ram_limit: float = 80.0,
        gpu_mem_limit: float = 80.0,
        gpu_util_limit: float | None = None,
    ):
        self.ram_limit = float(ram_limit)
        self.gpu_mem_limit = float(gpu_mem_limit)
        self.gpu_util_limit = None if gpu_util_limit is None else float(gpu_util_limit)

    def sample(self) -> ResourceSample:
        gpu_mem, gpu_util, used, total, name = read_gpu_sample()
        return ResourceSample(
            ram_percent=read_ram_percent(),
            gpu_mem_percent=gpu_mem,
            gpu_util_percent=gpu_util,
            gpu_used_mb=used,
            gpu_total_mb=total,
            gpu_name=name,
        )

    def check(self, sample: ResourceSample) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if math.isfinite(sample.ram_percent) and sample.ram_percent >= self.ram_limit:
            reasons.append(f"RAM {sample.ram_percent:.1f}% >= {self.ram_limit:.0f}%")
        if (
            sample.gpu_mem_percent is not None
            and sample.gpu_mem_percent >= self.gpu_mem_limit
        ):
            reasons.append(
                f"VRAM {sample.gpu_mem_percent:.1f}% >= {self.gpu_mem_limit:.0f}%"
            )
        if (
            sample.gpu_util_percent is not None
            and self.gpu_util_limit is not None
            and sample.gpu_util_percent >= self.gpu_util_limit
        ):
            reasons.append(
                f"GPU util {sample.gpu_util_percent:.1f}% >= "
                f"{self.gpu_util_limit:.0f}%"
            )
        return not reasons, reasons

    @staticmethod
    def format_sample(sample: ResourceSample) -> str:
        parts = [f"RAM {sample.ram_percent:.1f}%"]
        if sample.gpu_mem_percent is not None:
            parts.append(
                f"VRAM {sample.gpu_mem_percent:.1f}% "
                f"({sample.gpu_used_mb:.0f}/{sample.gpu_total_mb:.0f} MB)"
            )
        else:
            parts.append("VRAM n/d")
        if sample.gpu_util_percent is not None:
            parts.append(f"GPU util {sample.gpu_util_percent:.1f}%")
        if sample.gpu_name:
            parts.append(sample.gpu_name)
        return " | ".join(parts)
