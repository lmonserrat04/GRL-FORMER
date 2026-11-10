#!/usr/bin/env python
"""Runner de entrenamiento GNN con centinela Linux en segundo plano.

Uso:
    python scripts/train/run_gnn.py --seed 42 --run-name gnn_k5_d16

Los argumentos desconocidos se pasan tal cual a `training.gnn_cli`.

El centinela es `scripts/train/resource_guard.sh`, un proceso bash independiente
que muestrea RAM/VRAM cada 0.5 s sin imprimir cada muestra. Solo si se supera
un umbral escribe el motivo, termina el entrenamiento y sale con código 42.

Opciones del centinela:
    --ram-limit 80        porcentaje máximo de RAM
    --gpu-mem-limit 80    porcentaje máximo de VRAM
    --gpu-util-limit N    opcional; por defecto no corta por utilización
    --interval 0.5        segundos entre muestras del proceso Linux
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD_SCRIPT = ROOT / "scripts" / "train" / "resource_guard.sh"


def parse_args():
    parser = argparse.ArgumentParser(
        "Entrenamiento GNN con centinela Linux de recursos",
        add_help=True,
    )
    parser.add_argument("--ram-limit", type=float, default=80.0)
    parser.add_argument("--gpu-mem-limit", type=float, default=80.0)
    parser.add_argument("--gpu-util-limit", type=float, default=None)
    parser.add_argument("--interval", type=float, default=0.5)
    args, train_args = parser.parse_known_args()
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    return args, train_args


def _terminate(proc: subprocess.Popen, timeout: float = 20.0) -> None:
    """Termina el entrenamiento y, en Linux, todo su grupo de procesos."""
    if proc.poll() is not None:
        return

    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()
    else:
        proc.terminate()

    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
    else:
        proc.kill()
    proc.wait()


def _stop_monitor(monitor: subprocess.Popen) -> None:
    if monitor.poll() is not None:
        return
    try:
        monitor.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        monitor.terminate()
        try:
            monitor.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()


def main() -> int:
    args, train_args = parse_args()

    if os.name != "posix":
        print(
            "[guard] el centinela Linux requiere un entorno POSIX. "
            "Ejecútalo dentro del contenedor.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    bash = shutil.which("bash")
    if bash is None or not GUARD_SCRIPT.exists():
        print(
            f"[guard] no se encontró bash o {GUARD_SCRIPT}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    interval = max(0.1, float(args.interval))
    status_path = (
        Path(tempfile.gettempdir())
        / f"gnn_guard_{os.getpid()}_{int(time.time() * 1000)}.status"
    )
    gpu_util = "" if args.gpu_util_limit is None else str(args.gpu_util_limit)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "training.gnn_cli",
        *train_args,
    ]

    print(
        f"[guard] monitor Linux en segundo plano cada {interval:.1f}s | "
        f"RAM<{args.ram_limit:.0f}% | VRAM<{args.gpu_mem_limit:.0f}%",
        flush=True,
    )

    # El entrenamiento va en su propia sesión para poder terminarlo con su grupo.
    proc = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)
    monitor = subprocess.Popen(
        [
            bash,
            str(GUARD_SCRIPT),
            str(proc.pid),
            f"{args.ram_limit}",
            f"{args.gpu_mem_limit}",
            gpu_util,
            f"{interval}",
            str(status_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    rc = 0
    try:
        time.sleep(0.1)
        if monitor.poll() not in (None, 0) and proc.poll() is None:
            print(
                "[guard] el monitor Linux no pudo arrancar; "
                "se cancela el entrenamiento",
                file=sys.stderr,
                flush=True,
            )
            _terminate(proc)
            return 2

        rc = proc.wait()
    except KeyboardInterrupt:
        print("[guard] interrumpido; cerrando entrenamiento", flush=True)
        _terminate(proc)
        rc = 130
    finally:
        _stop_monitor(monitor)

        reason = ""
        if status_path.exists():
            reason = status_path.read_text(encoding="utf-8", errors="replace").strip()
            status_path.unlink(missing_ok=True)

        if reason:
            print(f"[guard] LÍMITE SUPERADO: {reason}", flush=True)
            rc = 42

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
