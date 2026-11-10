#!/usr/bin/env python
"""Runner de entrenamiento GNN con centinela de RAM/GPU.

Uso:
    python scripts/train/run_gnn.py --seed 42 --run-name gnn_k5_d16

Los argumentos desconocidos se pasan tal cual a `training.gnn_cli`.
Opciones del centinela:
    --ram-limit 80        porcentaje máximo de RAM del host/contenedor
    --gpu-mem-limit 80    porcentaje máximo de VRAM
    --gpu-util-limit N    opcional; por defecto no corta por utilización
    --interval 5          segundos entre muestras

Si se supera un límite, el proceso de entrenamiento se termina y el runner
sale con código 42. La utilización de GPU se registra, pero no se usa como
límite por defecto: entrenar al 100% de utilización es normal; lo peligroso
es llenar la VRAM.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.resource_guard import ResourceSentinel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        "Entrenamiento GNN con centinela de recursos",
        add_help=True,
    )
    parser.add_argument("--ram-limit", type=float, default=80.0)
    parser.add_argument("--gpu-mem-limit", type=float, default=80.0)
    parser.add_argument("--gpu-util-limit", type=float, default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    args, train_args = parser.parse_known_args()
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    return args, train_args


def _terminate(proc: subprocess.Popen, timeout: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> int:
    args, train_args = parse_args()
    sentinel = ResourceSentinel(
        ram_limit=args.ram_limit,
        gpu_mem_limit=args.gpu_mem_limit,
        gpu_util_limit=args.gpu_util_limit,
    )

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "training.gnn_cli",
        *train_args,
    ]
    print(f"[guard] límites: RAM<{args.ram_limit:.0f}% | "
          f"VRAM<{args.gpu_mem_limit:.0f}% | intervalo={args.interval:.1f}s",
          flush=True)
    print(f"[guard] comando: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, cwd=ROOT)
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                print(f"[guard] entrenamiento terminó con código {rc}", flush=True)
                return int(rc)

            time.sleep(max(args.interval, 0.5))
            sample = sentinel.sample()
            print(f"[guard] {sentinel.format_sample(sample)}", flush=True)

            ok, reasons = sentinel.check(sample)
            if not ok:
                print("[guard] LÍMITE SUPERADO: " + "; ".join(reasons), flush=True)
                _terminate(proc)
                return 42
    except KeyboardInterrupt:
        print("[guard] interrumpido por el usuario; cerrando entrenamiento", flush=True)
        _terminate(proc)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
