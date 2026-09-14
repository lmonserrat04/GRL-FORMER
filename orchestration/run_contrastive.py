"""Wrapper fino: contrastive global. Delega en training.train_contrastive.

Requiere que config tenga CKPT_TST1 y CKPT_TST2 ya definidos.
Guarda contrastive_global.pt con TST1, TST2 y proj heads.
"""
import time
from pathlib import Path

from training.train_contrastive import run_contrastive_global as _run


def run_contrastive(config: dict):
    """
    Fase 3: Contrastive GLOBAL (paper Sec. 3.3).
    Estrategia: freeze TST1, unfreeze TST2.
    """
    save_dir = Path(config["CHECKPOINTS_PATH"])

    # Verificaciones rápidas
    for key in ("CKPT_TST1", "CKPT_TST2"):
        if not config.get(key):
            raise ValueError(f"[orchestration] Falta config['{key}'] para correr contrastive")

    print(f"\n{'='*60}")
    print(f"[orchestration] FASE 3: Contrastive GLOBAL")
    print(f"  TST1 ← {config['CKPT_TST1']}")
    print(f"  TST2 ← {config['CKPT_TST2']}")
    print(f"  epochs : {config['T_CONTRASTIVE']['N_EPOCHS']}")
    print(f"  τ      : {config['T_CONTRASTIVE']['TEMPERATURE']}")
    print(f"  freeze : TST1 (unfreeze TST2)")
    print(f"{'='*60}")

    t0 = time.time()
    _run(config, save_dir=save_dir)
    elapsed = time.time() - t0

    ckpt = save_dir / "contrastive_global.pt"
    print(f"[orchestration] Contrastive completado en {elapsed/60:.1f} min")
    print(f"  checkpoint : {ckpt}")
    return ckpt
