"""Wrapper fino: finetune por fold. Delega en training.train_finetune.

Requiere config['CKPT_CONTRASTIVE'] definido (lo produce Fase 3).
"""
import time
from pathlib import Path

from training.train_finetune import finetune_fold as _run


def run_finetuning(config: dict, fold_idx: int):
    """
    Fase 4: Finetune de un fold.
    Carga projections congeladas del contrastive global.
    """
    save_dir = Path(config["CHECKPOINTS_PATH"])

    if not config.get("CKPT_CONTRASTIVE"):
        raise ValueError("[orchestration] Falta config['CKPT_CONTRASTIVE'] para finetune")

    print(f"\n{'='*60}")
    print(f"[orchestration] FASE 4: Finetune fold {fold_idx}")
    print(f"  TST1 ← {config['CKPT_TST1']}")
    print(f"  TST2 ← {config['CKPT_TST2']}")
    print(f"  Proj ← {config['CKPT_CONTRASTIVE']}")
    print(f"  epochs : {config['FINETUNING']['N_EPOCHS']}")
    print(f"  lr     : {config['FINETUNING']['LR']}")
    print(f"{'='*60}")

    t0 = time.time()
    metrics = _run(config, fold_idx=fold_idx, save_dir=save_dir)
    elapsed = time.time() - t0

    print(f"[orchestration] Finetune fold {fold_idx} completado en {elapsed/60:.1f} min")
    print(f"  AUC={metrics['auc']:.4f}  ACC={metrics['accuracy']:.4f}  "
          f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
          f"F1={metrics['f1']:.4f}")
    return metrics
