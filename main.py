"""
main.py — Pipeline completo TwoTST (config óptima del paper).

    python main.py --config config/config.yaml

Pasos:
    1. Pretrain TST1 (100 epochs, mask ∈ [0.25, 0.5]).
    2. Pretrain TST2 (100 epochs, mask=0.15).
    3. Por fold: contrastive (50 epochs, unfreeze both) + finetune
       (attention_pooling, encoders descongelados) + evaluación subject-level.
    4. Resumen: mean ± std + bootstrap 95% CI.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from training.train_pretrain_ts import run_pretrain_ts
from training.train_pretrain_fc import run_pretrain_fc
from training.train_finetune import finetune_fold
from utils.metrics import bootstrap_confidence_interval, get_reproducibility_info
from data.preprocessing.splitters import (
    get_subject_level_fold_splits,
    get_loso_fold_splits,
)
from data.loaders.dataloader import load_raw_data


# ──────────────────────────────────────────────────────────────────────
# Setup del experimento
# ──────────────────────────────────────────────────────────────────────

def create_experiment_dir(config: dict, config_path: str) -> dict:
    """Crea experiments/{timestamp}_{RUN_NAME}/ con logs/ y checkpoints/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get("RUN_NAME", "exp")
    exp_id = f"{timestamp}_{run_name}"

    exp_dir = Path(config.get("EXPERIMENTS_ROOT", "./experiments")) / exp_id
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, exp_dir / "config.yaml")

    config["EXP_ID"] = exp_id
    config["EXP_DIR"] = str(exp_dir)
    config["LOGS_PATH"] = str(exp_dir / "logs")
    config["CHECKPOINTS_PATH"] = str(exp_dir / "checkpoints")
    return config


def build_folds(config: dict) -> list[dict]:
    """Devuelve la lista de folds (kfold o loso) usando los splitters."""
    data = load_raw_data(config, use_interp=False)
    protocol = config.get("EVAL_PROTOCOL", "kfold")

    if protocol == "loso":
        if data["site_ids"] is None:
            raise ValueError("LOSO requiere site_ids.")
        splits = get_loso_fold_splits(
            data["labels"], data["subject_indices"], data["site_ids"],
            val_ratio=0.15, seed=config["SEED"],
        )
    else:
        splits = get_subject_level_fold_splits(
            data["labels"], data["subject_indices"],
            site_ids=data["site_ids"],
            n_splits=config.get("N_FOLDS", 5),
            val_ratio=0.15, seed=config["SEED"],
        )
    return splits


# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────
def run_pipeline(config: dict):
    ckpt_dir = Path(config["CHECKPOINTS_PATH"])

    # ─── 1. Pretrain TST1 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FASE 1: Pretrain TST1 (ROI-level masking)")
    print("=" * 60)
    run_pretrain_ts(config, fold_idx=0, save_dir=ckpt_dir)

    # ─── 2. Pretrain TST2 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FASE 2: Pretrain TST2 (element-level masking)")
    print("=" * 60)
    run_pretrain_fc(config, fold_idx=0, save_dir=ckpt_dir)

    config["CKPT_TST1"] = str(ckpt_dir / "best_pt_ts_fold_0.pt")
    config["CKPT_TST2"] = str(ckpt_dir / "best_pt_fc_fold_0.pt")

    # ─── 3. Contrastive GLOBAL (una vez) ─────────────────────────────
    print("\n" + "=" * 60)
    print("FASE 3: Contrastive global (paper Sec. 3.3)")
    print("=" * 60)
    from training.train_contrastive import run_contrastive_global
    run_contrastive_global(config, save_dir=ckpt_dir)
    config["CKPT_CONTRASTIVE"] = str(ckpt_dir / "contrastive_global.pt")

    # ─── 4. Folds (solo finetuning) ──────────────────────────────────
    folds = build_folds(config)
    print(f"\n{'=' * 60}")
    print(f"FASE 4: {config.get('EVAL_PROTOCOL', 'kfold').upper()} — {len(folds)} folds")
    print("=" * 60)

    fold_metrics = []
    for fold_idx, split in enumerate(folds):
        tag = split.get("test_site", f"fold{fold_idx}")
        print(f"\n──── Fold {fold_idx + 1}/{len(folds)} ({tag}) ────")
        metrics = finetune_fold(config, fold_idx=fold_idx, save_dir=ckpt_dir)
        metrics["fold_idx"] = fold_idx
        if "test_site" in split:
            metrics["test_site"] = split["test_site"]
        fold_metrics.append(metrics)

    # ─── 5. Resumen (igual que antes) ────────────────────────────────
    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {
        "protocol": config.get("EVAL_PROTOCOL", "kfold"),
        "n_folds": len(folds),
        "seed": config["SEED"],
        "exp_id": config["EXP_ID"],
    }

    print(f"\n{'=' * 60}")
    print(f"RESUMEN ({summary['protocol'].upper()}, mean ± std [95% CI])")
    print("=" * 60)

    for name in metric_names:
        vals = [m[name] for m in fold_metrics if name in m]
        if not vals:
            continue
        mean_v, std_v, lo, hi = bootstrap_confidence_interval(
            vals, n_bootstrap=1000, ci=0.95, seed=config["SEED"]
        )
        summary[name] = {
            "mean": mean_v, "std": std_v,
            "ci95_lower": lo, "ci95_upper": hi,
        }
        label = name.upper() if name == "auc" else name.capitalize()
        print(f"  {label:12s}: {mean_v:.4f} ± {std_v:.4f}  [{lo:.4f}, {hi:.4f}]")

    summary["all_folds"] = fold_metrics
    summary["reproducibility"] = get_reproducibility_info()

    results_path = Path(config["EXP_DIR"]) / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    print(f"\n✅ Pipeline completado.")
    print(f"   Resultados: {results_path}")
    return summary

# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TwoTST — pipeline completo")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Saltar pretrain si ya tienes checkpoints.")
    parser.add_argument("--ckpt_tst1", default=None,
                        help="Checkpoint TST1 externo (con --skip_pretrain).")
    parser.add_argument("--ckpt_tst2", default=None,
                        help="Checkpoint TST2 externo (con --skip_pretrain).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = create_experiment_dir(config, args.config)

    torch.manual_seed(config["SEED"])
    np.random.seed(config["SEED"])

    if args.skip_pretrain:
        if not (args.ckpt_tst1 and args.ckpt_tst2):
            raise ValueError("--skip_pretrain requiere --ckpt_tst1 y --ckpt_tst2")
        config["CKPT_TST1"] = args.ckpt_tst1
        config["CKPT_TST2"] = args.ckpt_tst2

        folds = build_folds(config)
        print(f"\nSaltando pretrain. {len(folds)} folds a evaluar.")
        fold_metrics = []
        for fold_idx, split in enumerate(folds):
            tag = split.get("test_site", f"fold{fold_idx}")
            print(f"\n──── Fold {fold_idx + 1}/{len(folds)} ({tag}) ────")
            metrics = finetune_fold(config, fold_idx=fold_idx,
                                    save_dir=Path(config["CHECKPOINTS_PATH"]))
            metrics["fold_idx"] = fold_idx
            if "test_site" in split:
                metrics["test_site"] = split["test_site"]
            fold_metrics.append(metrics)

        metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
        summary = {"protocol": config.get("EVAL_PROTOCOL", "kfold"),
                   "n_folds": len(folds), "seed": config["SEED"],
                   "exp_id": config["EXP_ID"]}
        for name in metric_names:
            vals = [m[name] for m in fold_metrics if name in m]
            if vals:
                mean_v, std_v, lo, hi = bootstrap_confidence_interval(
                    vals, n_bootstrap=1000, ci=0.95, seed=config["SEED"])
                summary[name] = {"mean": mean_v, "std": std_v,
                                 "ci95_lower": lo, "ci95_upper": hi}
        summary["all_folds"] = fold_metrics
        with open(Path(config["EXP_DIR"]) / "results.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
        return summary

    return run_pipeline(config)


if __name__ == "__main__":
    main()