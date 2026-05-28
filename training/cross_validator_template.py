# training/cross_validator_template.py

import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from orchestration.run_pretrain_ts import run_pretrain_ts
from orchestration.run_pretrain_fc import run_pretrain_fc
from orchestration.run_contrastive import run_contrastive
from orchestration.run_finetuning import run_finetuning
from test.test import test
from utils.checkpoint import get_checkpoint_path
import torch.nn as nn


def create_experiment_dir(config: dict) -> dict:
    """Crea los directorios del experimento y actualiza las rutas en config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{timestamp}_{config.get('RUN_NAME', 'exp')}"

    exp_dir = Path("experiments") / exp_id
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # NO Guarda copia del config usado
    #shutil.copy("config/config.yaml", exp_dir / "config.yaml")

    config["LOGS_PATH"] = str(exp_dir / "logs")
    config["CHECKPOINTS_PATH"] = str(exp_dir / "checkpoints")
    config["RESULTS_PATH"] = str(exp_dir / "results.csv")
    config["EXP_ID"] = exp_id
    return config


def split_fold(df: pd.DataFrame, fold: int, n_folds: int = 3):
    """
    División simple en train / val / test para un fold dado.
    Retorna tres DataFrames. test = val en este ejemplo simplificado.
    """
    n = len(df)
    fold_size = n // n_folds
    val_idx = list(range(fold * fold_size, (fold + 1) * fold_size))
    train_idx = [i for i in range(n) if i not in val_idx]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


def run_cross_validation(config, n_folds=3):
    """
    Ejecuta la validación cruzada completa sobre los datos indicados en config['CSV_PATH'].
    """
    config = create_experiment_dir(config)

    # Cargar el CSV con los metadatos
    csv_path = config.get("CSV_PATH", "data/datasets/synthetic/data.csv")
    df_global = pd.read_csv(csv_path)

    # Si el CSV no tiene índice original, lo creamos (0..N-1)
    if df_global.index.name is None and not isinstance(df_global.index, pd.RangeIndex):
        df_global = df_global.reset_index(drop=True)

    # Acumuladores para resultados de test
    results_cols = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AP', 'FPR', 'FNR', 'TPR', 'TNR']
    all_results = []
    accs = []

    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1} / {n_folds}")
        print(f"{'='*60}")

        df_train, df_val, df_test = split_fold(df_global, fold, n_folds)

        # Fase 1: Pretrain TST1 (series temporales)
        print(f"\n── Fase 1: pretrain_ts (fold {fold}) ──")
        run_pretrain_ts(config, df_train, df_val, df_test, fold)

        # Fase 2: Pretrain TST2 (conectividad funcional)
        print(f"\n── Fase 2: pretrain_fc (fold {fold}) ──")
        run_pretrain_fc(config, df_train, df_val, df_test, fold)

        # Fase 3: Contrastive learning
        print(f"\n── Fase 3: contrastive (fold {fold}) ──")
        chkpt_ts = get_checkpoint_path(config, "PT_TS", fold)
        chkpt_fc = get_checkpoint_path(config, "PT_FC", fold)
        run_contrastive(config, df_train, df_val, df_test, fold,
                        chkpt_ts=chkpt_ts, chkpt_fc=chkpt_fc)

        # Fase 4: Finetuning
        print(f"\n── Fase 4: finetuning (fold {fold}) ──")
        chkpt_cont = get_checkpoint_path(config, "CONT", fold)
        run_finetuning(config, df_train, df_val, df_test, fold,
                       chkpt_cont=chkpt_cont)

        # Fase 5: Evaluación
        print(f"\n── Fase 5: evaluación (fold {fold}) ──")
        config["EXPERIMENT_TYPE"] = "finetune"

        from training.setup import build_experiment, build_dataloaders

        chkpt_finetune = get_checkpoint_path(config, "FINETUNE", fold)
        exp_finetune = build_experiment(config, df_train, df_val, df_test,
                                        chkpt_cont=chkpt_finetune)
        model = exp_finetune.model
        _, _, test_loader = build_dataloaders(config, df_train, df_val, df_test,
                                              harmonizer=None, normalizer=None)
        criterion = nn.CrossEntropyLoss()
        vals = []
        test(model, fold, accs, vals, config, test_loader, criterion)
        if vals:
            all_results.append(vals[0])

    # Guardar resultados finales
    if all_results:
        results_df = pd.DataFrame(all_results, columns=results_cols)
        results_df.to_csv(config["RESULTS_PATH"], index=False)
        print(f"\nResultados guardados en {config['RESULTS_PATH']}")
        print(results_df.to_string(index=False))
        if accs:
            print(f"\nMean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    print("\n✅ Validación cruzada finalizada.")


if __name__ == "__main__":
    import sys

    # YAML por defecto
    config_path = "config/config_synth.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Asegurar dispositivo y rutas (ajustar si es necesario)
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    # Si el YAML no trae CSV_PATH, forzamos el sintético
    if "CSV_PATH" not in config:
        config["CSV_PATH"] = "data/datasets/synthetic_cv/data.csv"

    run_cross_validation(config, n_folds=config.get("N_FOLDS", 3))