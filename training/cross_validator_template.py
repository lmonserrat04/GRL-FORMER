from utils.checkpoint import get_checkpoint_path
from orchestration.run_pretrain_ts import run_pretrain_ts
from orchestration.run_pretrain_fc import run_pretrain_fc
from orchestration.run_contrastive import run_contrastive
from orchestration.run_finetuning import run_finetuning
from test.test import test
import torch.nn as nn
from orchestration.run_finetuning import run_finetuning
from utils.checkpoint import get_checkpoint_path
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset

# training/cross_validator_template.py

# ── Helpers ─────────────────────────────────────────────────────────────────
EXP_TO_CONFIG_KEY = {
    "pretrain_ts":  "PT_TST1",
    "pretrain_fc":  "PT_TST2",
    "contrastive":  "T_CONTRASTIVE",
    "finetune":     "FINETUNING",
}

def get_batch_size(config: dict) -> int:
    key = EXP_TO_CONFIG_KEY.get(config["EXPERIMENT_TYPE"])
    if key is None:
        raise ValueError(f"No config key mapped for '{config['EXPERIMENT_TYPE']}'")
    return config[key]["BATCH_SIZE"]

# ── Datasets sintéticos ────────────────────────────────────────────────────
class SyntheticTSDataset(Dataset):
    def __init__(self, n, seq_len, n_rois):
        torch.manual_seed(0)
        self.x = torch.randn(n, seq_len, n_rois)
        self.y = torch.randint(0, 2, (n,))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

class SyntheticFCDataset(Dataset):
    def __init__(self, n, pcc_dim):
        torch.manual_seed(0)
        self.x = torch.randn(n, pcc_dim)
        self.y = torch.randint(0, 2, (n,))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

class SyntheticDualDataset(Dataset):
    def __init__(self, n, seq_len, n_rois, pcc_dim):
        torch.manual_seed(0)
        self.ts  = torch.randn(n, seq_len, n_rois)
        self.pcc = torch.randn(n, pcc_dim)
        self.y   = torch.randint(0, 2, (n,))
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {"timeseries": self.ts[i], "pcc_vector": self.pcc[i]}, self.y[i]

class SyntheticFinetuneDataset(Dataset):
    def __init__(self, n, seq_len, n_rois, pcc_dim):
        torch.manual_seed(0)
        self.ts  = torch.randn(n, seq_len, n_rois)
        self.pcc = torch.randn(n, pcc_dim)
        self.y   = torch.randint(0, 2, (n,))
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {"timeseries": self.ts[i], "pcc_vector": self.pcc[i], "label": self.y[i]}, self.y[i]

def finetune_collate(batch):
    ts_list, pcc_list, labels_list = [], [], []
    for data_dict, label in batch:
        ts_list.append(data_dict['timeseries'])
        pcc_list.append(data_dict['pcc_vector'])
        labels_list.append(label)
    return {
        'timeseries': torch.stack(ts_list),
        'pcc_vector': torch.stack(pcc_list),
        'label': torch.tensor(labels_list)
    }

# ── Loader factories (ahora usan parámetros del config) ────────────────────
def _get_loader_factories(config):
    n_rois = config["N_ROIS"]
    seq_len = config["MAX_SEQ_LEN"]
    pcc_dim = config["TST2"]["PCC_DIM"]

    def make_ts_loader(n, bs):
        return DataLoader(SyntheticTSDataset(n, seq_len, n_rois), batch_size=bs, shuffle=True, num_workers=0)

    def make_fc_loader(n, bs):
        return DataLoader(SyntheticFCDataset(n, pcc_dim), batch_size=bs, shuffle=True, num_workers=0)

    def make_dual_loader(n, bs):
        return DataLoader(SyntheticDualDataset(n, seq_len, n_rois, pcc_dim), batch_size=bs, shuffle=True, num_workers=0)

    def make_finetune_loader(n, bs):
        ds = SyntheticFinetuneDataset(n, seq_len, n_rois, pcc_dim)
        return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, collate_fn=finetune_collate)

    return {
        "pretrain_ts": make_ts_loader,
        "pretrain_fc": make_fc_loader,
        "contrastive": make_dual_loader,
        "finetune":    make_finetune_loader,
    }

# ── Monkey-patch build_dataloaders ─────────────────────────────────────────
import training.setup as setup_module

def patch_dataloaders(config, n_samples):
    factories = _get_loader_factories(config)

    def _patched_build_dataloaders(config, df_train, df_val, df_test,
                                   harmonizer=None, normalizer=None):
        exp_type = config["EXPERIMENT_TYPE"]
        bs       = get_batch_size(config)
        factory  = factories[exp_type]
        return (
            factory(n_samples,      bs),
            factory(n_samples // 2, bs),
            factory(n_samples // 4, bs),
        )
    setup_module.build_dataloaders = _patched_build_dataloaders

# ── Monkey-patch create_dual_stream_model ──────────────────────────────────
import models.dual_stream as dual_module

def patch_dual_stream():
    def _patched_create_dual_stream(config, name_chkpt_pt_ts=None, name_chkpt_pt_fc=None,
                                    name_chkpt_cont=None):
        from models.dual_stream import DualStreamModel
        model = DualStreamModel(config)
        if name_chkpt_pt_ts and Path(name_chkpt_pt_ts).exists():
            model.load_pretrained_tst1(name_chkpt_pt_ts)
            print(f"  Loaded TST1 ← {name_chkpt_pt_ts}")
        else:
            print(f"  [smoke] skip TST1 load ({name_chkpt_pt_ts})")
        if name_chkpt_pt_fc and Path(name_chkpt_pt_fc).exists():
            model.load_pretrained_tst2(name_chkpt_pt_fc)
            print(f"  Loaded TST2 ← {name_chkpt_pt_fc}")
        else:
            print(f"  [smoke] skip TST2 load ({name_chkpt_pt_fc})")
        if name_chkpt_cont and Path(name_chkpt_cont).exists():
            model.load_pretrained_contrastive(name_chkpt_cont)
            print(f"  Loaded contrastive ← {name_chkpt_cont}")
        return model

    dual_module.create_dual_stream_model = _patched_create_dual_stream
    try:
        setup_module.create_dual_stream_model = _patched_create_dual_stream
    except AttributeError:
        pass

# ── create_experiment_dir ──────────────────────────────────────────────────
def create_experiment_dir(config: dict) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id    = f"{timestamp}_{config.get('RUN_NAME', 'exp')}"
    exp_dir   = Path("experiments") / exp_id
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config["LOGS_PATH"]        = str(exp_dir / "logs")
    config["CHECKPOINTS_PATH"] = str(exp_dir / "checkpoints")
    config["RESULTS_PATH"]     = str(exp_dir / "results.csv")
    config["EXP_ID"]           = exp_id
    return config

# ── split_fold sintético ───────────────────────────────────────────────────
def split_fold(df: pd.DataFrame, fold: int, n_folds: int = 3):
    n        = len(df)
    fold_size = n // n_folds
    val_idx  = list(range(fold * fold_size, (fold + 1) * fold_size))
    train_idx = [i for i in range(n) if i not in val_idx]
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True))

# ── DataFrame dummy ────────────────────────────────────────────────────────
def make_dummy_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id":         [f"sub_{i:03d}" for i in range(n)],
        "DX_GROUP":           np.random.randint(0, 2, n),
        "AGE_AT_SCAN":        np.random.uniform(10, 60, n),
        "SEX":                np.random.randint(1, 3, n),
        "EYE_STATUS_AT_SCAN": np.random.randint(1, 3, n),
    })

# ── Función principal (orquestación) ───────────────────────────────────────
def run_cross_validation(config, n_folds=3, n_samples=32):
    """
    Ejecuta validación cruzada con datos sintéticos.
    config: diccionario de configuración cargado del YAML.
    """
    # Crear directorio de experimento
    config = create_experiment_dir(config)

    # Aplicar monkey-patches
    patch_dataloaders(config, n_samples)
    patch_dual_stream()

    # DataFrame sintético
    df_global = make_dummy_df(n_samples)

    results_cols = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AP', 'FPR', 'FNR', 'TPR', 'TNR']
    all_results = []
    accs = []

    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1} / {n_folds}")
        print(f"{'='*60}")

        df_train, df_val, df_test = split_fold(df_global, fold, n_folds)

        # 1. Pretrain TST1
        print(f"\n── Fase 1: pretrain_ts (fold {fold}) ──")
        run_pretrain_ts(config, df_train, df_val, df_test, fold)

        # 2. Pretrain TST2
        print(f"\n── Fase 2: pretrain_fc (fold {fold}) ──")
        run_pretrain_fc(config, df_train, df_val, df_test, fold)

        # 3. Contrastive
        print(f"\n── Fase 3: contrastive (fold {fold}) ──")
        chkpt_ts = get_checkpoint_path(config, "PT_TS", fold)
        chkpt_fc = get_checkpoint_path(config, "PT_FC", fold)
        run_contrastive(config, df_train, df_val, df_test, fold,
                        chkpt_ts=chkpt_ts, chkpt_fc=chkpt_fc)

        # 4. Finetuning
        print(f"\n── Fase 4: finetuning (fold {fold}) ──")
        chkpt_cont = get_checkpoint_path(config, "CONT", fold)
        run_finetuning(config, df_train, df_val, df_test, fold,
                       chkpt_cont=chkpt_cont)

        # 5. Test (descomentar cuando esté listo)
        print(f"\n── Fase 5: evaluación (fold {fold}) ──")
        config["EXPERIMENT_TYPE"] = "finetune"
        chkpt_finetune = get_checkpoint_path(config, "FINETUNE", fold)
        exp_finetune = setup_module.build_experiment(config, df_train, df_val, df_test, chkpt_cont=chkpt_finetune)
        model = exp_finetune["model"]
        _, _, test_loader = setup_module.build_dataloaders(config, df_train, df_val, df_test,
                                              harmonizer=None, normalizer=None)
        criterion = nn.CrossEntropyLoss()
        vals = []
        test(model, fold, accs, vals, config, test_loader, criterion)
        if vals:
            all_results.append(vals[0])

   
    results_df = pd.DataFrame(all_results, columns=results_cols)
    results_df.to_csv(config["RESULTS_PATH"], index=False)
    print(f"\nResultados guardados en {config['RESULTS_PATH']}")
    print(results_df.to_string(index=False))
    print(f"\nMean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

if __name__ == "__main__":
    # Configuración de ejemplo para pruebas autónomas
    config = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "RUN_NAME": "smoke_test_cv",
    "N_ROIS": 200,
    "MAX_SEQ_LEN": 100,
    "WINDOW_SIZE": 128,
    "NUM_WORKERS": 0,
    "SEED": 42,

    "CHECKPOINTS": {
        "PT_TS":    "best_pt_ts_model",
        "PT_FC":    "best_pt_fc_model",
        "CONT":     "best_cont_model",
        "FINETUNE": "best_finetuned_model",
    },

    "TST1": {
        "D_MODEL": 64, "DIM_FEEDFORWARD": 128, "NUM_ENCODER_LAYERS": 2,
        "N_HEADS": 4, "ENC_DROP": 0.1, "USE_CLS_TOKEN": True
    },
    "TST2": {
        "D_MODEL": 64, "N_HEADS": 4, "NUM_ENCODER_LAYERS": 2,
        "DIM_FEEDFORWARD": 128, "ENC_DROP": 0.1, "PCC_DIM": 19900
    },
    "DUAL_STREAM": {
        "FUSION_TYPE": "attention_pooling", "NUM_CLASSES": 2,
        "CLASSIFIER_DROPOUT": 0.1
    },
    "FUSION": {
        "ATTENTION_POOLING": {"HIDDEN_DIM": 128}
    },

    # --- Fases con optimizador + scheduler nuevos ---
    "PT_TST1": {
        "N_EPOCHS": 2,
        "BATCH_SIZE": 8,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "MASK_RATIO": None,
        "OPTIMIZER": "AdamW",
        "SCHEDULER": "CosineAnnealingLR",
        "SCHEDULER_PARAMS": {
            "T_max": 2,
            "eta_min": 1e-6          # LR * 0.01 = 1e-4 * 0.01 = 1e-6
        }
    },

    "PT_TST2": {
        "N_EPOCHS": 2,
        "BATCH_SIZE": 8,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "MASK_RATIO": 0.15,
        "OPTIMIZER": "AdamW",
        "SCHEDULER": "CosineAnnealingLR",
        "SCHEDULER_PARAMS": {
            "T_max": 2,
            "eta_min": 1e-6
        }
    },

    "T_CONTRASTIVE": {
        "N_EPOCHS": 2,
        "BATCH_SIZE": 8,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "TEMPERATURE": 0.07,
        "PROJ_HIDDEN_DIM": 64,
        "PROJ_OUTPUT_DIM": 32,
        "OPTIMIZER": "Adam",
    },

    "FINETUNING": {
        "N_EPOCHS": 2,
        "BATCH_SIZE": 8,
        "LR": 5e-5,
        "WEIGHT_DECAY": 1e-4,
        "OPTIMIZER": "AdamW",
        "SCHEDULER": "CosineAnnealingLR",
        "SCHEDULER_PARAMS": {
            "T_max": 2,
            "eta_min": 5e-7          # 5e-5 * 0.01 = 5e-7
        }
    },

    "PATIENCE": 5,
    "MIN_DELTA": 0.001,
    "ATLAS": "cc200",
    "LABEL_COL": "DX_GROUP",
    "FACTORS": ["AGE_AT_SCAN", "SEX", "EYE_STATUS_AT_SCAN"],
    "TR": 2.0,
    "PREFIX": "interp_",
    "MIN_TIMESTEPS": 116,
}
   
    print("Smoke-test: run_cross_validation with finetune + test")
    print(f"  device  : {config['DEVICE']}")
    run_cross_validation(config, n_folds=3)
    print("\n✓  Cross-validation smoke-test finished without errors.")