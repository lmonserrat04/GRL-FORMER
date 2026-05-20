from orchestration import run_contrastive, run_pretrain_fc, run_pretrain_ts
from utils.checkpoint import get_checkpoint_path

def run_cross_validation(config, df, n_folds=5):
    config = run_contrastive.create_experiment_dir(config)

    for fold in range(n_folds):
        df_train, df_val, df_test = split_fold(df, fold)

        run_pretrain_ts(config, df_train, df_val, df_test, fold)
        run_pretrain_fc(config, df_train, df_val, df_test, fold)

        run_contrastive(
            config, df_train, df_val, df_test, fold,
            chkpt_ts=get_checkpoint_path(config, "PT_TS", fold),
            chkpt_fc=get_checkpoint_path(config, "PT_FC", fold),
        )

        # run_finetune(
        #     config, df_train, df_val, df_test, fold,
        #     chkpt_cont=get_checkpoint_path(config, "CONT", fold),
        # )


# main_test_cross_validation.py

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset

# ── 1. Config ──────────────────────────────────────────────────────────────────
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
        "D_MODEL": 64,
        "DIM_FEEDFORWARD": 128,
        "NUM_ENCODER_LAYERS": 2,
        "N_HEADS": 4,
        "ENC_DROP": 0.1,
        "USE_CLS_TOKEN": True,
    },
    "TST2": {
        "D_MODEL": 64,
        "N_HEADS": 4,
        "NUM_ENCODER_LAYERS": 2,
        "DIM_FEEDFORWARD": 128,
        "ENC_DROP": 0.1,
        "PCC_DIM": 19900,
    },
    "DUAL_STREAM": {
        "FUSION_TYPE": "attention_pooling",
        "NUM_CLASSES": 2,
        "CLASSIFIER_DROPOUT": 0.1,
    },
    "FUSION": {
        "ATTENTION_POOLING": {"HIDDEN_DIM": 128}
    },
    "PT_TST1": {
        "N_EPOCHS": 2,
        "BATCH_SIZE": 8,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "MASK_RATIO": None,
        "T_MAX": 2,
        "ETA_MIN": 1e-6,
    },
    "PT_TST2": {
        "N_EPOCHS": 2,
        "BATCH_SIZE": 8,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "MASK_RATIO": 0.15,
    },
    "T_CONTRASTIVE": {
        "N_EPOCHS": 2,
        "BATCH_SIZE": 8,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "TEMPERATURE": 0.07,
        "PROJ_HIDDEN_DIM": 64,
        "PROJ_OUTPUT_DIM": 32,
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

# ── 2. Helpers ─────────────────────────────────────────────────────────────────
EXP_TO_CONFIG_KEY = {
    "pretrain_ts": "PT_TST1",
    "pretrain_fc": "PT_TST2",
    "contrastive": "T_CONTRASTIVE",
}

def get_batch_size(config: dict) -> int:
    key = EXP_TO_CONFIG_KEY.get(config["EXPERIMENT_TYPE"])
    if key is None:
        raise ValueError(f"No config key mapped for '{config['EXPERIMENT_TYPE']}'")
    return config[key]["BATCH_SIZE"]

# ── 3. Datasets sintéticos ─────────────────────────────────────────────────────
N_SAMPLES = 32
SEQ_LEN   = 100
N_ROIS    = 200
PCC_DIM   = config["TST2"]["PCC_DIM"]

class SyntheticTSDataset(Dataset):
    """(x: SEQ_LEN×N_ROIS, y) para pretrain_ts"""
    def __init__(self, n):
        torch.manual_seed(0)
        self.x = torch.randn(n, SEQ_LEN, N_ROIS)
        self.y = torch.randint(0, 2, (n,))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

class SyntheticFCDataset(Dataset):
    """(x: PCC_DIM, y) para pretrain_fc"""
    def __init__(self, n):
        torch.manual_seed(0)
        self.x = torch.randn(n, PCC_DIM)
        self.y = torch.randint(0, 2, (n,))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

class SyntheticDualDataset(Dataset):
    """({"timeseries":..., "pcc_vector":...}, y) para contrastive"""
    def __init__(self, n):
        torch.manual_seed(0)
        self.ts  = torch.randn(n, SEQ_LEN, N_ROIS)
        self.pcc = torch.randn(n, PCC_DIM)
        self.y   = torch.randint(0, 2, (n,))
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {"timeseries": self.ts[i], "pcc_vector": self.pcc[i]}, self.y[i]

# ── 4. Loader factories por fase ───────────────────────────────────────────────
def make_ts_loader(n, bs):
    return DataLoader(SyntheticTSDataset(n), batch_size=bs, shuffle=True, num_workers=0)

def make_fc_loader(n, bs):
    return DataLoader(SyntheticFCDataset(n), batch_size=bs, shuffle=True, num_workers=0)

def make_dual_loader(n, bs):
    return DataLoader(SyntheticDualDataset(n), batch_size=bs, shuffle=True, num_workers=0)

LOADER_FACTORY = {
    "pretrain_ts": make_ts_loader,
    "pretrain_fc": make_fc_loader,
    "contrastive": make_dual_loader,
}

# ── 5. Monkey-patch build_dataloaders ─────────────────────────────────────────
import training.setup as setup_module

def _patched_build_dataloaders(config, df_train, df_val, df_test,
                               harmonizer=None, normalizer=None):
    exp_type = config["EXPERIMENT_TYPE"]
    bs       = get_batch_size(config)
    factory  = LOADER_FACTORY[exp_type]
    return (
        factory(N_SAMPLES,      bs),
        factory(N_SAMPLES // 2, bs),
        factory(N_SAMPLES // 4, bs),
    )

setup_module.build_dataloaders = _patched_build_dataloaders

# ── 6. Monkey-patch create_dual_stream_model (skip carga de pesos) ─────────────
import models.dual_stream as dual_module

def _patched_create_dual_stream(config, name_chkpt_pt_ts=None, name_chkpt_pt_fc=None):
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
    return model

dual_module.create_dual_stream_model = _patched_create_dual_stream
try:
    setup_module.create_dual_stream_model = _patched_create_dual_stream
except AttributeError:
    pass

# ── 7. create_experiment_dir sin shutil.copy ──────────────────────────────────
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

# ── 8. split_fold sintético ────────────────────────────────────────────────────
def split_fold(df: pd.DataFrame, fold: int, n_folds: int = 3):
    """Parte el DataFrame en train/val/test por índice de fold."""
    n        = len(df)
    fold_size = n // n_folds
    val_idx  = list(range(fold * fold_size, (fold + 1) * fold_size))
    train_idx = [i for i in range(n) if i not in val_idx]
    # test = misma partición que val para el smoke (no importa solapamiento)
    return df.iloc[train_idx].reset_index(drop=True), \
           df.iloc[val_idx].reset_index(drop=True),   \
           df.iloc[val_idx].reset_index(drop=True)

# ── 9. Dummy DataFrame global ──────────────────────────────────────────────────
def make_dummy_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id":         [f"sub_{i:03d}" for i in range(n)],
        "DX_GROUP":           np.random.randint(0, 2, n),
        "AGE_AT_SCAN":        np.random.uniform(10, 60, n),
        "SEX":                np.random.randint(1, 3, n),
        "EYE_STATUS_AT_SCAN": np.random.randint(1, 3, n),
    })

df_global = make_dummy_df(N_SAMPLES)

# ── 10. Run ────────────────────────────────────────────────────────────────────
from utils.checkpoint import get_checkpoint_path
from orchestration.run_pretrain_ts import run_pretrain_ts
from orchestration.run_pretrain_fc import run_pretrain_fc
from orchestration.run_contrastive import run_contrastive

def run_cross_validation(config, df, n_folds=3):
    config = create_experiment_dir(config)

    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1} / {n_folds}")
        print(f"{'='*60}")

        df_train, df_val, df_test = split_fold(df, fold, n_folds)

        print(f"\n── Fase 1: pretrain_ts  (fold {fold}) ──")
        run_pretrain_ts(config, df_train, df_val, df_test, fold)

        print(f"\n── Fase 2: pretrain_fc  (fold {fold}) ──")
        run_pretrain_fc(config, df_train, df_val, df_test, fold)

        print(f"\n── Fase 3: contrastive  (fold {fold}) ──")
        run_contrastive(
            config, df_train, df_val, df_test, fold,
            chkpt_ts=get_checkpoint_path(config, "PT_TS", fold),
            chkpt_fc=get_checkpoint_path(config, "PT_FC", fold),
        )

if __name__ == "__main__":
    print("Smoke-test: run_cross_validation")
    print(f"  device  : {config['DEVICE']}")
    print(f"  folds   : 3")
    print(f"  samples : {N_SAMPLES}")
    run_cross_validation(config, df_global, n_folds=3)
    print("\n✓  Cross-validation smoke-test finished without errors.")