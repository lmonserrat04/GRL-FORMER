from pathlib import Path
from tqdm import tqdm
from training.setup import build_experiment
from pretraining.pretrain_ts import train_one_epoch, validate
from training.callbacks import EarlyStopping
import torch.nn as nn
from utils.checkpoint import get_checkpoint_path


def run_pretrain_ts(config: dict, df_train, df_val, df_test, fold):
    config["EXPERIMENT_TYPE"] = "pretrain_ts"
    exp_ts = build_experiment(config, df_train, df_val, df_test) # diccionario que devuelve : "model", "task", "optimizer"
                                                                    # "scheduler", "train_loader","val_loader", "device"

    model = exp_ts['model']
    epochs = config["PT_TST1"]["N_EPOCHS"]
    early_stopping = EarlyStopping(model,config)

    train_losses = []
    val_losses = []

    

    
    with tqdm(range(epochs), unit="epoch") as tepoch:
        
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            train_running_loss = 0.0
            val_running_loss = 0.0
            model.train()
            
            # Train
            train_running_loss+= train_one_epoch(
                **exp_ts,
                mask_ratio= config["PT_TST1"]["MASK_RATIO"]

            )
            model.eval()
            # Validate
            val_running_loss+= validate(
                **exp_ts,
                mask_ratio= config["PT_TST1"]["MASK_RATIO"]
            )

            # Update learning rate
            exp_ts['scheduler'].step()

            avg_train_loss = train_running_loss/ len(exp_ts["train_loader"])
            avg_val_loss = val_running_loss / len(exp_ts["val_loader"])
            
            # Record losses
            train_losses.append(train_running_loss)
            val_losses.append(val_running_loss)

            best: nn.Module | None = early_stopping(model,avg_val_loss)

            if best:
                model.load_state_dict(early_stopping.best_model.state_dict())
                print(f"Early stopping pretraining timeseries, Epoca: {epoch + 1}. Mejor loss: {early_stopping.min_val_loss:4f}")
                break


            tepoch.set_postfix(v_loss=f"{avg_val_loss:.4f}")

            print(f"Train Loss: {train_running_loss:.4f}, Val Loss: {val_running_loss:.4f}")
            # print(f"LR: {current_lr:.6f}"

        save_path = get_checkpoint_path(config, "PT_TS", fold)
        torch.save(model.state_dict(), save_path)


# main_test_pretrain_ts.py
"""
Synthetic smoke-test for run_pretrain_ts.
Bypasses build_experiment's real dataloaders by monkey-patching
build_dataloaders with a factory that returns torch DataLoaders
built from random tensors.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ── 1.  Minimal config (mirrors config.yaml) ──────────────────────────────────
config = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EXPERIMENT_TYPE": "pretrain_ts",

    # Dataset dims
    "N_ROIS": 200,
    "MAX_SEQ_LEN": 100,   # synthetic ts length
    "WINDOW_SIZE": 128,
    "PCC_DIM": 19900,

    # DataLoader
    "BATCH_SIZE": 8,
    "NUM_WORKERS": 0,      # keep 0 for tests (avoids fork issues)
    "SEED": 42,

    # Architecture
    "TST1": {
        "D_MODEL": 64,          # smaller → faster test
        "DIM_FEEDFORWARD": 128,
        "NUM_ENCODER_LAYERS": 2,
        "N_HEADS": 4,
        "ENC_DROP": 0.1,
        "USE_CLS_TOKEN": True,
    },

    # Pre-training loop
    "PT_TST1": {
        "N_EPOCHS": 3,          # just a few epochs to smoke-test
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "MASK_RATIO": None,
        "T_MAX": 3,
        "ETA_MIN": 1e-6,
    },

    # Callbacks
    "PATIENCE": 5,
    "MIN_DELTA": 0.001,

    # Misc (referenced inside build_experiment)
    "ATLAS": "cc200",
    "LABEL_COL": "DX_GROUP",
    "FACTORS": ["AGE_AT_SCAN", "SEX", "EYE_STATUS_AT_SCAN"],
    "TR": 2.0,
    "PREFIX": "interp_",
    "MIN_TIMESTEPS": 116,
}

# ── 2.  Synthetic DataLoader factory ──────────────────────────────────────────
N_SAMPLES   = 32   # subjects per split
SEQ_LEN     = 100  # time steps  (shape[0] per sample)
N_ROIS      = 200  # ROIs        (shape[1] per sample)

def make_synthetic_loader(n: int, batch_size: int) -> DataLoader:
    """
    Returns a DataLoader whose batches look like:
        x  : (B, SEQ_LEN, N_ROIS)   float32
        y  : (B,)                    long  {0, 1}
    Adjust to match what your Dataset.__getitem__ actually returns.
    """
    torch.manual_seed(0)
    x = torch.randn(n, SEQ_LEN, N_ROIS)
    y = torch.randint(0, 2, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

# ── 3.  Monkey-patch build_dataloaders before importing build_experiment ───────
import training.setup as setup_module   # import the module first …

def _patched_build_dataloaders(config, df_train, df_val, df_test,
                               harmonizer=None, normalizer=None):
    bs = config["BATCH_SIZE"]
    return (
        make_synthetic_loader(N_SAMPLES, bs),   # train
        make_synthetic_loader(N_SAMPLES // 2, bs),  # val
        make_synthetic_loader(N_SAMPLES // 4, bs),  # test
    )

setup_module.build_dataloaders = _patched_build_dataloaders  # … then patch it

# ── 4.  Dummy DataFrames (build_experiment receives them but won't use them) ──
def make_dummy_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id":      [f"sub_{i:03d}" for i in range(n)],
        "DX_GROUP":        np.random.randint(0, 2, n),
        "AGE_AT_SCAN":     np.random.uniform(10, 60, n),
        "SEX":             np.random.randint(1, 3, n),
        "EYE_STATUS_AT_SCAN": np.random.randint(1, 3, n),
    })

df_train = make_dummy_df(N_SAMPLES)
df_val   = make_dummy_df(N_SAMPLES // 2)
df_test  = make_dummy_df(N_SAMPLES // 4)

# ── 5.  Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Smoke-test: run_pretrain_ts  (synthetic data)")
    print(f"  device : {config['DEVICE']}")
    print(f"  epochs : {config['PT_TST1']['N_EPOCHS']}")
    print(f"  batch  : {config['BATCH_SIZE']}")
    print("=" * 60)

    run_pretrain_ts(config, df_train, df_val, df_test, fold = 0)

    print("\n✓  Smoke-test finished without errors.")