from tqdm import tqdm
from training.setup import build_experiment
from pretraining.pretrain_fc import train_one_epoch, validate
from training.callbacks import EarlyStopping
import torch.nn as nn
from utils.checkpoint import get_checkpoint_path



def run_pretrain_fc(config: dict, df_train, df_val, df_test, fold):
    config["EXPERIMENT_TYPE"] = "pretrain_fc"
    exp_ts = build_experiment(config, df_train, df_val, df_test) # diccionario que devuelve : "model", "task", "optimizer"
                                                                    # "scheduler", "train_loader","val_loader", "device"

    model = exp_ts['model']
    epochs = config["PT_TST2"]["N_EPOCHS"]
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
                mask_ratio= config["PT_TST2"]["MASK_RATIO"]

            )
            model.eval()
            # Validate
            val_running_loss+= validate(
                **exp_ts,
                mask_ratio= config["PT_TST2"]["MASK_RATIO"]
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
                    print(f"Early stopping pretraining pcc, Epoca: {epoch + 1}. Mejor loss: {early_stopping.min_val_loss:4f}")
                    break


            tepoch.set_postfix(v_loss=f"{avg_val_loss:.4f}")
            
            
            
            
            print(f"Train Loss: {train_running_loss:.4f}, Val Loss: {val_running_loss:.4f}")
            # print(f"LR: {current_lr:.6f}"

        save_path = get_checkpoint_path(config, "PT_FC", fold)
        torch.save(model.state_dict(), save_path)
        
# main_test_pretrain_fc.py

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils.experiment import create_experiment_dir

# ── 1. Config ──────────────────────────────────────────────────────────────────
config = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EXPERIMENT_TYPE": "pretrain_fc",

    "N_ROIS": 200,
    "MAX_SEQ_LEN": 100,
    "WINDOW_SIZE": 128,

    "NUM_WORKERS": 0,
    "SEED": 42,

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
        "PCC_DIM": 19900,        # <-- leído desde acá en build_model_fc
    },

    "PT_TST1": {
        "N_EPOCHS": 3,
        "BATCH_SIZE": 8,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "MASK_RATIO": None,
        "T_MAX": 3,
        "ETA_MIN": 1e-6,
    },

    "PT_TST2": {
        "N_EPOCHS": 3,
        "BATCH_SIZE": 8,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "MASK_RATIO": 0.15,
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

# ── 2. Helper batch size ───────────────────────────────────────────────────────
EXP_TO_CONFIG_KEY = {
    "pretrain_ts": "PT_TST1",
    "pretrain_fc": "PT_TST2",
}

def get_batch_size(config: dict) -> int:
    key = EXP_TO_CONFIG_KEY.get(config["EXPERIMENT_TYPE"])
    if key is None:
        raise ValueError(f"No config key mapped for '{config['EXPERIMENT_TYPE']}'")
    return config[key]["BATCH_SIZE"]

# ── 3. Synthetic DataLoader factory ───────────────────────────────────────────
N_SAMPLES = 32
PCC_DIM   = config["TST2"]["PCC_DIM"]   # 19900 — forma de cada sample FC

def make_synthetic_loader(n: int, batch_size: int) -> DataLoader:
    """
    Cada batch:
        x : (B, PCC_DIM)   float32  — vector FC aplanado (upper-triangle PCC)
        y : (B,)           long     — label binaria {0, 1}
    """
    torch.manual_seed(0)
    x  = torch.randn(n, PCC_DIM)
    y  = torch.randint(0, 2, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

# ── 4. Monkey-patch build_dataloaders ─────────────────────────────────────────
import training.setup as setup_module

def _patched_build_dataloaders(config, df_train, df_val, df_test,
                               harmonizer=None, normalizer=None):
    bs = get_batch_size(config)
    return (
        make_synthetic_loader(N_SAMPLES,       bs),
        make_synthetic_loader(N_SAMPLES // 2,  bs),
        make_synthetic_loader(N_SAMPLES // 4,  bs),
    )

setup_module.build_dataloaders = _patched_build_dataloaders

# ── 5. Dummy DataFrames ────────────────────────────────────────────────────────
def make_dummy_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id":         [f"sub_{i:03d}" for i in range(n)],
        "DX_GROUP":           np.random.randint(0, 2, n),
        "AGE_AT_SCAN":        np.random.uniform(10, 60, n),
        "SEX":                np.random.randint(1, 3, n),
        "EYE_STATUS_AT_SCAN": np.random.randint(1, 3, n),
    })

df_train = make_dummy_df(N_SAMPLES)
df_val   = make_dummy_df(N_SAMPLES // 2)
df_test  = make_dummy_df(N_SAMPLES // 4)

# ── 6. Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Smoke-test : run_pretrain_fc  (synthetic data)")
    print(f"  device   : {config['DEVICE']}")
    print(f"  epochs   : {config['PT_TST2']['N_EPOCHS']}")
    print(f"  batch    : {get_batch_size(config)}")
    print(f"  pcc_dim  : {PCC_DIM}")
    print("=" * 60)

    config = create_experiment_dir(config)
    run_pretrain_fc(config, df_train, df_val, df_test,0)

    print("\n✓  Smoke-test finished without errors.")
