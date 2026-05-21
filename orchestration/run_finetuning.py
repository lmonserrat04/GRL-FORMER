from pathlib import Path
import torch
from tqdm import tqdm
from training.setup import build_experiment
from training.train_finetune import train_one_epoch, validate
from training.callbacks import EarlyStopping
import torch.nn as nn
from utils.checkpoint import get_checkpoint_path

def run_finetuning(config: dict, df_train, df_val, df_test, fold, chkpt_cont):
    config["EXPERIMENT_TYPE"] = "finetune"
    exp_ts = build_experiment(config, df_train, df_val, df_test, chkpt_cont= chkpt_cont) # diccionario que devuelve : "model", "task", "optimizer"
                                                                    # "scheduler", "train_loader","val_loader", "device"

    model = exp_ts['model']
    epochs = config["FINETUNING"]["N_EPOCHS"]
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
            )
            model.eval()
            # Validate
            val_running_loss+= validate(
                **exp_ts,
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

        save_path = get_checkpoint_path(config, "FINETUNE", fold)
        torch.save(model.state_dict(), save_path)   


# main_test_finetuning.py

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset

# ── 1. Config (solo lo que necesita run_finetuning) ───────────────────────────
config = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EXPERIMENT_TYPE": "finetune",
    "RUN_NAME": "smoke_test_finetune",

    "N_ROIS": 200,
    "MAX_SEQ_LEN": 100,
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
    "FINETUNING": {
        "N_EPOCHS": 2,
        "BATCH_SIZE": 8,
        "LR": 5e-5,
        "WEIGHT_DECAY": 1e-4,
    },
    "PATIENCE": 3,
    "MIN_DELTA": 0.001,
}

# ── 2. Helpers ─────────────────────────────────────────────────────────────────
EXP_TO_CONFIG_KEY = {
    "pretrain_ts": "PT_TST1",
    "pretrain_fc": "PT_TST2",
    "contrastive": "T_CONTRASTIVE",
    "finetune":    "FINETUNING",
}

def get_batch_size(config: dict) -> int:
    key = EXP_TO_CONFIG_KEY.get(config["EXPERIMENT_TYPE"])
    if key is None:
        raise ValueError(f"No config key mapped for '{config['EXPERIMENT_TYPE']}'")
    return config[key]["BATCH_SIZE"]

# ── 3. Dataset sintético dual (finetune recibe los dos streams) ────────────────
N_SAMPLES = 32
SEQ_LEN   = 100
N_ROIS    = 200
PCC_DIM   = config["TST2"]["PCC_DIM"]

class SyntheticDualDataset(Dataset):
    def __init__(self, n):
        torch.manual_seed(0)
        self.ts  = torch.randn(n, SEQ_LEN, N_ROIS)
        self.pcc = torch.randn(n, PCC_DIM)
        self.y   = torch.randint(0, 2, (n,))

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return {"timeseries": self.ts[i], "pcc_vector": self.pcc[i], "label": self.y[i]}

def make_dual_loader(n, bs):
    return DataLoader(SyntheticDualDataset(n), batch_size=bs,
                      shuffle=True, num_workers=0)

# ── 4. Monkey-patch build_dataloaders ─────────────────────────────────────────
import training.setup as setup_module

def _patched_build_dataloaders(config, df_train, df_val, df_test,
                               harmonizer=None, normalizer=None):
    bs = get_batch_size(config)
    return (
        make_dual_loader(N_SAMPLES,      bs),
        make_dual_loader(N_SAMPLES // 2, bs),
        make_dual_loader(N_SAMPLES // 4, bs),
    )

setup_module.build_dataloaders = _patched_build_dataloaders

# ── 5. Monkey-patch create_dual_stream_model ──────────────────────────────────
import models.dual_stream as dual_module
from models.dual_stream import DualStreamModel

def _patched_create_dual_stream(config, name_chkpt_pt_ts=None,
                                name_chkpt_pt_fc=None,
                                name_chkpt_cont=None):
    model = DualStreamModel(config)
    if name_chkpt_pt_ts and name_chkpt_pt_fc:
        model.load_pretrained_tst1(name_chkpt_pt_ts)
        model.load_pretrained_tst2(name_chkpt_pt_fc)
        print(f"  Loaded TST1 ← {name_chkpt_pt_ts}")
        print(f"  Loaded TST2 ← {name_chkpt_pt_fc}")
    elif name_chkpt_cont:
        model.load_pretrained_contrastive(name_chkpt_cont)
        print(f"  Loaded contrastive ← {name_chkpt_cont}")
    else:
        raise ValueError("Missing checkpoint names")
    return model

dual_module.create_dual_stream_model = _patched_create_dual_stream
try:
    setup_module.create_dual_stream_model = _patched_create_dual_stream
except AttributeError:
    pass

# ── 6. create_experiment_dir sintético ────────────────────────────────────────
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

# ── 7. Checkpoint dummy ────────────────────────────────────────────────────────
def create_dummy_checkpoint(config: dict, fold: int) -> Path:
    """
    Instancia DualStreamModel con pesos aleatorios y lo guarda en disco.
    run_finetuning lo cargará como si fuera el resultado de run_contrastive.
    """
    from utils.checkpoint import get_checkpoint_path
    model     = DualStreamModel(config)
    save_path = get_checkpoint_path(config, "CONT", fold)
    torch.save(model.state_dict(), save_path)
    print(f"  Dummy contrastive checkpoint → {save_path}")
    return save_path

# ── 8. Dummy DataFrames ────────────────────────────────────────────────────────
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

# ── 9. Run ─────────────────────────────────────────────────────────────────────
from utils.checkpoint import get_checkpoint_path

if __name__ == "__main__":
    FOLD = 0
    config = create_experiment_dir(config)

    # Crear checkpoint dummy para que load_pretrained_contrastive no explote
    chkpt_cont = create_dummy_checkpoint(config, fold=FOLD)

    print("=" * 60)
    print("Smoke-test : run_finetuning  (synthetic data)")
    print(f"  device   : {config['DEVICE']}")
    print(f"  epochs   : {config['FINETUNING']['N_EPOCHS']}")
    print(f"  batch    : {get_batch_size(config)}")
    print(f"  chkpt    : {chkpt_cont}")
    print("=" * 60)

    run_finetuning(config, df_train, df_val, df_test, fold=FOLD,
                   chkpt_cont=chkpt_cont)

    print("\n✓  Smoke-test finished without errors.")

