from pathlib import Path
import torch
#from training.callbacks import EarlyStopping
from training.setup import build_experiment
from training.tasks.contrastive import ContrastiveTask
from tqdm import tqdm
from utils.checkpoint import get_checkpoint_path


def run_contrastive(
    config: dict, df_train, df_val, df_test, fold, chkpt_ts, chkpt_fc
):
    """
    """

    config["EXPERIMENT_TYPE"] = "contrastive"
    task : ContrastiveTask

    exp = build_experiment(config, df_train, df_val, df_test, chkpt_ts = chkpt_ts, chkpt_fc = chkpt_fc)
    model       = exp["model"]
    task: ContrastiveTask    = exp["task"]
    optimizer   = exp["optimizer"]
    scheduler   = exp["scheduler"]
    train_loader= exp["train_loader"]
    val_loader  = exp["val_loader"]
    device      = exp["device"]
    # diccionario que devuelve : "model", "task", "optimizer"
    # "scheduler", "train_loader","val_loader", "device"
    
    epochs = config["T_CONTRASTIVE"]["N_EPOCHS"]
    #early_stopping = EarlyStopping(model,config)
    
    model.train()
    task.contrastive_module.train()
    losses = []
    
    with tqdm(range(epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Contrastive Epoch {epoch+1}")
            epoch_loss = 0.0
            num_batches = 0
            
            
            for batch,_ in train_loader:
                timeseries = batch['timeseries'].to(device)
                pcc_vector = batch['pcc_vector'].to(device)
                
                # Calcular pérdida contrastiva
                optimizer.zero_grad()
                loss = task.execution_step(model,timeseries, pcc_vector)
                
                # Retropropagación
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            tepoch.set_postfix(contr_loss=f"{avg_loss:.4f}")
            losses.append(avg_loss)
        
        save_path = get_checkpoint_path(config, "CONT", fold)
        torch.save(model.state_dict(), save_path)

    return losses

# main_test_contrastive.py
"""
Smoke-test para run_contrastive con datos sintéticos.
El loader devuelve ({"timeseries": ..., "pcc_vector": ...}, label)
para respetar la estructura que espera el loop de entrenamiento.
"""

import shutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset

# ── 1. Config ──────────────────────────────────────────────────────────────────
config = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EXPERIMENT_TYPE": "contrastive",
    "RUN_NAME": "smoke_test_contrastive",

    "N_ROIS": 200,
    "MAX_SEQ_LEN": 100,
    "WINDOW_SIZE": 128,
    "NUM_WORKERS": 0,
    "SEED": 42,

    # paths de checkpoints preentrenados (usaremos dummies en el test)
    "CHKPT_TST1": None,   # None → skip carga de pesos
    "CHKPT_TST2": None,

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
        "ATTENTION_POOLING": {
            "HIDDEN_DIM": 128,
        }
    },
    "T_CONTRASTIVE": {
        "N_EPOCHS": 2,          # mínimo para smoke-test
        "BATCH_SIZE": 8,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "TEMPERATURE": 0.07,
        "PROJ_HIDDEN_DIM": 64,
        "PROJ_OUTPUT_DIM": 32,
    },
    "PT_TST1": {               # necesario para EXP_TO_CONFIG_KEY
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

# ── 2. Batch size helper ───────────────────────────────────────────────────────
EXP_TO_CONFIG_KEY = {
    "pretrain_ts":  "PT_TST1",
    "pretrain_fc":  "PT_TST2",
    "contrastive":  "T_CONTRASTIVE",
}

def get_batch_size(config: dict) -> int:
    key = EXP_TO_CONFIG_KEY.get(config["EXPERIMENT_TYPE"])
    if key is None:
        raise ValueError(f"No config key mapped for '{config['EXPERIMENT_TYPE']}'")
    return config[key]["BATCH_SIZE"]

# ── 3. Dataset sintético con la estructura que espera el loop ─────────────────
N_SAMPLES = 32
SEQ_LEN   = 100
N_ROIS    = config["TST1"]["D_MODEL"]          # no: usar N_ROIS real
N_ROIS    = 200
PCC_DIM   = config["TST2"]["PCC_DIM"]          # 19900

class SyntheticDualDataset(Dataset):
    """
    Devuelve ({"timeseries": Tensor(SEQ_LEN, N_ROIS),
               "pcc_vector": Tensor(PCC_DIM)}, label)
    para que el loop pueda hacer:
        batch['timeseries'], batch['pcc_vector']
    """
    def __init__(self, n: int):
        torch.manual_seed(0)
        self.ts  = torch.randn(n, SEQ_LEN, N_ROIS)
        self.pcc = torch.randn(n, PCC_DIM)
        self.y   = torch.randint(0, 2, (n,))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            {"timeseries": self.ts[idx], "pcc_vector": self.pcc[idx]},
            self.y[idx],
        )

def make_synthetic_loader(n: int, batch_size: int) -> DataLoader:
    return DataLoader(
        SyntheticDualDataset(n),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

# ── 4. Monkey-patch build_dataloaders ─────────────────────────────────────────
import training.setup as setup_module

def _patched_build_dataloaders(config, df_train, df_val, df_test,
                               harmonizer=None, normalizer=None):
    bs = get_batch_size(config)
    return (
        make_synthetic_loader(N_SAMPLES,      bs),
        make_synthetic_loader(N_SAMPLES // 2, bs),
        make_synthetic_loader(N_SAMPLES // 4, bs),
    )

setup_module.build_dataloaders = _patched_build_dataloaders

# ── 5. Monkey-patch create_dual_stream_model para saltear carga de pesos ──────
#    En el test CHKPT_TST1/TST2 son None, así que parcheamos para no explotar
import models.dual_stream as dual_module   # ajusta el import path si difiere

_original_create = dual_module.create_dual_stream_model

def _patched_create_dual_stream_model(config, name_chkpt_pt_ts=None,
                                      name_chkpt_pt_fc=None):
    from models.dual_stream import DualStreamModel
    model = DualStreamModel(config)
    if name_chkpt_pt_ts:
        model.load_pretrained_tst1(name_chkpt_pt_ts)
        print(f"  Loaded TST1 weights from {name_chkpt_pt_ts}")
    else:
        print("  [smoke] Skipping TST1 checkpoint load (None)")
    if name_chkpt_pt_fc:
        model.load_pretrained_tst2(name_chkpt_pt_fc)
        print(f"  Loaded TST2 weights from {name_chkpt_pt_fc}")
    else:
        print("  [smoke] Skipping TST2 checkpoint load (None)")
    return model

dual_module.create_dual_stream_model = _patched_create_dual_stream_model
# también parchear la referencia dentro de setup si fue importada con from...import
try:
    setup_module.create_dual_stream_model = _patched_create_dual_stream_model
except AttributeError:
    pass

# ── 6. create_experiment_dir sintético (no copia config.yaml real) ────────────
def create_experiment_dir(config: dict) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = config.get("RUN_NAME", "exp")
    exp_id    = f"{timestamp}_{run_name}"

    exp_dir = Path("experiments") / exp_id
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    config["LOGS_PATH"]        = str(exp_dir / "logs")
    config["CHECKPOINTS_PATH"] = str(exp_dir / "checkpoints")
    config["RESULTS_PATH"]     = str(exp_dir / "results.csv")
    config["EXP_ID"]           = exp_id
    return config

# ── 7. Dummy DataFrames ────────────────────────────────────────────────────────
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

# ── 8. Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = create_experiment_dir(config)

    print("=" * 60)
    print("Smoke-test : run_contrastive  (synthetic data)")
    print(f"  device   : {config['DEVICE']}")
    print(f"  epochs   : {config['T_CONTRASTIVE']['N_EPOCHS']}")
    print(f"  batch    : {get_batch_size(config)}")
    print(f"  ts shape : ({SEQ_LEN}, {N_ROIS})")
    print(f"  pcc_dim  : {PCC_DIM}")
    print(f"  chkpts   : None (smoke mode)")
    print("=" * 60)

    #losses = run_contrastive(config, df_train, df_val, df_test, fold=0)

    #print(f"\n  Losses por epoch: {[f'{l:.4f}' for l in losses]}")
    print("✓  Smoke-test finished without errors.")