"""Smoke test: 30 sujetos, 2 epochs por fase, kfold 3."""
import yaml, numpy as np, torch, pandas as pd
from pathlib import Path
from datetime import datetime

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Reducir todo a la mínima expresión
config["PT_TST1"]["N_EPOCHS"] = 2
config["PT_TST1"]["PATIENCE"] = 10
config["PT_TST1"]["BATCH_SIZE"] = 8
config["PT_TST2"]["N_EPOCHS"] = 2
config["PT_TST2"]["PATIENCE"] = 10
config["PT_TST2"]["BATCH_SIZE"] = 8
config["T_CONTRASTIVE"]["N_EPOCHS"] = 2
config["T_CONTRASTIVE"]["BATCH_SIZE"] = 8
config["FINETUNING"]["N_EPOCHS"] = 2
config["FINETUNING"]["PATIENCE"] = 10
config["FINETUNING"]["BATCH_SIZE"] = 8
config["N_FOLDS"] = 3
config["EVAL_PROTOCOL"] = "kfold"
config["NUM_WORKERS"] = 0

# Subset del CSV a 30 sujetos
df = pd.read_csv(config["CSV_PATH"])
df_sub = df.sample(n=min(30, len(df)), random_state=42).reset_index(drop=True)
tmp_csv = Path("experiments_test/meta_subset.csv")
tmp_csv.parent.mkdir(exist_ok=True)
df_sub.to_csv(tmp_csv, index=False)
config["CSV_PATH"] = str(tmp_csv)

# Directorio de salida
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = Path(f"experiments_test/dryrun_{ts}")
(exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
config["EXP_DIR"] = str(exp_dir)
config["CHECKPOINTS_PATH"] = str(exp_dir / "checkpoints")

torch.manual_seed(config["SEED"])
np.random.seed(config["SEED"])

# === Ejecutar las 4 fases ===
from training.train_pretrain_ts import run_pretrain_ts
from training.train_pretrain_fc import run_pretrain_fc
from training.train_contrastive import run_contrastive_global
from training.train_finetune import finetune_fold

ckpt = Path(config["CHECKPOINTS_PATH"])

print("\n>>> FASE 1: Pretrain TST1")
run_pretrain_ts(config, fold_idx=0, save_dir=ckpt)
config["CKPT_TST1"] = str(ckpt / "best_pt_ts_fold_0.pt")

print("\n>>> FASE 2: Pretrain TST2")
run_pretrain_fc(config, fold_idx=0, save_dir=ckpt)
config["CKPT_TST2"] = str(ckpt / "best_pt_fc_fold_0.pt")

print("\n>>> FASE 3: Contrastive GLOBAL")
run_contrastive_global(config, save_dir=ckpt)
config["CKPT_CONTRASTIVE"] = str(ckpt / "contrastive_global.pt")

print("\n>>> FASE 4: Finetune fold 0")
m = finetune_fold(config, fold_idx=0, save_dir=ckpt)
print(f"\nMétricas fold 0: AUC={m['auc']:.4f}  ACC={m['accuracy']:.4f}")

print("\n✅ SMOKE TEST OK")
