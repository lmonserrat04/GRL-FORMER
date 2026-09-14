import yaml, numpy as np, torch, pandas as pd
from pathlib import Path
from datetime import datetime

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

config["T_CONTRASTIVE"]["N_EPOCHS"] = 20
config["T_CONTRASTIVE"]["BATCH_SIZE"] = 32
config["NUM_WORKERS"] = 0

df = pd.read_csv(config["CSV_PATH"])
tmp = Path("experiments_test/meta_subset.csv"); tmp.parent.mkdir(exist_ok=True)
df.sample(n=100, random_state=42).reset_index(drop=True).to_csv(tmp, index=False)
config["CSV_PATH"] = str(tmp)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = Path(f"experiments_test/contrastive_test_{ts}")
(exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
config["CHECKPOINTS_PATH"] = str(exp_dir / "checkpoints")

torch.manual_seed(config["SEED"]); np.random.seed(config["SEED"])

from training.train_pretrain_ts import run_pretrain_ts
from training.train_pretrain_fc import run_pretrain_fc
from training.train_contrastive import run_contrastive_global

ckpt = Path(config["CHECKPOINTS_PATH"])
run_pretrain_ts(config, fold_idx=0, save_dir=ckpt)
config["CKPT_TST1"] = str(ckpt / "best_pt_ts_fold_0.pt")
run_pretrain_fc(config, fold_idx=0, save_dir=ckpt)
config["CKPT_TST2"] = str(ckpt / "best_pt_fc_fold_0.pt")

print("\n>>> CONTRASTIVE 20 epochs, 100 sujetos, batch=32")
run_contrastive_global(config, save_dir=ckpt)
print("\n✅ test_contrastive OK")
