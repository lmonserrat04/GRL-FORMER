from datetime import datetime
import shutil
from pathlib import Path

# ├── experiments/
# │   └── {timestamp}_{run_name}/        # ej: 20260406_143022_transformer_baseline
# │       ├── config.yaml                # copia exacta del config usado
# │       ├── logs/
# │       │   ├── train_log_fold1.txt
# │       │   └── test_log_fold1.txt
# │       ├── checkpoints/
# │       │   └── best_model_fold1.pt
# │       └── results.csv

def create_experiment_dir(config: dict) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = config.get("RUN_NAME", "exp")
    exp_id    = f"{timestamp}_{run_name}"

    exp_dir = Path("experiments") / exp_id
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Guardar copia del config usado
    shutil.copy("config/config.yaml", exp_dir / "config.yaml")

    config["LOGS_PATH"]         = str(exp_dir / "logs")
    config["CHECKPOINTS_PATH"]  = str(exp_dir / "checkpoints")
    config["RESULTS_PATH"]      = str(exp_dir / "results.csv")
    config["EXP_ID"]            = exp_id

    return config