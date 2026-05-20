# training/utils.py

from pathlib import Path

def get_checkpoint_path(config: dict, phase: str, fold: int) -> Path:
    """
    Construye el path de un checkpoint desde config.
    
    Uso:
        get_checkpoint_path(config, "PT_TS", fold=0)
        → experiments/.../checkpoints/best_pt_ts_model_fold_0.pt
    """
    prefix   = config["CHECKPOINTS"][phase]        # lee el prefijo del yaml
    chkpt_dir = Path(config["CHECKPOINTS_PATH"])
    return chkpt_dir / f"{prefix}_fold_{fold}.pt"