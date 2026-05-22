# main.py
import argparse
from pathlib import Path
import torch
import yaml
from utils.seed import set_seed
from training.cross_validator_template import run_cross_validation

def main(args):
    with open(args.config, "r", encoding = 'utf-8') as f:
        config = yaml.safe_load(f)

   
    
    config.setdefault("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["SEED"])

    if args.mode == "train":
        print(f"Starting cross-validation with config: {args.config}")
        # Parámetros opcionales (pueden ir en el YAML o pasarse aquí)
        n_folds = config.get("N_FOLDS", 3)
        n_samples = config.get("N_SAMPLES_SYNTH", 32)
        run_cross_validation(config, n_folds=n_folds, n_samples=n_samples)

    # elif args.mode == "eval":
    #     ... (tu código futuro)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train")
    parser.add_argument("--config", default="./config/config.yaml")
    args = parser.parse_args()
    main(args)