import argparse
from pathlib import Path
import torch
import yaml
import pandas as pd
from utils.seed import set_seed
from data.loaders.dataloader import get_dataloader
from model.models import build_model
from training.cross_validator import CrossValidator
from utils.experiment import create_experiment_dir


def main(args):

    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['SEED'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["DEVICE"] = device
    
    if args.mode == "train":
    
       
        config = create_experiment_dir(config)
        cv = CrossValidator(config)
        cv.run()

    # elif args.mode == "eval":
    #     data  = get_dataloader(config, split="test")
    #     model = build_model(config)
    #     model.load_state_dict(torch.load(config.checkpoint))
    #     evaluate(model, data, config)

    # elif args.mode == "infer":
    #     model = build_model(config)
    #     model.load_state_dict(torch.load(config.checkpoint))
    #     run_inference(model, config)  # lee inputs de otra fuente



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "infer"])
    parser.add_argument("--config", default="./config/config.yaml")
    args = parser.parse_args()
    main(args)