import copy
from pathlib import Path

import torch

class EarlyStopping():
    def __init__(self, model,config:dict):
        self.best_model = copy.deepcopy(model)
        self.config = config
        self.patience = self.config["PATIENCE"]
        self.counter = 0
        self.min_delta = self.config["MIN_DELTA"]
        self.min_val_loss = float('inf')

    def __call__(self, model, avg_val_loss):
        delta = self.min_val_loss - avg_val_loss

        if delta >= self.min_delta:
           
            self.counter = 0
            self.min_val_loss = avg_val_loss          
            self.best_model = copy.deepcopy(model)   
        else:
            
            self.counter += 1
            if self.counter >= self.patience: 

                checkpoint_path = Path(self.config["CHECKPOINTS_PATH"]) / f"best_model.pt"
                torch.save(model.state_dict(), checkpoint_path)        
                return self.best_model                

        return None  