import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from pathlib import Path
import random
from data.augmentation.augmentation import get_window



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config: dict, df_subset: pd.DataFrame, split: str) -> DataLoader:
    required_keys = ["CSV_PATH", "INTERP_PATH", "ATLAS", 
                 "LABEL_COL", "BATCH_SIZE", "WINDOW_SIZE", "N_ROIS"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Falta la clave requerida en config: '{key}'")
        
    allowed_splits = ['train', 'val', 'test']
    if split not in allowed_splits:
        raise ValueError("Split no reconocido, se esperaba 'train', 'val' o 'test'")
     
    

    dataset = SingleAtlas(cfg=config,df_subset=df_subset , split = split, prefix="interp_")
    

    
    if split == "train":

        #______________Weighted Random Sampling solo en train____________________________________
        class_counts = torch.bincount(dataset.labels_strat)
        class_weights = 1.0 / class_counts.float()

        sample_weights = class_weights[dataset.labels_strat]  # Indexacion avanzada de PyTorch

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        

    else:
        sampler = None
 
    g = torch.Generator()
    g.manual_seed(config["SEED"])
    
   


    return DataLoader(
        dataset,
        batch_size=config["BATCH_SIZE"],
        num_workers=config.get("NUM_WORKERS", 0),
        sampler= sampler,
        shuffle=False,
        worker_init_fn=seed_worker,  
        generator=g,          
    )


class SingleAtlas(Dataset):
    def __init__(self, *, cfg: dict,df_subset: pd.DataFrame,split: str, prefix: str = "interp_"):
        self.cfg = cfg
        self.split = split

        interp_path = Path(cfg["INTERP_PATH"]).resolve()
        if not interp_path.exists():
            raise FileNotFoundError(f"No se encontró: {interp_path}")
        

        print("Cargando SingleAtlas en RAM...")

        self.x = []
        for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
           
          
            filename = f"{prefix}{row['FILE_ID']}_rois_{cfg['ATLAS']}.1D"
            filepath = interp_path / filename 
            
            self.x.append(
                torch.tensor(np.loadtxt(filepath), dtype=torch.float32)
            )

        self.labels = torch.tensor(df_subset[cfg["LABEL_COL"]].values, dtype=torch.long)

        additional_text = ""
        if self.split == 'train':
            self.labels_strat= torch.tensor(pd.Series(df_subset["STRATIFY"]).astype("category").cat.codes.values)
            additional_text = f"Nro labels estratificacion: {self.labels_strat.numel()}"

        print(f"✓ Cargado: {len(self.x)} " +
              f"individuos | Labels dx únicos: {self.labels.unique().tolist() }\n"+
             additional_text)



    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x: torch.Tensor = self.x[idx]

        if self.split == 'train':

            window: torch.Tensor = torch.as_tensor(get_window(x, self.cfg["WINDOW_SIZE"], mode= 'random')) # Comparte memoria

        elif self.split == 'val' or self.split == 'test':
            window: torch.Tensor = torch.as_tensor(get_window(x, self.cfg["WINDOW_SIZE"], mode= 'central')) # Comparte memoria

        else:
            raise ValueError("Split no reconocido, se esperaba 'train', 'val' o 'test'")
        
        if window.shape != (self.cfg["N_ROIS"],self.cfg["WINDOW_SIZE"]):
            raise ValueError("Error en shapes al calcular ventana aleatoria en el dataset, se esperaba (N_ROIS, WINDOWS_SIZE)")

        return window.T, self.labels[idx] # (T, N_ROIS)