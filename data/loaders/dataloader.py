import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from pathlib import Path
import random
from data.augmentation.augmentation import get_random_window



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config: dict, split: str) -> DataLoader:
    required_keys = ["CSV_PATH", "INTERP_PATH", "ATLAS", 
                 "LABEL_COL", "BATCH_SIZE", "WINDOWS_SIZE", "N_ROIS"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Falta la clave requerida en config: '{key}'")

    dataset = SingleAtlas(cfg=config, prefix="interp_")

    #______________Weighted Random Sampling solo en train____________________________________
    class_counts = torch.bincount(dataset.labels_strat)
    class_weights = 1.0 / class_counts.float()

    sample_weights = class_weights[dataset.labels_strat]  # Indexacion avanzada de PyTorch

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )




   
    g = torch.Generator()
    g.manual_seed(config["SEED"])

    return DataLoader(
        dataset,
        batch_size=config["BATCH_SIZE"],
        num_workers=config.get("NUM_WORKERS", 0),
        sampler= sampler,
        worker_init_fn=seed_worker,  
        generator=g,          
    )


class SingleAtlas(Dataset):
    def __init__(self, *, cfg: dict, prefix: str = "interp_"):
        self.cfg = cfg

        interp_path = Path(cfg["INTERP_PATH"]).resolve()
        if not interp_path.exists():
            raise FileNotFoundError(f"No se encontró: {interp_path}")

        df: pd.DataFrame = pd.read_csv(cfg["CSV_PATH"])
        df["STRATIFY"] = df['SITE_ID'].astype(str) + '_' + df['DX_GROUP'].astype(str)

        print("Cargando SingleAtlas en RAM...")

        self.x = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
          
            filename = f"{prefix}{row[cfg['FILE_ID']]}_rois_{cfg['ATLAS']}.1D"
            filepath = interp_path / filename 
            
            self.x.append(
                torch.tensor(np.loadtxt(filepath), dtype=torch.float32)
            )

        self.labels = torch.tensor(df[cfg["LABEL_COL"]].values, dtype=torch.long)
        self.labels_strat= torch.tensor(pd.Series(df["STRATIFY"]).astype("category").cat.codes.values)

        print(f"✓ Cargado: {len(self.x)}" +
              f"individuos | Labels dx únicos: {self.labels.unique().tolist() }\n"+
              f"Nro labels estratificacion: {self.labels_strat.numel()}")



    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x: torch.Tensor = self.x[idx]
        
        window: torch.Tensor = torch.as_tensor(get_random_window(x, self.cfg["WINDOWS_SIZE"])) # Comparte memoria
        
        if window.shape != (self.cfg["N_ROIS"],self.cfg["WINDOWS_SIZE"]):
            raise ValueError("Error en shapes al calcular ventana aleatoria en el dataset, se esperaba (N_ROIS, WINDOWS_SIZE)")

        return window, self.labels[idx]