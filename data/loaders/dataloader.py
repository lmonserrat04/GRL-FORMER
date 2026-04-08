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


def get_dataloader(config: dict, df_subset: pd.DataFrame, split: str, normalizer=None, harmonizer=None) -> DataLoader:
    required_keys = ["CSV_PATH", "INTERP_PATH", "ATLAS", 
                     "LABEL_COL", "BATCH_SIZE", "WINDOW_SIZE", "N_ROIS"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Falta la clave requerida en config: '{key}'")
        
    allowed_splits = ['train', 'val', 'test']
    if split not in allowed_splits:
        raise ValueError("Split no reconocido, se esperaba 'train', 'val' o 'test'")
     
    # Pasamos los objetos normalizer y harmonizer al Dataset
    dataset = SingleAtlas(
        config=config,
        df_subset=df_subset, 
        split=split, 
        prefix=config.get("PREFIX", "interp_"),
        normalizer=normalizer,
        harmonizer=harmonizer
    )
    
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
        sampler=sampler,
        shuffle=False,
        worker_init_fn=seed_worker,  
        generator=g,          
    )


class SingleAtlas(Dataset):
    def __init__(self, *, config: dict, df_subset: pd.DataFrame, split: str, prefix: str, normalizer=None, harmonizer=None):
        self.cfg = config
        self.split = split

        interp_path = Path(config["INTERP_PATH"]).resolve()
        if not interp_path.exists():
            raise FileNotFoundError(f"No se encontró: {interp_path}")
        
        print(f"Cargando SingleAtlas en RAM ({self.split})...")

        self.x = []
        for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
            filename = f"{prefix}{row['FILE_ID']}_rois_{config['ATLAS']}.1D"
            filepath = interp_path / filename 
            
            self.x.append(
                torch.tensor(np.loadtxt(filepath), dtype=torch.float32)
            )

        self.labels = torch.tensor(df_subset[config["LABEL_COL"]].values, dtype=torch.long)

        additional_text = ""
        if self.split == 'train':
            self.labels_strat= torch.tensor(pd.Series(df_subset["STRATIFY"]).astype("category").cat.codes.values)
            additional_text = f"Nro labels estratificacion: {self.labels_strat.numel()}"

        # --- SECCIÓN: Limpieza Matricial Vectorizada (Normalización + Harmonización) ---
        if normalizer is not None and harmonizer is not None:
            print(f"Aplicando Normalización y Residualización masiva ({self.split})...")
            # 1. Empaquetar: self.x es lista de (R, T). torch.stack da (B, R, T). Permutamos a (B, T, R)
            X = torch.stack(self.x).permute(0, 2, 1)
            
            # 2. Hardware acceleration: Mover a GPU si existe para lstsq
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X = X.to(device)

            # 3. Flujo jerárquico estricto (Fit solo en train)
            if self.split == 'train':
                normalizer.fit(X)
                X = normalizer.transform(X)
                harmonizer.fit(X, df_subset)
                X = harmonizer.transform(X, df_subset)
            else:
                X = normalizer.transform(X)
                X = harmonizer.transform(X, df_subset)

            # 4. Desempaquetar: Retornar a CPU y a la forma original (B, R, T)
            X = X.cpu().permute(0, 2, 1)
            self.x = list(X.unbind(0))  # unbind separa el tensor 3D en una tupla/lista de tensores 2D
        # -----------------------------------------------------------------------------

        print(f"✓ Cargado: {len(self.x)} " +
              f"individuos | Labels dx únicos: {self.labels.unique().tolist() }\n"+
             additional_text)


    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x: torch.Tensor = self.x[idx]

        if self.split == 'train':
            window: torch.Tensor = torch.as_tensor(get_window(x, self.cfg["WINDOW_SIZE"], mode='random')) 
        elif self.split == 'val' or self.split == 'test':
            window: torch.Tensor = torch.as_tensor(get_window(x, self.cfg["WINDOW_SIZE"], mode='central')) 
        else:
            raise ValueError("Split no reconocido, se esperaba 'train', 'val' o 'test'")
        
        if window.shape != (self.cfg["N_ROIS"], self.cfg["WINDOW_SIZE"]):
            raise ValueError("Error en shapes al calcular ventana aleatoria en el dataset, se esperaba (N_ROIS, WINDOWS_SIZE)")

        # Retorna transponiendo (T, N_ROIS) para que el Transformer procese la secuencia temporal
        return window.T, self.labels[idx]