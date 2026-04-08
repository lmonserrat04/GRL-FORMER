
import numpy as np
import pandas as pd

import torch

class ResidualHarmonizer:
    def __init__(self, factors, site_col_name: str = "SITE_ID"):
        self.factors = factors
        self.site_col_name = site_col_name
        self.confound_cols = None
        self.betas = None  # Almacenará los pesos del modelo lineal

    def fit(self, X: torch.Tensor, df: pd.DataFrame):

        #X -> (N, T, N_ROIS)
        site_dummies = pd.get_dummies(df[self.site_col_name], drop_first=True)
        confounds = pd.concat([site_dummies, df[self.factors]], axis=1)
        self.confound_cols = confounds.columns

        # Convertir a tensor en el mismo dispositivo que X (CPU/GPU)
        C = torch.tensor(confounds.values, dtype=torch.float32, device=X.device)
        
        # Añadir columna de 1s para calcular el Intercepto (sesgo)
        intercept = torch.ones((C.shape[0], 1), dtype=torch.float32, device=X.device)
        C = torch.cat([intercept, C], dim=1)

        B, T, R = X.shape
        X_reshaped = X.reshape(B, -1) 

        # Resuelve la regresión lineal masiva de forma vectorizada
        self.betas = torch.linalg.lstsq(C, X_reshaped).solution
        
        return self

    def transform(self, X: torch.Tensor, df: pd.DataFrame):
        site_dummies = pd.get_dummies(df[self.site_col_name], drop_first=True)
        confounds = pd.concat([site_dummies, df[self.factors]], axis=1)
        confounds = confounds.reindex(columns=self.confound_cols, fill_value=0)
        
        # Convertir a tensor y añadir intercepto
        C = torch.tensor(confounds.values, dtype=torch.float32, device=X.device)
        intercept = torch.ones((C.shape[0], 1), dtype=torch.float32, device=X.device)
        C = torch.cat([intercept, C], dim=1)

        B, T, R = X.shape

        # Predecimos el ruido mediante multiplicación matricial directa
        predicted_noise = torch.matmul(C, self.betas)
        predicted_noise = predicted_noise.reshape(B, T, R)
        
        return X - predicted_noise   

class GlobalNormalizer:
    """Normaliza todas las series basándose en los estadísticos globales del Train"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: torch.Tensor):
        # X shape esperado: (B, T, R)
        # Calculamos sobre el batch (dim 0) y el tiempo (dim 1), conservando las ROIs
        self.mean_ = X.mean(dim=(0, 1), keepdim=True)
        self.std_ = X.std(dim=(0, 1), keepdim=True)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: torch.Tensor):
        return (X - self.mean_) / self.std_