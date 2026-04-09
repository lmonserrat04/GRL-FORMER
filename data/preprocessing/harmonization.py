import numpy as np
import pandas as pd
import torch

class ResidualHarmonizer:
    def __init__(self, factors, site_col_name: str = "SITE_ID"):
        self.factors = factors
        self.site_col_name = site_col_name
        self.confound_cols = None
        self.betas = None  

    def _prepare_confounds(self, df: pd.DataFrame, device, include_target: bool = False):
        # 1. Site dummies
        site_dummies = pd.get_dummies(df[self.site_col_name], drop_first=True).astype(float)
        
        # 2. Factores numéricos
        factors_df = df[self.factors].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        
        # --- CAMBIO AQUÍ: Incluir la etiqueta de diagnóstico si estamos en el FIT ---
        if include_target:
            target = df["DX_GROUP"].astype(float) # O el nombre de tu columna de etiquetas
            confounds = pd.concat([site_dummies, factors_df, target], axis=1)
        else:
            confounds = pd.concat([site_dummies, factors_df], axis=1)
        # -------------------------------------------------------------------------

        confounds_np = confounds.to_numpy(dtype=np.float32)
        C = torch.tensor(confounds_np, dtype=torch.float32, device=device)
        
        intercept = torch.ones((C.shape[0], 1), dtype=torch.float32, device=device)
        C = torch.cat([intercept, C], dim=1)
        return C

        
    def fit(self, X: torch.Tensor, df: pd.DataFrame, mask: torch.Tensor):
        """
        X: (B, T, R) - Datos con padding
        df: DataFrame con metadatos de los B sujetos
        mask: (B, T) - Booleano (True para datos reales, False para padding)
        """
        C = self._prepare_confounds(df, X.device,include_target=True)
        B, T, R = X.shape
        K = C.shape[1] # Número de confounds + intercepto

        # 1. Expandir C para que cada paso de tiempo tenga sus factores: (B, K) -> (B, T, K)
        C_expanded = C.unsqueeze(1).expand(-1, T, -1).reshape(B * T, K)
        
        # 2. Aplanar X: (B, T, R) -> (B * T, R)
        X_flattened = X.reshape(B * T, R)
        
        # 3. Aplanar Máscara: (B, T) -> (B * T)
        mask_flattened = mask.reshape(-1)

        # 4. Filtrar solo los timesteps válidos de todos los sujetos (Eliminar Padding)
        X_valid = X_flattened[mask_flattened]
        C_valid = C_expanded[mask_flattened]

        # 5. Resolver OLS: X = C * Beta -> Beta = (C^T * C)^-1 * C^T * X
        # betas shape: (K, R)
        self.betas = torch.linalg.lstsq(C_valid, X_valid).solution[:-1, :]
        
        return self

    def transform(self, X: torch.Tensor, df: pd.DataFrame, mask: torch.Tensor):
        C = self._prepare_confounds(df, X.device)
        B, T, R = X.shape

        # Predecir el ruido: C (B, K) @ Betas (K, R) -> (B, R)
        # Esto nos da el offset por sujeto
        bias_per_subject = torch.matmul(C, self.betas) 
        
        # Expandir el ruido a todos los timesteps: (B, R) -> (B, 1, R) -> (B, T, R)
        predicted_noise = bias_per_subject.unsqueeze(1).expand(-1, T, -1)
        
        # Aplicar corrección y limpiar el padding con la máscara para seguridad
        X_corrected = X - predicted_noise
        X_corrected[~mask] = 0.0 

        return X_corrected

class GlobalNormalizer:
    """Normaliza basándose en estadísticos globales ignorando el padding temporal"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: torch.Tensor, mask: torch.Tensor):
        # X: (B, T, R), mask: (B, T)
        mask_ext = mask.unsqueeze(-1) # (B, T, 1) para broadcasting
        
        # Sumar solo valores válidos
        sum_x = (X * mask_ext).sum(dim=(0, 1))
        count = mask_ext.sum(dim=(0, 1))
        
        # Media global por ROI
        self.mean_ = (sum_x / count).reshape(1, 1, -1)
        
        # Desviación estándar corregida (ignorando ceros)
        diff_sq = (((X - self.mean_) ** 2) * mask_ext).sum(dim=(0, 1))
        self.std_ = torch.sqrt(diff_sq / count).reshape(1, 1, -1)
        
        # Evitar división por cero
        self.std_ = torch.where(self.std_ == 0, torch.ones_like(self.std_), self.std_)
        return self

    def transform(self, X: torch.Tensor, mask: torch.Tensor = None):
        X_norm = (X - self.mean_) / self.std_
        if mask is not None:
            X_norm[~mask] = 0.0
        return X_norm