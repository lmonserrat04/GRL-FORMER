import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

from data.loaders.pcc_utils import compute_pcc_vector


# ------------------------------------------------------------------------------
# Funciones de formateo (getters) – cada una sabe cómo preparar (datos, etiqueta)
# ------------------------------------------------------------------------------
def _getter_pretrain_ts(ts, pcc, label):
    return ts.T, label                     # (T, N_ROIS), label

def _getter_pretrain_fc(ts, pcc, label):
    return pcc, label                      # (PCC_dim,), label

def _getter_contrastive(ts, pcc, label):
    return {"timeseries": ts.T, "pcc_vector": pcc}, label

def _getter_finetune(ts, pcc, label):
    return {"timeseries": ts.T, "pcc_vector": pcc}, label   # etiqueta NO va en el dict

ITEM_GETTERS = {
    "pretrain_ts":  _getter_pretrain_ts,
    "pretrain_fc":  _getter_pretrain_fc,
    "contrastive":  _getter_contrastive,
    "finetune":     _getter_finetune,
}


class FullABIDEDataset:
    """
    Carga todos los datos desde archivos .1D y calcula los vectores PCC.
    Aplica normalización/armonización si se proporcionan las instancias correspondientes.
    """
    def __init__(
        self,
        config: dict,
        df_full: pd.DataFrame,
        normalizer: Optional[object] = None,
        harmonizer: Optional[object] = None,
    ):
        self.cfg = config
        self.df = df_full

        raw_path = Path(config["RAW_PATH"]).resolve()
        if not raw_path.exists():
            raise FileNotFoundError(f"RAW_PATH no encontrado: {raw_path}")

        self.timeseries_list: List[torch.Tensor] = []
        self.pcc_list: List[torch.Tensor] = []
        self.labels: torch.Tensor = torch.tensor(df_full[config["LABEL_COL"]].values, dtype=torch.long)

        # IDs de sujeto y sitio (asumimos columnas 'SUB_ID' y 'SITE_ID' en el CSV)
        self.subject_ids = df_full['SUB_ID'].values if 'SUB_ID' in df_full.columns else np.arange(len(df_full))
        self.site_ids = df_full['SITE_ID'].values if 'SITE_ID' in df_full.columns else None

        # Cargar series temporales
        print(f"Cargando {len(df_full)} series desde {raw_path}...")
        for _, row in tqdm(df_full.iterrows(), total=len(df_full)):
            filename = f"{row['FILE_ID']}_rois_{config['ATLAS']}.1D"
            filepath = raw_path / filename
            ts = np.loadtxt(filepath).T  # (N_ROIS, T)  (suponiendo que el archivo .1D tiene forma (T, N_ROIS))
            self.timeseries_list.append(torch.tensor(ts, dtype=torch.float32))

        # Aplicar normalización y armonización (si se proporcionan)
        if harmonizer is not None or normalizer is not None:
            self._apply_cleaning(normalizer, harmonizer)

        # Recortar series a MAX_SEQ_LEN
        max_len = config.get("MAX_SEQ_LEN")
        if max_len is not None:
            self.timeseries_list = [
                ts[:, :max_len] if ts.shape[1] > max_len else ts
                for ts in self.timeseries_list
            ]
            print(f"Series recortadas a {max_len} timepoints")

        # Calcular vectores PCC
        print("Calculando vectores PCC...")
        self.pcc_list = [compute_pcc_vector(ts) for ts in tqdm(self.timeseries_list)]

        # Verificar dimensión PCC
        expected_dim = config.get("TST2", {}).get("PCC_DIM")
        if expected_dim is not None and self.pcc_list[0].shape[0] != expected_dim:
            raise ValueError(f"Dimensión PCC calculada {self.pcc_list[0].shape[0]} no coincide con config TST2.PCC_DIM={expected_dim}")

    def _apply_cleaning(self, normalizer, harmonizer):
        """
        Aplica normalización/armonización usando la misma lógica de padding que tenías.
        """
        # Encontrar longitud máxima
        max_t = max(ts.shape[1] for ts in self.timeseries_list)
        n_rois = self.cfg["N_ROIS"]

        # Crear tensor acolchado (B, R, T_max)
        X_padded = torch.zeros((len(self.timeseries_list), n_rois, max_t))
        mask = torch.zeros((len(self.timeseries_list), max_t), dtype=torch.bool)
        for i, ts in enumerate(self.timeseries_list):
            t_len = ts.shape[1]
            X_padded[i, :, :t_len] = ts
            mask[i, :t_len] = True

        # Permutar a (B, T, R) para normalizadores
        X_padded = X_padded.permute(0, 2, 1)  # (B, T, R)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_padded = X_padded.to(device)
        mask = mask.to(device)

        # Necesitamos el DataFrame para pasar site_ids al harmonizer (si lo requiere)
        df_subset = self.df  # DataFrame completo

        # Siempre se hace fit en todos los datos? En tu código original, solo en train.
        # Pero aquí estamos en la carga completa; el fit debería hacerse solo con train.
        # Por simplicidad, asumimos que los objetos normalizer/harmonizer ya fueron fitteados
        # externamente y aquí solo aplicamos transform.
        if harmonizer is not None:
            X_padded = harmonizer.transform(X_padded, df_subset, mask)
        if normalizer is not None:
            X_padded = normalizer.transform(X_padded, mask)

        # Volver a (B, R, T) y quitar padding
        X_padded = X_padded.cpu().permute(0, 2, 1)
        for i, ts in enumerate(self.timeseries_list):
            t_len = ts.shape[1]
            self.timeseries_list[i] = X_padded[i, :, :t_len]


class SubsetDataset(Dataset):
    def __init__(self, full_dataset: FullABIDEDataset, indices: np.ndarray, item_getter):
        self.full = full_dataset
        self.indices = indices
        self.item_getter = item_getter

        # Etiquetas para WeightedRandomSampler
        self.labels = self.full.labels[indices]

        # Stratify por sitio solo si está activado en la configuración
        stratify_by_site = self.full.cfg.get("STRATIFY_BY_SITE", False)
        if stratify_by_site and hasattr(self.full, 'SITE_ID') and self.full.site_ids is not None:
            strat_labels = [f"{self.full.labels[idx]}_{self.full.site_ids[idx]}" for idx in indices]
            # Convierte etiquetas compuestas en enteros. Ej: ["0_NYU","1_UCLA","0_NYU"] → [0,1,0]
            self.labels_strat = torch.tensor(
                pd.factorize(pd.Series(strat_labels))[0], dtype=torch.long
            )
        else:
            # Si no se estratifica por sitio, labels_strat es igual a labels
            self.labels_strat = self.labels.clone()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ts = self.full.timeseries_list[real_idx]
        pcc = self.full.pcc_list[real_idx]
        label = self.full.labels[real_idx]
        return self.item_getter(ts, pcc, label)