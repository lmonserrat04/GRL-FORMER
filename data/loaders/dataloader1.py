import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional

from data.loaders.base_dataset import FullABIDEDataset, SubsetDataset, ITEM_GETTERS


def get_batch_size(config: dict) -> int:
    phase_map = {
        "pretrain_ts": "PT_TST1",
        "pretrain_fc": "PT_TST2",
        "contrastive": "T_CONTRASTIVE",
        "finetune":    "FINETUNING",
    }
    phase_key = phase_map.get(config["EXPERIMENT_TYPE"])
    if not phase_key:
        raise ValueError(f"No se encontró fase para {config['EXPERIMENT_TYPE']}")
    return config[phase_key]["BATCH_SIZE"]


def build_dataloaders(
    config: dict,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    harmonizer=None,
    normalizer=None,
):
    exp_type = config["EXPERIMENT_TYPE"]
    bs = get_batch_size(config)

    # Dataset completo (una sola carga)
    df_full = pd.concat([df_train, df_val, df_test]).drop_duplicates().sort_index()
    full_ds = FullABIDEDataset(
        config=config,
        df_full=df_full,
        normalizer=normalizer,
        harmonizer=harmonizer,
    )

    train_indices = df_train.index.to_numpy()
    val_indices = df_val.index.to_numpy()
    test_indices = df_test.index.to_numpy()

    # ---- Aquí inyectamos el getter correspondiente a la fase ----
    getter = ITEM_GETTERS[exp_type]
    train_ds = SubsetDataset(full_ds, train_indices, getter)
    val_ds   = SubsetDataset(full_ds, val_indices, getter)
    test_ds  = SubsetDataset(full_ds, test_indices, getter)

    loader_kwargs = dict(
        batch_size=bs,
        num_workers=config.get("NUM_WORKERS", 0),
        pin_memory=torch.cuda.is_available(),
    )

    if config.get("USE_WEIGHTED_SAMPLER", False):
        strat_labels = getattr(train_ds, 'labels_strat', train_ds.labels)
        class_counts = np.bincount(strat_labels.numpy())
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[strat_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    val_loader  = DataLoader(val_ds,  shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader