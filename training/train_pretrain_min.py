"""
Script de pre-entrenamiento TwoTST
Configuración óptima según el paper:
  - TST1: mask ratio aleatorio entre 0.25 y 0.5
  - TST2: mask ratio fijo de 0.15
  - Contrastive learning con ambos encoders descongelados (unfreeze both)
  - Fusión: Attention Pooling
  - Fine-tuning con encoders descongelados + projection heads entrenables
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pretrain.pretrain_ts import pretrain_transformer_ts, PretrainTSDataset
from pretrain.pretrain_fc import pretrain_transformer_fc, PretrainFCDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Semilla y dispositivo
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = load_config(args.config) if args.config else {}

    # Carga de datos
    print("Loading data...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    timeseries     = data['timeseries']
    pcc_vectors    = data['pcc_vectors']
    labels         = data['labels']
    subject_indices = data.get('subject_indices')
    site_ids        = data.get('site_ids')

    print(f"Timeseries shape : {timeseries.shape}")
    print(f"PCC vectors shape: {pcc_vectors.shape}")

    # Split a nivel de sujeto para evitar data leakage
    if subject_indices is not None:
        from utils.splitters import get_subject_level_train_val_test_split
        train_idx, val_idx, test_idx = get_subject_level_train_val_test_split(
            labels, subject_indices, site_ids=site_ids,
            train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=args.seed
        )
    else:
        indices = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, random_state=args.seed, stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=2/3, random_state=args.seed,
            stratify=labels[temp_idx]
        )

    print(f"Split — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # ------------------------------------------------------------------ #
    # Fase 1 — Pre-entrenamiento TST1 (ROI Time-Series Transformer)
    # Mask ratio: aleatorio entre 0.25 y 0.50 (configuración óptima del paper)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Fase 1: Pre-entrenamiento TST1 (Time-Series Transformer)")
    print("=" * 60)

    train_ts_loader = DataLoader(
        PretrainTSDataset(timeseries[train_idx]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    val_ts_loader = DataLoader(
        PretrainTSDataset(timeseries[val_idx]),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    tst1_config = config.get('tst1', {})
    tst1_config.update({
        'n_rois'      : timeseries.shape[2],   # 200 (CC200)
        'max_seq_len' : timeseries.shape[1],   # 100 time points
        # Arquitectura óptima (Table 2 del paper)
        'd_model'     : tst1_config.get('d_model', 512),
        'n_layers'    : tst1_config.get('n_layers', 6),
        'n_heads'     : tst1_config.get('n_heads', 8),
        'd_ff'        : tst1_config.get('d_ff', 2048),
        'dropout'     : tst1_config.get('dropout', 0.1),
    })

    pretrain_transformer_ts(
        train_loader = train_ts_loader,
        val_loader   = val_ts_loader,
        model_config = tst1_config,
        epochs       = args.tst1_epochs,          # 100
        lr           = args.lr,                   # 1e-4
        weight_decay = args.weight_decay,         # 1e-4
        mask_ratio   = None,                      # None → aleatorio 0.25–0.50
        device       = device,
        save_dir     = os.path.join(args.save_dir, 'tst1'),
        log_dir      = os.path.join(args.log_dir,  'tst1'),
    )

    # ------------------------------------------------------------------ #
    # Fase 2 — Pre-entrenamiento TST2 (PCC / Functional-Connectivity)
    # Mask ratio fijo: 0.15 (configuración óptima del paper)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Fase 2: Pre-entrenamiento TST2 (PCC Transformer)")
    print("=" * 60)

    train_fc_loader = DataLoader(
        PretrainFCDataset(pcc_vectors[train_idx]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_fc_loader = DataLoader(
        PretrainFCDataset(pcc_vectors[val_idx]),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    tst2_config = config.get('tst2', {})
    tst2_config.update({
        'pcc_dim' : pcc_vectors.shape[1],   # 19 900 (CC200, triángulo superior)
        # Arquitectura óptima (Table 2 del paper)
        'd_model' : tst2_config.get('d_model', 256),
        'n_layers': tst2_config.get('n_layers', 2),
        'n_heads' : tst2_config.get('n_heads', 8),
        'd_ff'    : tst2_config.get('d_ff', 512),
        'dropout' : tst2_config.get('dropout', 0.1),
    })

    pretrain_transformer_fc(
        train_loader = train_fc_loader,
        val_loader   = val_fc_loader,
        model_config = tst2_config,
        epochs       = args.tst2_epochs,          # 100
        lr           = args.lr,                   # 1e-4
        weight_decay = args.weight_decay,         # 1e-4
        mask_ratio   = 0.15,                      # fijo según paper
        device       = device,
        save_dir     = os.path.join(args.save_dir, 'tst2'),
        log_dir      = os.path.join(args.log_dir,  'tst2'),
    )

    print("\n" + "=" * 60)
    print("Pre-entrenamiento completado.")
    print("Siguiente paso: contrastive learning con ambos encoders")
    print("descongelados (unfreeze both) y fusión Attention Pooling.")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TwoTST — Pre-entrenamiento (configuración óptima)')

    # Datos
    parser.add_argument('--data_path', type=str,
                        default='/root/workplace/exp/TwoTST/data/processed/processed_data.pkl')
    parser.add_argument('--config', type=str, default=None,
                        help='Ruta opcional al archivo YAML de configuración')

    # Hiperparámetros de entrenamiento (Table 2 del paper)
    parser.add_argument('--tst1_epochs',  type=int,   default=100)
    parser.add_argument('--tst2_epochs',  type=int,   default=100)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Infraestructura
    parser.add_argument('--device',      type=str, default='cuda')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir',    type=str,
                        default='/root/workplace/exp/TwoTST/checkpoints')
    parser.add_argument('--log_dir',     type=str,
                        default='/root/workplace/exp/TwoTST/logs')

    args = parser.parse_args()
    main(args)