"""
Script de fine-tuning TwoTST
Configuración óptima según el paper (Brain Sci. 2026, 16, 277):

  Pre-training  : TST1 (mask 0.25-0.50) + TST2 (mask 0.15)   [Table 2]
  Contrastive   : InfoNCE, unfreeze both encoders, τ=0.07      [Sec. 4.3.2]
  Fusion        : Attention Pooling                             [Table 3]
  Fine-tuning   : encoders descongelados + projection heads    [Sec. 3.4.6]
                  lr=5e-5, dropout=0.3, early stopping p=20    [Table 2]
  Evaluación    : LOSO (19 sitios), majority voting            [Sec. 4.3.4]
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.dual_stream import DualStreamModel, create_dual_stream_model
from pretrain.contrastive import ContrastiveWrapper
from utils.data_loader import get_finetune_loaders
from utils.metrics import (
    aggregate_window_predictions_to_subject_level,
    bootstrap_confidence_interval,
    get_reproducibility_info,
)


# ------------------------------------------------------------------ #
#  Métricas
# ------------------------------------------------------------------ #

def get_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy'   : accuracy_score(y_true, y_pred),
        'f1'         : f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


# ------------------------------------------------------------------ #
#  Bucle de entrenamiento por epoch
# ------------------------------------------------------------------ #

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        ts  = batch['timeseries'].to(device)
        pcc = batch['pcc_vector'].to(device)
        y   = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(ts, pcc)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    with torch.no_grad():
        for batch in loader:
            ts  = batch['timeseries'].to(device)
            pcc = batch['pcc_vector'].to(device)
            y   = batch['label'].to(device)

            logits = model(ts, pcc)
            total_loss += criterion(logits, y).item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss / len(loader), get_metrics(all_labels, all_preds, all_probs)


# ------------------------------------------------------------------ #
#  Fase contrastiva  (unfreeze both encoders — configuración óptima)
# ------------------------------------------------------------------ #

def run_contrastive_phase(model, loader, device):
    """
    Alinea las representaciones de TST1 y TST2 mediante InfoNCE.
    Ambos encoders permanecen descongelados (unfreeze both > freeze TST1
    > freeze TST2 > freeze both, según Sec. 4.3.2).
    Hiperparámetros fijos según Table 2:
      epochs=50, lr=1e-4, wd=1e-4, τ=0.07,
      proj hidden=256, proj output=128
    """
    print("\n--- Fase Contrastiva (unfreeze both, InfoNCE, τ=0.07) ---")

    contrastive_module = ContrastiveWrapper(
        dim_ts      = model.dim_ts,
        dim_fc      = model.dim_fc,
        temperature = 0.07,             # τ óptimo, Table 2
        hidden_dim  = 256,              # proj hidden dim, Table 2
        output_dim  = 128,              # proj output dim, Table 2
    ).to(device)

    # Ambos encoders descongelados: optimizar todo
    params = list(model.parameters()) + list(contrastive_module.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4)

    for epoch in range(1, 51):          # 50 epochs, Table 2
        model.train()
        contrastive_module.train()
        epoch_loss, align_sum = 0.0, 0.0

        for batch in loader:
            ts  = batch['timeseries'].to(device)
            pcc = batch['pcc_vector'].to(device)

            optimizer.zero_grad()
            h_ts, h_fc = model.get_features(ts, pcc)
            loss, _, align = contrastive_module(h_ts, h_fc)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            align_sum  += align.item() if hasattr(align, 'item') else align

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Contrastive Epoch {epoch:3d}/50 | "
                  f"Loss={epoch_loss/len(loader):.4f} | "
                  f"Align(cos)={align_sum/len(loader):.4f}")
        


# ------------------------------------------------------------------ #
#  Fine-tuning de un fold / sitio LOSO
# ------------------------------------------------------------------ #

def finetune_fold(
    train_loader,
    val_loader,
    test_loader,
    model_config,
    tst1_checkpoint,
    tst2_checkpoint,
    contrastive_module=None,
    epochs=100,
    lr=5e-5,
    weight_decay=1e-4,
    early_stopping_patience=20,
    device='cuda',
    save_dir=None,
    fold_idx=0,
    split_info=None,
):
    """
    Fine-tuning con la configuración óptima del paper:
    - Encoders descongelados
    - Projection heads del contrastive learning incluidos
    - Attention Pooling como fusión
    - Early stopping patience=20 (Table 2)
    - Criterion: CrossEntropyLoss (Ec. 16)
    """
    # Crear modelo con Attention Pooling (fusión óptima, Table 3)
    model = create_dual_stream_model(**model_config).to(device)

    # Cargar pesos pre-entrenados
    if tst1_checkpoint and os.path.exists(tst1_checkpoint):
        model.load_pretrained_tst1(tst1_checkpoint, strict=False)
        print(f"  TST1 pesos cargados desde: {tst1_checkpoint}")

    if tst2_checkpoint and os.path.exists(tst2_checkpoint):
        model.load_pretrained_tst2(tst2_checkpoint, strict=False)
        print(f"  TST2 pesos cargados desde: {tst2_checkpoint}")

    # Fase contrastiva (siempre activa — mejora AUC en todas las fusiones
    # salvo Bilinear, Table 6)
    run_contrastive_phase(model, train_loader, device)

    # Transferir projection heads al modelo para fine-tuning conjunto
    # if hasattr(model, 'set_projection_heads'):
    #     model.set_projection_heads(contrastive_module)

    print("\n--- Fine-tuning (encoders descongelados, Attention Pooling) ---")

    # Encoders descongelados + projection heads entrenables (Sec. 3.4.6)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_val_auc   = 0.0
    best_state     = None
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )

        val_auc = val_metrics.get('auc', 0.0)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f} AUC={val_auc:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= early_stopping_patience:
                print(f"  Early stopping en epoch {epoch} "
                      f"(patience={early_stopping_patience})")
                break

    # Restaurar mejor checkpoint
    model.load_state_dict(best_state)

    # ---- Evaluación en test ---- #
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ts  = batch['timeseries'].to(device)
            pcc = batch['pcc_vector'].to(device)
            y   = batch['label'].to(device)
            logits = model(ts, pcc)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Agregación a nivel de sujeto (majority voting, Sec. 4.3.4)
    if split_info and split_info.get('subject_indices') is not None \
            and split_info.get('test_idx') is not None:
        test_idx     = split_info['test_idx']
        subj_indices = split_info['subject_indices']
        n_windows    = len(test_idx)
        n_subjects   = len(np.unique(subj_indices[test_idx]))
        if n_windows > n_subjects:
            all_labels, all_preds, all_probs = \
                aggregate_window_predictions_to_subject_level(
                    all_labels, all_preds, all_probs,
                    test_idx, subj_indices,
                    strategy='majority_vote',   # majority voting, Sec. 4.3.4
                )
            print(f"\n  Fold {fold_idx} → evaluación a nivel de sujeto (majority voting)")
        else:
            print(f"\n  Fold {fold_idx} → evaluación a nivel de ventana")
    else:
        print(f"\n  Fold {fold_idx} → evaluación estándar")

    test_metrics = get_metrics(all_labels, all_preds, all_probs)
    print(f"  AUC={test_metrics.get('auc',0):.4f}  "
          f"ACC={test_metrics['accuracy']:.4f}  "
          f"Sens={test_metrics.get('sensitivity',0):.4f}  "
          f"Spec={test_metrics.get('specificity',0):.4f}  "
          f"F1={test_metrics['f1']:.4f}")

    # Guardar checkpoint
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state_dict' : model.state_dict(),
            'metrics'          : test_metrics,
            'config'           : model_config,
        }, os.path.join(save_dir, f'fold{fold_idx}_best.pt'))

    return test_metrics


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    timeseries  = data['timeseries']
    pcc_vectors = data['pcc_vectors']
    site_ids    = data.get('site_ids')

    # LOSO: número de folds = número de sitios (Sec. 4.3.4)
    n_sites = len(np.unique(site_ids)) if site_ids is not None else None
    n_folds = n_sites if args.eval_protocol == 'loso' else args.n_folds
    print(f"Protocolo: {args.eval_protocol.upper()} | Folds: {n_folds}")

    # Configuración fija del modelo (Table 2 + Sec. 3.4.6)
    model_config = {
        'n_rois'       : timeseries.shape[2],        # 200 (CC200)
        'time_points'  : timeseries.shape[1],        # 100
        'pcc_dim'      : pcc_vectors.shape[1],       # 19 900
        'tst1_emb_dim' : 512,    # d1, Table 2
        'tst2_d_model' : 256,    # d2, Table 2
        'fusion_type'  : 'attention_pooling',        # óptimo, Table 3
        'num_classes'  : 2,
        'dropout'      : 0.3,    # fine-tuning dropout, Table 2
    }

    all_metrics = []

    for fold_idx in range(n_folds):
        print(f"\n{'='*60}")
        print(f"  {'Sitio' if args.eval_protocol == 'loso' else 'Fold'} "
              f"{fold_idx + 1}/{n_folds}")
        print('='*60)

        train_loader, val_loader, test_loader, split_info = get_finetune_loaders(
            args.data_path,
            batch_size              = args.batch_size,
            num_workers             = args.num_workers,
            n_folds                 = n_folds,
            fold_idx                = fold_idx,
            val_ratio               = 0.15,
            seed                    = args.seed,
            use_subject_level_split = True,    # siempre: evita data leakage
            eval_protocol           = args.eval_protocol,
        )

        fold_metrics = finetune_fold(
            train_loader          = train_loader,
            val_loader            = val_loader,
            test_loader           = test_loader,
            model_config          = model_config,
            tst1_checkpoint       = args.tst1_checkpoint,
            tst2_checkpoint       = args.tst2_checkpoint,
            epochs                = args.epochs,
            lr                    = args.lr,
            weight_decay          = args.weight_decay,
            early_stopping_patience = 20,       # Table 2
            device                = device,
            save_dir              = args.save_dir,
            fold_idx              = fold_idx,
            split_info            = split_info,
        )
        all_metrics.append(fold_metrics)

    # ---- Resumen con IC Bootstrap 95% ---- #
    metric_names = ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1']
    print(f"\n{'='*60}")
    print(f"Resultados finales ({args.eval_protocol.upper()}, mean ± std [95% CI]):")
    print('='*60)

    mean_metrics, std_metrics, ci_metrics = {}, {}, {}
    for metric in metric_names:
        values = [m.get(metric, 0.0) for m in all_metrics]
        mean_v, std_v, lo, hi = bootstrap_confidence_interval(
            values, n_bootstrap=1000, ci=0.95, seed=args.seed
        )
        mean_metrics[metric] = mean_v
        std_metrics[metric]  = std_v
        ci_metrics[metric]   = (lo, hi)
        label = metric.upper() if metric == 'auc' else metric.capitalize()
        print(f"  {label:12s}: {mean_v:.4f} ± {std_v:.4f}  [{lo:.4f}, {hi:.4f}]")

    # Guardar resultados
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        repro = get_reproducibility_info()
        repro.update({'seed': args.seed, 'n_folds': n_folds,
                      'eval_protocol': args.eval_protocol})
        results = {
            'all_metrics'    : all_metrics,
            'mean_metrics'   : mean_metrics,
            'std_metrics'    : std_metrics,
            'ci_95_metrics'  : ci_metrics,
            'config'         : model_config,
            'args'           : vars(args),
            'reproducibility': repro,
        }
        out_path = os.path.join(args.save_dir, 'results.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(results, f, protocol=4)
        print(f"\nResultados guardados en: {out_path}")


# ------------------------------------------------------------------ #
#  CLI  (solo parámetros que cambian entre experimentos)
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TwoTST Fine-tuning — configuración óptima del paper'
    )

    # Rutas
    parser.add_argument('--data_path', type=str,
                        default='/root/workplace/exp/TwoTST/data/processed/processed_data.pkl')
    parser.add_argument('--tst1_checkpoint', type=str, default=None,
                        help='Checkpoint pre-entrenado de TST1')
    parser.add_argument('--tst2_checkpoint', type=str, default=None,
                        help='Checkpoint pre-entrenado de TST2')
    parser.add_argument('--save_dir', type=str,
                        default='/root/workplace/exp/TwoTST/checkpoints/finetune')

    # Protocolo de evaluación
    parser.add_argument('--eval_protocol', type=str, default='loso',
                        choices=['loso', 'kfold'],
                        help='loso: Leave-One-Site-Out (19 sitios, paper) | '
                             'kfold: K-fold CV')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Número de folds (solo para kfold)')

    # Hiperparámetros de fine-tuning (Table 2)
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Infraestructura
    parser.add_argument('--device',      type=str, default='cuda')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    main(args)