"""
TST1 Pre-training Script
Pre-trains Temporal Transformer using ROI-level masking strategy
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from training.tasks.reconstruction import ReconstructionTask

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_ts import TransformerTS, create_transformer_ts
from data.augmentation.mask_utils import mask_roi_level


class PretrainTSDataset(torch.utils.data.Dataset):
    """TST1 Pre-training Dataset"""
    
    def __init__(self, timeseries, normalize=True):
        """
        Args:
            timeseries: ndarray, shape (n_samples, n_timepoints, n_rois)
            normalize: Whether to perform normalization
        """
        self.timeseries = timeseries.astype(np.float32)
        self.normalize = normalize
        
        if normalize:
            self.mean = np.mean(self.timeseries)
            self.std = np.std(self.timeseries) + 1e-8
    
    def __len__(self):
        return len(self.timeseries)
    
    def __getitem__(self, idx):
        ts = self.timeseries[idx].copy()
        
        if self.normalize:
            ts = (ts - self.mean) / self.std
        
        return torch.tensor(ts, dtype=torch.float32)


def train_one_epoch(model,task: ReconstructionTask , optimizer,scheduler,train_loader, val_loader, device, mask_ratio=None):
    """Train one epoch"""
    
    total_loss = 0.0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        # Apply ROI-level mask
        masked_batch, mask, target, roi_mask = mask_roi_level(batch, mask_ratio)
        
        # Forward pass
        optimizer.zero_grad()
        loss, pred = task.execution_step(model, masked_batch, mask,target)
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()


        
    return total_loss


def validate(model,task: ReconstructionTask , optimizer,scheduler,train_loader, val_loader, device, mask_ratio=None):
    """Validation"""
    
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            # Apply ROI-level mask
            masked_batch, mask, target, roi_mask = mask_roi_level(batch, mask_ratio)
            
            #Forward pass
            loss, pred = task.execution_step(model, masked_batch, mask,target)

            total_loss += loss.item()
            
    
    return total_loss


def pretrain_transformer_ts(
    train_loader,
    val_loader,
    model_config=None,
    epochs=100,
    lr=1e-4,
    weight_decay=1e-4,
    mask_ratio=None,
    device='cuda',
    save_dir='checkpoints',
    log_dir=None
):
    """
    Pre-train TST1 model
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model_config: Model configuration
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        mask_ratio: Masking ratio (None for random 0.25 or 0.5)
        device: Device
        save_dir: Save directory
        log_dir: TensorBoard log directory
    
    Returns:
        model: Trained model
    """
    # Create model
    model = create_transformer_ts(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create TensorBoard log directory
    if log_dir is None:
        log_dir = os.path.join(save_dir, '../logs/tst1')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, mask_ratio
        )
        
        # Validate
        val_loss = validate(model, val_loader, device, mask_ratio)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # Save best model only
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'best_epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': model_config,
                'train_losses': train_losses.copy(),
                'val_losses': val_losses.copy()
            }
            torch.save(checkpoint, os.path.join(save_dir, 'tst1_best.pt'))
            print(f"Saved best model at epoch {epoch} (val_loss: {val_loss:.4f})")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    torch.save(history, os.path.join(save_dir, 'tst1_history.pt'))
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\nTraining completed. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"Run 'tensorboard --logdir={log_dir}' to view training curves")
    return model


def main(args):
    import pickle
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    timeseries = data['timeseries']
    labels = data['labels']
    
    print(f"Timeseries shape: {timeseries.shape}")
    
    # 7:1:2 data split
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(labels))
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=args.seed, stratify=labels
    )
    
    # Second split: val vs test
    val_test_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=2/3, random_state=args.seed, stratify=val_test_labels
    )
    
    train_ts = timeseries[train_idx]
    val_ts = timeseries[val_idx]
    test_ts = timeseries[test_idx]
    
    print(f"Data split - Train: {len(train_ts)}, Val: {len(val_ts)}, Test: {len(test_ts)}")
    
    # Create dataset and dataloader
    train_dataset = PretrainTSDataset(train_ts)
    val_dataset = PretrainTSDataset(val_ts)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model configuration
    model_config = {
        'n_rois': timeseries.shape[2],
        'emb_dim': args.emb_dim,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'max_seq_len': timeseries.shape[1],
        'use_cls_token': True
    }
    
    # Pre-training
    model = pretrain_transformer_ts(
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TST1 Pretraining')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, 
                        default='/root/workplace/exp/TwoTST/data/processed/processed_data.pkl')
    
    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--mask_ratio', type=float, default=None,
                        help='Mask ratio (None for random 0.25/0.5)')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, 
                        default='/root/workplace/exp/TwoTST/checkpoints/tst1')
    parser.add_argument('--log_dir', type=str, 
                        default='/root/workplace/exp/TwoTST/logs/tst1',
                        help='TensorBoard log directory')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)