"""
TST2 Pre-training Script
Pre-trains Connection Transformer using element-level masking strategy
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_fc import TransformerFC, create_transformer_fc, MaskedMSELoss
from pretrain.mask_utils import mask_pcc_level


class PretrainFCDataset(torch.utils.data.Dataset):
    """TST2 Pre-training Dataset"""
    
    def __init__(self, pcc_vectors, normalize=True):
        """
        Args:
            pcc_vectors: ndarray, shape (n_samples, pcc_dim)
            normalize: Whether to perform normalization
        """
        self.pcc_vectors = pcc_vectors.astype(np.float32)
        self.normalize = normalize
        
        if normalize:
            self.mean = np.mean(self.pcc_vectors)
            self.std = np.std(self.pcc_vectors) + 1e-8
    
    def __len__(self):
        return len(self.pcc_vectors)
    
    def __getitem__(self, idx):
        pcc = self.pcc_vectors[idx].copy()
        
        if self.normalize:
            pcc = (pcc - self.mean) / self.std
        
        return torch.tensor(pcc, dtype=torch.float32)


def train_epoch(model, train_loader, optimizer, criterion, device, mask_ratio=0.15):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)
        
        # Apply PCC element-level mask
        masked_batch, mask, target = mask_pcc_level(batch, mask_ratio)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(masked_batch, mode='pretrain')
        
        # Calculate MSE loss at masked positions
        loss = criterion(output, target, mask)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device, mask_ratio=0.15):
    """Validation"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            # Apply PCC element-level mask
            masked_batch, mask, target = mask_pcc_level(batch, mask_ratio)
            
            # Forward pass
            output = model(masked_batch, mode='pretrain')
            
            # Calculate MSE loss at masked positions
            loss = criterion(output, target, mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def pretrain_transformer_fc(
    train_loader,
    val_loader,
    model_config=None,
    epochs=100,
    lr=1e-4,
    weight_decay=1e-4,
    mask_ratio=0.15,
    device='cuda',
    save_dir='checkpoints',
    log_dir=None
):
    """
    Pre-train TST2 model
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model_config: Model configuration
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        mask_ratio: Masking ratio
        device: Device
        save_dir: Save directory
        log_dir: TensorBoard log directory
    
    Returns:
        model: Trained model
    """
    # Create model
    model = create_transformer_fc(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = MaskedMSELoss()
    
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
        log_dir = os.path.join(save_dir, '../logs/tst2')
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
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, mask_ratio
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, mask_ratio)
        
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
            torch.save(checkpoint, os.path.join(save_dir, 'tst2_best.pt'))
            print(f"Saved best model at epoch {epoch} (val_loss: {val_loss:.4f})")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    torch.save(history, os.path.join(save_dir, 'tst2_history.pt'))
    
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
    
    pcc_vectors = data['pcc_vectors']
    labels = data['labels']
    
    print(f"PCC vectors shape: {pcc_vectors.shape}")
    
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
    
    train_pcc = pcc_vectors[train_idx]
    val_pcc = pcc_vectors[val_idx]
    test_pcc = pcc_vectors[test_idx]
    
    print(f"Data split - Train: {len(train_pcc)}, Val: {len(val_pcc)}, Test: {len(test_pcc)}")
    
    # Create dataset and dataloader
    train_dataset = PretrainFCDataset(train_pcc)
    val_dataset = PretrainFCDataset(val_pcc)
    
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
        'pcc_dim': pcc_vectors.shape[1],
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout
    }
    
    # Pre-training
    model = pretrain_transformer_fc(
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
    parser = argparse.ArgumentParser(description='TST2 Pretraining')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, 
                        default='/root/workplace/exp/TwoTST/data/processed/processed_data.pkl')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, 
                        default='/root/workplace/exp/TwoTST/checkpoints/tst2')
    parser.add_argument('--log_dir', type=str, 
                        default='/root/workplace/exp/TwoTST/logs/tst2',
                        help='TensorBoard log directory')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)