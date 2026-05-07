"""
TST2 Pre-training Script
Pre-trains Connection Transformer using element-level masking strategy
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from training.tasks.reconstruction import ReconstructionTask

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.augmentation.mask_utils import mask_pcc_level


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


def train_one_epoch(model,task: ReconstructionTask , optimizer,scheduler,train_loader, val_loader,device, mask_ratio=0.15):
    """Train one epoch"""
   
    total_loss = 0.0
    
    for batch,_ in train_loader:
        batch = batch.to(device)
        
        # Apply PCC element-level mask
        masked_batch, mask, target = mask_pcc_level(batch, mask_ratio)
        
        # Forward pass
        optimizer.zero_grad()
        loss = task.execution_step(model, masked_batch, mask,target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
       
    return total_loss


def validate(model,task: ReconstructionTask , optimizer,scheduler,train_loader, val_loader,device, mask_ratio=0.15):
    """Validation"""
    
    total_loss = 0.0

    with torch.no_grad():
        for batch,_ in val_loader:
            batch = batch.to(device)
            
            # Apply PCC element-level mask
            masked_batch, mask, target = mask_pcc_level(batch, mask_ratio)
            
            # Forward pass
            loss = task.execution_step(model, masked_batch, mask,target)
        
            total_loss += loss.item()

    
    return total_loss
