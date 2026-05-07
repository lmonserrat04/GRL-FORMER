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
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from training.tasks.reconstruction import ReconstructionTask

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def train_one_epoch(model,task: ReconstructionTask , optimizer,scheduler,train_loader, val_loader,device, mask_ratio=None):
    """Train one epoch"""
    
    total_loss = 0.0

   
    
    for batch,_ in train_loader:
        

        batch = batch.to(device)
        
        # Apply ROI-level mask
        masked_batch, mask, target, _ = mask_roi_level(batch, mask_ratio)
        
        # Forward pass
        optimizer.zero_grad()
        loss = task.execution_step(model, masked_batch, mask,target)
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
        for batch,_ in val_loader:
            batch = batch.to(device)
            
            # Apply ROI-level mask
            masked_batch, mask, target, _ = mask_roi_level(batch, mask_ratio)
            
            #Forward pass
            loss = task.execution_step(model, masked_batch, mask,target)

            total_loss += loss.item()
            
    
    return total_loss
