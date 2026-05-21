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

from training.tasks.classification import ClassificationTask

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def train_one_epoch(model,task: ClassificationTask , optimizer,scheduler,train_loader, val_loader,device, mask_ratio = None):
    """Train one epoch"""
   
    total_loss = 0.0
    
    for batch in train_loader:
        ts  = batch['timeseries'].to(device)
        pcc = batch['pcc_vector'].to(device)
        y   = batch['label'].to(device)


        optimizer.zero_grad()
        
       
        loss = task.execution_step(model, ts_batch= ts, pcc_batch= pcc , targets= y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
       
    return total_loss


def validate(model,task: ClassificationTask , optimizer,scheduler,train_loader, val_loader,device, mask_ratio = None):
    """Validation"""
    
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            ts  = batch['timeseries'].to(device)
            pcc = batch['pcc_vector'].to(device)
            y   = batch['label'].to(device)

            # Forward pass
            loss = task.execution_step(model, ts_batch= ts, pcc_batch= pcc , targets= y)
        
            total_loss += loss.item()

    
    return total_loss
