"""
Masking Strategy Module
Implements ROI-level masking and PCC element-level masking
"""

import numpy as np
import torch
import random


def mask_roi_level(timeseries, mask_ratio = None):
    """
    ROI-level masking strategy
    Randomly masks a certain proportion of ROIs, setting the entire column of the time series to zero
    
    Args:
        timeseries: torch.Tensor, shape (batch, T, n_rois) or (T, n_rois)
        mask_ratio: Masking ratio

    Returns:
        masked_ts: Masked time series
        mask: Mask positions, True indicates masked
        target: Original values at the masked positions
    """

    if mask_ratio is None:
        # Randomly choose masking ratio
        mask_ratio = 0.25 if random.random() < 0.5 else 0.5
    
    
    is_batch = timeseries.dim() == 3
    
    if not is_batch:
        timeseries = timeseries.unsqueeze(0)
    
    batch_size, T, n_rois = timeseries.shape
    device = timeseries.device
    
    # Calculate the number of ROIs to mask
    num_mask = int(n_rois * mask_ratio)
    
    # Generate different masks for each sample
    masked_ts = timeseries.clone()
    mask = torch.zeros(batch_size, n_rois, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        # Randomly select ROI indices to mask
        mask_indices = torch.randperm(n_rois, device=device)[:num_mask]
        mask[i, mask_indices] = True
        # Set the entire column of the selected ROIs to zero
        masked_ts[i, :, mask_indices] = 0
    
    # Expand mask to time dimension (batch, T, n_rois)
    mask_expanded = mask.unsqueeze(1).expand(-1, T, -1)
    
    # Get the original values of the masked positions
    target = timeseries.clone()
    
    if not is_batch:
        masked_ts = masked_ts.squeeze(0)
        mask_expanded = mask_expanded.squeeze(0)
        target = target.squeeze(0)
        mask = mask.squeeze(0)
    
    return masked_ts, mask_expanded, target, mask


def mask_pcc_level(pcc_vector, mask_ratio=0.15):
    """
    PCC element-level masking strategy
    Randomly masks a certain proportion of PCC values
    
    Args:
        pcc_vector: torch.Tensor, shape (batch, pcc_dim) or (pcc_dim,)
        mask_ratio: Masking ratio, default 0.15
    
    Returns:
        masked_pcc: Masked PCC vector
        mask: Mask positions, True indicates masked
        target: Original values at the masked positions
    """
    is_batch = pcc_vector.dim() == 2
    
    if not is_batch:
        pcc_vector = pcc_vector.unsqueeze(0)
    
    batch_size, pcc_dim = pcc_vector.shape
    device = pcc_vector.device
    
    # Calculate the number of elements to mask
    num_mask = int(pcc_dim * mask_ratio)
    
    # Generate different masks for each sample
    masked_pcc = pcc_vector.clone()
    mask = torch.zeros(batch_size, pcc_dim, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        # Randomly select element indices to mask
        mask_indices = torch.randperm(pcc_dim, device=device)[:num_mask]
        mask[i, mask_indices] = True
        # Set the selected elements to zero
        masked_pcc[i, mask_indices] = 0
    
    # Get the original values of the masked positions
    target = pcc_vector.clone()
    
    if not is_batch:
        masked_pcc = masked_pcc.squeeze(0)
        mask = mask.squeeze(0)
        target = target.squeeze(0)
    
    return masked_pcc, mask, target


class ROIMaskTransform:
    """
    ROI-level mask transformation class
    Used for online masking during data loading
    """
    
    def __init__(self, mask_ratio=0.5):
        """
        Args:
            mask_ratio: Masking ratio
        """
        self.mask_ratio = mask_ratio
    
    def __call__(self, timeseries):
        """
        Args:
            timeseries: torch.Tensor, shape (T, n_rois)
        
        Returns:
            masked_ts: Masked time series
            target: Original time series
            mask: Mask positions (T, n_rois)
            roi_mask: ROI-level mask (n_rois,)
        """
        masked_ts, mask, target, roi_mask = mask_roi_level(
            timeseries, self.mask_ratio
        )
        return masked_ts, target, mask, roi_mask


class PCCMaskTransform:
    """
    PCC element-level mask transformation class
    Used for online masking during data loading
    """
    
    def __init__(self, mask_ratio=0.15):
        """
        Args:
            mask_ratio: Masking ratio, default 0.15
        """
        self.mask_ratio = mask_ratio
    
    def __call__(self, pcc_vector):
        """
        Args:
            pcc_vector: torch.Tensor, shape (pcc_dim,)
        
        Returns:
            masked_pcc: Masked PCC vector
            target: Original PCC vector
            mask: Mask positions
        """
        masked_pcc, mask, target = mask_pcc_level(pcc_vector, self.mask_ratio)
        return masked_pcc, target, mask


def create_attention_mask_from_roi_mask(roi_mask, seq_len):
    """
    Creates attention mask from ROI mask
    Time points corresponding to masked ROIs should not be seen by other positions
    
    Args:
        roi_mask: torch.Tensor, shape (batch, n_rois) or (n_rois,)
        seq_len: Sequence length (number of time points)
    
    Returns:
        attn_mask: Attention mask, shape (batch, seq_len, seq_len) or (seq_len, seq_len)
    """
    is_batch = roi_mask.dim() == 2
    
    if not is_batch:
        roi_mask = roi_mask.unsqueeze(0)
    
    batch_size, n_rois = roi_mask.shape
    device = roi_mask.device
    
    # For Transformer, we usually don't need a special attention mask
    # because the purpose of masking is to let the model learn reconstruction, not to block information flow
    # Here it returns an all-zero mask (indicating all positions are visible)
    attn_mask = torch.zeros(batch_size, seq_len, seq_len, device=device)
    
    if not is_batch:
        attn_mask = attn_mask.squeeze(0)
    
    return attn_mask


def batch_mask_roi_level(batch_timeseries, mask_ratio=None):
    """
    Batch ROI-level masking
    
    Args:
        batch_timeseries: torch.Tensor, shape (batch, T, n_rois)
        mask_ratio: Masking ratio
    
    Returns:
        masked_ts: Masked time series
        mask: Mask positions (batch, T, n_rois)
        target: Original time series
        roi_mask: ROI-level mask (batch, n_rois)
    """
    return mask_roi_level(batch_timeseries, mask_ratio)


def batch_mask_pcc_level(batch_pcc_vectors, mask_ratio=0.15):
    """
    Batch PCC element-level masking
    
    Args:
        batch_pcc_vectors: torch.Tensor, shape (batch, pcc_dim)
        mask_ratio: Masking ratio
    
    Returns:
        masked_pcc: Masked PCC vector
        mask: Mask positions
        target: Original PCC vector
    """
    return mask_pcc_level(batch_pcc_vectors, mask_ratio)


# Helper function for testing
def test_mask_functions():
    """Test masking functions"""
    print("Testing ROI-level mask...")
    ts = torch.randn(4, 100, 200)  # batch=4, T=100, n_rois=200
    masked_ts, mask, target, roi_mask = mask_roi_level(ts, mask_ratio=0.25)
    print(f"  Input shape: {ts.shape}")
    print(f"  Masked shape: {masked_ts.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  ROI mask shape: {roi_mask.shape}")
    print(f"  Masked ROIs per sample: {roi_mask.sum(dim=1)}")
    
    print("\nTesting PCC-level mask...")
    pcc = torch.randn(4, 19900)  # batch=4, pcc_dim=19900
    masked_pcc, mask, target = mask_pcc_level(pcc, mask_ratio=0.15)
    print(f"  Input shape: {pcc.shape}")
    print(f"  Masked shape: {masked_pcc.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Masked elements per sample: {mask.sum(dim=1)}")
    
    print("\nAll tests passed!")


if __name__ == '__main__':
    test_mask_functions()