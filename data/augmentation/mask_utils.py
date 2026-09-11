"""
Masking Strategy Module
Implements ROI-level masking and PCC element-level masking
"""

import numpy as np
import torch
import random


def mask_roi_level(timeseries, mask_ratio=None):
    """
    ROI-level masking strategy
    Randomly masks a certain proportion of ROIs, setting the entire column of the time series to zero

    Args:
        timeseries: torch.Tensor, shape (batch, T, n_rois) or (T, n_rois)
        mask_ratio: Masking ratio; if None, randomly chooses 0.25 or 0.5

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

    def __init__(self, mask_ratio=None):
        """
        Args:
            mask_ratio: Masking ratio; if None, randomly chooses 0.25 or 0.5
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


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    # ─── TEST 1: mask_roi_level (batch) ──────────────────────────────
    print("── TEST 1: mask_roi_level (batch, ratio fijo) ────────────────")
    B, T, R = 4, 100, 200
    ts = torch.randn(B, T, R)
    ratio = 0.25
    masked, mask, target, roi_mask = mask_roi_level(ts, mask_ratio=ratio)

    assert masked.shape == (B, T, R), masked.shape
    assert mask.shape == (B, T, R), mask.shape
    assert target.shape == (B, T, R), target.shape
    assert roi_mask.shape == (B, R), roi_mask.shape

    # Exactamente int(R * ratio) ROIs enmascaradas por muestra
    expected_per_sample = int(R * ratio)
    assert (roi_mask.sum(dim=1) == expected_per_sample).all(), \
        f"ROIs enmascaradas: {roi_mask.sum(dim=1).tolist()} (esperado {expected_per_sample})"

    # Las columnas enmascaradas deben ser exactamente cero en masked
    assert (masked[roi_mask.unsqueeze(1).expand(-1, T, -1)] == 0).all()

    # target debe conservar los valores originales (no ceros)
    assert torch.allclose(target, ts), "target debe ser la copia original"

    # Las columnas no enmascaradas no deben haberse tocado
    not_masked = ~roi_mask.unsqueeze(1).expand(-1, T, -1)
    assert torch.allclose(masked[not_masked], ts[not_masked])

    print(f"  ✓ shapes OK, {expected_per_sample} ROIs enmascaradas por muestra")
    print(f"  ✓ columnas enmascaradas en cero, resto intactas")
    print(f"  ✓ target == original\n")

    # ─── TEST 2: mask_roi_level (sin batch) ──────────────────────────
    print("── TEST 2: mask_roi_level (sin batch) ────────────────────────")
    ts_nb = torch.randn(T, R)
    masked_nb, mask_nb, target_nb, roi_mask_nb = mask_roi_level(ts_nb, mask_ratio=0.5)
    assert masked_nb.shape == (T, R)
    assert mask_nb.shape == (T, R)
    assert target_nb.shape == (T, R)
    assert roi_mask_nb.shape == (R,)
    assert roi_mask_nb.sum() == int(R * 0.5)
    print(f"  ✓ shapes sin batch: {masked_nb.shape} / {roi_mask_nb.shape}\n")

    # ─── TEST 3: mask_roi_level ratio=None (aleatorio 0.25 ó 0.5) ────
    print("── TEST 3: mask_roi_level ratio=None ─────────────────────────")
    valid_counts = {int(R * 0.25), int(R * 0.5)}
    for _ in range(10):
        _, _, _, rm = mask_roi_level(ts, mask_ratio=None)
        assert int(rm.sum(dim=1)[0]) in valid_counts, \
            f"ratio=None dio {rm.sum(dim=1)[0]} ROIs (esperado {valid_counts})"
    print(f"  ✓ ratio=None produce {valid_counts} ROIs\n")

    # ─── TEST 4: mask_pcc_level (batch) ──────────────────────────────
    print("── TEST 4: mask_pcc_level (batch) ────────────────────────────")
    D = 19900
    pcc = torch.randn(B, D)
    masked_pcc, mask_pcc, target_pcc = mask_pcc_level(pcc, mask_ratio=0.15)

    assert masked_pcc.shape == (B, D)
    assert mask_pcc.shape == (B, D)
    assert target_pcc.shape == (B, D)

    expected_elem = int(D * 0.15)
    assert (mask_pcc.sum(dim=1) == expected_elem).all(), \
        f"elementos enmascarados: {mask_pcc.sum(dim=1).tolist()}"
    assert (masked_pcc[mask_pcc] == 0).all()
    assert torch.allclose(target_pcc, pcc)
    print(f"  ✓ {expected_elem} elementos enmascarados por muestra")
    print(f"  ✓ target == original\n")

    # ─── TEST 5: mask_pcc_level (sin batch) ──────────────────────────
    print("── TEST 5: mask_pcc_level (sin batch) ────────────────────────")
    pcc_nb = torch.randn(D)
    m1, m2, m3 = mask_pcc_level(pcc_nb, mask_ratio=0.15)
    assert m1.shape == (D,) and m2.shape == (D,) and m3.shape == (D,)
    assert m2.sum() == int(D * 0.15)
    print(f"  ✓ shapes sin batch: {m1.shape}\n")

    # ─── TEST 6: máscaras distintas entre muestras del batch ─────────
    print("── TEST 6: máscaras distintas entre muestras ────────────────")
    _, _, _, roi_mask_b = mask_roi_level(ts, mask_ratio=0.25)
    # Comprobar que al menos dos filas del batch son distintas
    diffs = (roi_mask_b.unsqueeze(0) != roi_mask_b.unsqueeze(1)).any(dim=-1)
    assert diffs[0, 1].item() is True or diffs[0, 2].item() is True, \
        "Las máscaras del batch son idénticas — ¿bug?"
    print(f"  ✓ máscaras distintas por muestra\n")

    # ─── TEST 7: Transforms ──────────────────────────────────────────
    print("── TEST 7: ROIMaskTransform / PCCMaskTransform ──────────────")
    roi_t = ROIMaskTransform(mask_ratio=0.25)
    a, b, c, d = roi_t(torch.randn(T, R))
    assert a.shape == (T, R) and d.shape == (R,)

    pcc_t = PCCMaskTransform(mask_ratio=0.15)
    a, b, c = pcc_t(torch.randn(D))
    assert a.shape == (D,)
    print(f"  ✓ transforms funcionan\n")

    # ─── TEST 8: create_attention_mask_from_roi_mask ─────────────────
    print("── TEST 8: create_attention_mask_from_roi_mask ──────────────")
    attn = create_attention_mask_from_roi_mask(roi_mask, seq_len=T)
    assert attn.shape == (B, T, T)
    assert (attn == 0).all()
    print(f"  ✓ attention mask all-zero de shape {tuple(attn.shape)}\n")

    print("✅ Todos los tests de mask_utils.py pasaron.")