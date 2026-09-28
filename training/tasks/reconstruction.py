"""
Reconstruction task — usada en pretrain TST1 (ROI-level mask) y TST2 (element-level mask).

Ambos trainers llaman:
    loss = task.execution_step(model, masked_batch, mask, target)

La máscara puede tener la forma de la salida del modelo:
    - TST1: (B, T, R)
    - TST2: (B, D)
En ambos casos se hace boolean indexing sobre `pred` y `target`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionTask:
    """MSE sobre posiciones enmascaradas."""

    def __init__(self, device):
        self.device = device

    def execution_step(
        self,
        model: nn.Module,
        masked_batch: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            model:        TST1 o TST2 (forward en modo 'pretrain').
            masked_batch: entrada con las posiciones enmascaradas a cero.
            mask:         booleano, True = posición enmascarada.
            target:       valores originales antes del enmascaramiento.

        Returns:
            loss escalar (MSE sobre posiciones enmascaradas).
        """
        masked_batch = masked_batch.to(self.device)
        mask = mask.to(self.device)
        target = target.to(self.device)

        pred = model(masked_batch, mode='pretrain')

        loss = F.mse_loss(pred[mask], target[mask], reduction='mean')
        return loss

