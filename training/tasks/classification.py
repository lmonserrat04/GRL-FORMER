"""
Classification task — usada en finetune.

Firma que espera el trainer:
    loss = task.execution_step(model, ts_batch, pcc_batch, targets)

El modelo es DualStreamModel y su forward devuelve logits (B, num_classes).
Criterio: CrossEntropyLoss (Ec. 16 del paper).
"""

import torch
import torch.nn as nn

from models.dual_stream import DualStreamModel


class ClassificationTask:
    """CrossEntropy sobre los logits del DualStreamModel."""

    def __init__(self, device, criterion=None):
        """
        Args:
            device:    torch.device
            criterion: función de pérdida opcional. Por defecto usa
                       nn.CrossEntropyLoss(). Para focal loss, pasar
                       training.losses.FocalLoss(...).
        """
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def execution_step(
        self,
        model: DualStreamModel,
        ts_batch: torch.Tensor,
        pcc_batch: torch.Tensor,
        targets: torch.Tensor,
        return_logits : bool = False
    ) -> torch.Tensor | tuple:
        """
        Args:
            model:     DualStreamModel.
            ts_batch:  (B, T, R) series temporales.
            pcc_batch: (B, D) vectores PCC.
            targets:   (B,) etiquetas enteras.

        Returns:
            loss escalar (CrossEntropy) y opcionalmente logits.
        """

        ts_batch = ts_batch.to(self.device)
        pcc_batch = pcc_batch.to(self.device)
        targets = targets.to(self.device)

        logits = model(ts_batch, pcc_batch)

        if return_logits: 
            return logits , self.criterion(logits, targets)
        else:
            return self.criterion(logits, targets)


