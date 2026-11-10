"""Tarea de clasificación para la GNN temporal.

Misma interfaz que `ClassificationTask`, pero sin el vector PCC: la GNN
solo consume la serie temporal (B, T, R).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GNNClassificationTask:
    """CrossEntropy sobre los logits de `TemporalGNN`."""

    def __init__(self, device, label_smoothing: float = 0.0):
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def execution_step(
        self,
        model: nn.Module,
        ts_batch: torch.Tensor,
        targets: torch.Tensor,
        pcc_matrix: torch.Tensor | None = None,
        return_logits: bool = False,
    ):
        """
        Args:
            model:      TemporalGNN.
            ts_batch:   (B, T, R) series BOLD.
            targets:    (B,) etiquetas 0/1.
            pcc_matrix: (B, R, R) PCC cacheada por sujeto (modo static_fc).
            return_logits: devuelve (logits, loss) si es True.

        Returns:
            loss, o (logits, loss) si `return_logits=True`.
        """
        logits = model(ts_batch, pcc_matrix=pcc_matrix)
        loss = self.criterion(logits, targets)
        if return_logits:
            return logits, loss
        return loss
