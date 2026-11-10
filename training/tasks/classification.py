"""
Classification task — usada en finetune.

Firma que espera el trainer:
    loss = task.execution_step(model, ts_batch, pcc_batch, targets)

El modelo es DualStreamModel y su forward devuelve logits (B, num_classes).
Criterio: CrossEntropyLoss (Ec. 16 del paper).

Si el clasificador auxiliar sobre h_ts está habilitado, la pérdida global es:
    L = CE(logits_fused, y) + lambda * CE(logits_ts, y)
"""

import torch
import torch.nn as nn

from models.dual_stream import DualStreamModel


class ClassificationTask:
    """CrossEntropy sobre fused, con pérdida auxiliar opcional sobre h_ts."""

    def __init__(self, device, lambda_aux: float = 0.0, use_aux: bool = False):
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_aux = float(lambda_aux)
        self.use_aux = bool(use_aux) and self.lambda_aux != 0.0

        if self.lambda_aux < 0:
            raise ValueError("lambda_aux debe ser >= 0.")

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
            Pérdida global escalar y, opcionalmente, los logits principales
            (los de fused, que son los que se usan para las métricas).
        """

        ts_batch = ts_batch.to(self.device)
        pcc_batch = pcc_batch.to(self.device)
        targets = targets.to(self.device)

        if self.use_aux:
            logits, aux_logits = model(
                ts_batch, pcc_batch, return_aux_logits=True
            )
            loss = self.criterion(logits, targets)
            loss = loss + self.lambda_aux * self.criterion(aux_logits, targets)
        else:
            logits = model(ts_batch, pcc_batch)
            loss = self.criterion(logits, targets)

        if return_logits:
            return logits, loss
        else:
            return loss


