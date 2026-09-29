"""
Classification task — usada en finetune.

Soporta dos modos de domain adversarial training, según model.multilayer:

  - single (multilayer=False):
      model devuelve (tag_logits, domain_logits)
      1 domain loss.

  - multi (multilayer=True):
      model devuelve (tag_logits, dom_ts, dom_fc, dom_fused)
      3 domain losses, se combinan como (ts + fc + fused) / 3.

Criterio: CrossEntropyLoss (Ec. 16 del paper).
"""

import torch
import torch.nn as nn

from models.dual_stream import DualStreamModel


class ClassificationTask:
    """CrossEntropy sobre los logits del DualStreamModel."""

    def __init__(self, device):
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def execution_step(
        self,
        model: DualStreamModel,
        ts_batch: torch.Tensor,
        pcc_batch: torch.Tensor,
        tag_targets: torch.Tensor,
        domain_targets: torch.Tensor | None = None,
        return_tag_logits: bool = False,
        return_domain_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Forward + cálculo de pérdidas.

        Returns:
            Dependiendo de los flags y del modo (single/multi), retorna:
              Sin domain_targets:
                - tag_loss                                        (flags off)
                - (tag_logits, tag_loss)                          (return_tag_logits)
              Single-layer con domain_targets:
                - (tag_loss, domain_loss)
                - (tag_logits, tag_loss, domain_loss)             (return_tag_logits)
                - (tag_loss, domain_loss, domain_logits)          (return_domain_logits)
                - (tag_logits, tag_loss, domain_loss, domain_logits)
              Multi-layer con domain_targets:
                - (tag_loss, domain_loss)
                - (tag_logits, tag_loss, domain_loss)             (return_tag_logits)
                - (tag_loss, domain_loss, dom_ts, dom_fc, dom_fused)              (return_domain_logits)
                - (tag_logits, tag_loss, domain_loss, dom_ts, dom_fc, dom_fused)

            Donde domain_loss es la media de las losses disponibles
            (1 en single, 3 en multi).
        """
        if return_domain_logits and domain_targets is None:
            raise ValueError(
                "return_domain_logits=True requiere domain_targets; "
                "no tiene sentido devolver logits de dominio sin etiquetas."
            )

        ts_batch = ts_batch.to(self.device)
        pcc_batch = pcc_batch.to(self.device)
        tag_targets = tag_targets.to(self.device)

        if domain_targets is not None:
            domain_targets = domain_targets.to(self.device)

        need_domain = domain_targets is not None
        is_multilayer = getattr(model, "multilayer", False)

        # ─── Forward ─────────────────────────────────────────────────
        if need_domain:
            if is_multilayer:
                tag_logits, dom_ts, dom_fc, dom_fused = model(
                    ts_batch, pcc_batch, return_domain_logits=True,
                )
            else:
                tag_logits, dom_fused = model(
                    ts_batch, pcc_batch, return_domain_logits=True,
                )
                dom_ts, dom_fc = None, None
        else:
            tag_logits = model(
                ts_batch, pcc_batch, return_domain_logits=False,
            )
            dom_ts, dom_fc, dom_fused = None, None, None

        # ─── Tag loss ────────────────────────────────────────────────
        tag_loss = self.criterion(tag_logits, tag_targets)

        # ─── Domain loss(es) ─────────────────────────────────────────
        if need_domain:
            if is_multilayer:
                dom_loss = (
                    self.criterion(dom_ts, domain_targets)
                    + self.criterion(dom_fc, domain_targets)
                    + self.criterion(dom_fused, domain_targets)
                ) / 3.0
            else:
                dom_loss = self.criterion(dom_fused, domain_targets)
        else:
            dom_loss = None

        # ─── Construcción de la tupla de retorno ─────────────────────
        outputs: list[torch.Tensor] = []

        if return_tag_logits:
            outputs.append(tag_logits)

        outputs.append(tag_loss)

        if dom_loss is not None:
            outputs.append(dom_loss)

        if return_domain_logits:
            if is_multilayer:
                outputs.extend([dom_ts, dom_fc, dom_fused])
            else:
                outputs.append(dom_fused)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)
