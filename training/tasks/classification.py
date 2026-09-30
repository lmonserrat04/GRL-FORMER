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
import torch.nn.functional as F

from models.dual_stream import DualStreamModel


class ClassificationTask:
    """CrossEntropy sobre los logits del DualStreamModel."""

    def __init__(self, device):
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def execution_step(
        self,
        model,
        ts_batch: torch.Tensor,
        pcc_batch: torch.Tensor,
        tag_targets: torch.Tensor,
        domain_targets: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        return_tag_logits: bool = False,
        return_domain_logits: bool = False,
    ):
        if return_domain_logits and domain_targets is None:
            raise ValueError(
                "return_domain_logits=True requiere domain_targets."
            )

        ts_batch = ts_batch.to(self.device)
        pcc_batch = pcc_batch.to(self.device)
        tag_targets = tag_targets.to(self.device)
        if domain_targets is not None:
            domain_targets = domain_targets.to(self.device)

        need_domain = domain_targets is not None

        # ─── Forward ─────────────────────────────────────────────────
        if need_domain:
            outputs = model(ts_batch, pcc_batch, return_domain_logits=True)
            tag_logits = outputs[0]
            domain_logits_list = list(outputs[1:])
        else:
            tag_logits = model(ts_batch, pcc_batch, return_domain_logits=False)
            domain_logits_list = []

        # ─── Tag loss (opcionalmente ponderada por muestra) ────────
        if sample_weights is not None:
            per_sample = F.cross_entropy(
                tag_logits, tag_targets, reduction="none"
            )
            tag_loss = (per_sample * sample_weights).mean()
        else:
            tag_loss = self.criterion(tag_logits, tag_targets)

        # ─── Domain loss (media sobre las locations activas) ────────
        if need_domain and domain_logits_list:
            dom_loss = sum(
                self.criterion(dl, domain_targets)
                for dl in domain_logits_list
            ) / len(domain_logits_list)
        else:
            dom_loss = None

        # ─── Return ──────────────────────────────────────────────────
        out = []
        if return_tag_logits:
            out.append(tag_logits)
        out.append(tag_loss)
        if dom_loss is not None:
            out.append(dom_loss)
        if return_domain_logits:
            out.extend(domain_logits_list)
        return out[0] if len(out) == 1 else tuple(out)