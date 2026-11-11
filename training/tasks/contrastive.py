"""Contrastive learning task — pérdida InfoNCE sobre features ya extraídas.

La task NO ejecuta el modelo. Recibe (h_ts, h_fc) ya computados por el
trainer y devuelve (loss, align).

Diseño:
  - h_ts viene de TST1 (features en modo finetune)
  - h_fc viene de TST2 (features en modo finetune)
  - El trainer es responsable del freezing/no_grad de TST1

Objetivos disponibles (``T_CONTRASTIVE.OBJECTIVE``):

  ``subject``
      InfoNCE original del paper: el positivo de un ancla es el mismo
      sujeto en la otra rama.

  ``diagnosis_site``
      InfoNCE supervisada por diagnóstico con exclusión intra-sitio:
        * positivos: mismo diagnóstico y sitio distinto;
        * negativos: diagnóstico opuesto, cualquier sitio (incluido el
          mismo sitio);
        * excluidos: mismo diagnóstico y mismo sitio (incluido el propio
          sujeto), fuera del numerador y del denominador.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE bidireccional (τ default 0.07)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_ts: torch.Tensor, h_fc: torch.Tensor) -> torch.Tensor:
        B = h_ts.shape[0]
        h_ts = F.normalize(h_ts, p=2, dim=1)
        h_fc = F.normalize(h_fc, p=2, dim=1)
        sim = torch.matmul(h_ts, h_fc.T) / self.temperature
        labels = torch.arange(B, device=h_ts.device)
        loss_ts2fc = F.cross_entropy(sim, labels)
        loss_fc2ts = F.cross_entropy(sim.T, labels)
        return (loss_ts2fc + loss_fc2ts) / 2


def diagnosis_site_masks(labels: torch.Tensor, sites: torch.Tensor):
    """Máscaras (positivos, negativos, válidos) para la regla del experimento.

    Args:
        labels: (B,) enteros.
        sites:  (B,) enteros (sitio codificado).

    Returns:
        positive: (B, B) mismo diagnóstico y sitio distinto.
        negative: (B, B) diagnóstico opuesto (cualquier sitio).
        valid:    (B, B) positive | negative.
    """
    labels = labels.view(-1, 1) if torch.is_tensor(labels) else torch.as_tensor(np.asarray(labels)).view(-1, 1)
    sites = _factorize_sites(sites, labels.device).view(-1, 1)
    same_label = labels.eq(labels.T)
    same_site = sites.eq(sites.T)
    positive = same_label & ~same_site
    negative = ~same_label
    valid = positive | negative
    return positive, negative, valid


class DiagnosisSiteInfoNCELoss(nn.Module):
    """InfoNCE supervisada por diagnóstico con exclusión intra-sitio.

    Para un ancla i:

        L_i = -1/|P(i)| Σ_{p∈P(i)}
              log exp(z_i·z_p/τ) / Σ_{a∈P(i)∪N(i)} exp(z_i·z_a/τ)

    Las anclas sin positivos aportan 0 y se incluyen en el denominador de
    la media (media sobre todos los anclajes).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _direction(sim: torch.Tensor, positive: torch.Tensor,
                   valid: torch.Tensor) -> torch.Tensor:
        n = sim.size(0)
        has_valid = valid.any(dim=1)
        has_positive = positive.any(dim=1)

        masked = sim.masked_fill(~valid, float("-inf"))
        log_denom = torch.logsumexp(masked, dim=1)
        log_denom = torch.where(
            has_valid, log_denom, torch.zeros_like(log_denom)
        )

        n_pos = positive.sum(dim=1).float()
        log_prob = (sim - log_denom.unsqueeze(1)) * positive.float()
        per_anchor = -log_prob.sum(dim=1) / n_pos.clamp(min=1.0)
        per_anchor = per_anchor * has_positive.float()
        return per_anchor.sum() / n

    def forward(self, h_ts: torch.Tensor, h_fc: torch.Tensor,
                labels: torch.Tensor, sites: torch.Tensor) -> torch.Tensor:
        z_ts = F.normalize(h_ts, p=2, dim=1)
        z_fc = F.normalize(h_fc, p=2, dim=1)
        sim = torch.matmul(z_ts, z_fc.T) / self.temperature

        if not torch.is_tensor(sites):
            sites = _factorize_sites(sites, z_ts.device)
        positive, _, valid = diagnosis_site_masks(labels, sites)
        loss_ts2fc = self._direction(sim, positive, valid)
        loss_fc2ts = self._direction(sim.T, positive.T, valid.T)
        return (loss_ts2fc + loss_fc2ts) / 2


class ProjectionHead(nn.Module):
    """MLP de proyección: input → hidden → output (BatchNorm + ReLU)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def _factorize_sites(sites, device) -> torch.Tensor:
    """Convierte sitios (str, list o tensor) a códigos enteros en ``device``."""
    if torch.is_tensor(sites):
        return sites.to(device).long()
    arr = np.asarray(sites)
    _, inverse = np.unique(arr, return_inverse=True)
    return torch.as_tensor(inverse, device=device, dtype=torch.long)


def positive_alignment(z_ts: torch.Tensor, z_fc: torch.Tensor,
                       positive: torch.Tensor) -> torch.Tensor:
    """Coseno medio de los pares positivos (0 si no hay positivos)."""
    n_pos = positive.sum()
    if int(n_pos) == 0:
        return z_ts.new_zeros(())
    cos = torch.matmul(z_ts, z_fc.T)
    return (cos * positive.float()).sum() / n_pos


def positive_coverage(labels: torch.Tensor, sites) -> torch.Tensor:
    """Fracción de anclas con al menos un positivo válido en el batch."""
    labels = labels.view(-1).long()
    sites = _factorize_sites(sites, labels.device)
    positive, _, _ = diagnosis_site_masks(labels, sites)
    return positive.any(dim=1).float().mean()


class ContrastiveWrapper(nn.Module):
    """Proj heads + pérdida contrastiva. Devuelve (loss, z_ts, align)."""

    def __init__(self, dim_ts, dim_fc, hidden_dim=256, output_dim=128,
                 temperature=0.07, objective="subject"):
        super().__init__()
        if objective not in ("subject", "diagnosis_site"):
            raise ValueError(
                f"OBJECTIVE desconocido: {objective!r}. "
                "Usa 'subject' o 'diagnosis_site'."
            )
        self.objective = objective
        self.proj_ts = ProjectionHead(dim_ts, hidden_dim, output_dim)
        self.proj_fc = ProjectionHead(dim_fc, hidden_dim, output_dim)
        if objective == "diagnosis_site":
            self.criterion = DiagnosisSiteInfoNCELoss(temperature)
        else:
            self.criterion = InfoNCELoss(temperature)

    def forward(self, h_ts, h_fc, labels=None, sites=None):
        z_ts = self.proj_ts(h_ts)
        z_fc = self.proj_fc(h_fc)

        if self.objective == "diagnosis_site":
            if labels is None or sites is None:
                raise ValueError(
                    "OBJECTIVE=diagnosis_site requiere labels y sites en el batch."
                )
            labels = labels.to(z_ts.device).long()
            sites = _factorize_sites(sites, z_ts.device)
            loss = self.criterion(z_ts, z_fc, labels, sites)
            positive, _, _ = diagnosis_site_masks(labels, sites)
            z_ts_n = F.normalize(z_ts, p=2, dim=1)
            z_fc_n = F.normalize(z_fc, p=2, dim=1)
            align = positive_alignment(z_ts_n, z_fc_n, positive)
            return loss, z_ts, align

        loss = self.criterion(z_ts, z_fc)
        # Diagnóstico: cosine sim de positivos por identidad.
        z_ts_n = F.normalize(z_ts, p=2, dim=1)
        z_fc_n = F.normalize(z_fc, p=2, dim=1)
        align = (z_ts_n * z_fc_n).sum(dim=1).mean()
        return loss, z_ts, align


class ContrastiveTask(nn.Module):
    """
    Task contrastiva. Solo calcula la pérdida a partir de features extraídas.

    Uso en el trainer:
        h_ts = tst1(ts, mode='finetune')       # ← el trainer extrae features
        h_fc = tst2(pcc, mode='finetune')
        loss, align = task.execution_step(h_ts, h_fc, labels, sites)
    """

    def __init__(self, dim_ts, dim_fc, hidden_dim=256, output_dim=128,
                 temperature=0.07, objective="subject"):
        super().__init__()
        self.objective = objective
        self.contrastive_module = ContrastiveWrapper(
            dim_ts=dim_ts, dim_fc=dim_fc,
            hidden_dim=hidden_dim, output_dim=output_dim,
            temperature=temperature, objective=objective,
        )

    def execution_step(self, h_ts: torch.Tensor, h_fc: torch.Tensor,
                       labels: torch.Tensor | None = None,
                       sites=None):
        """
        Args:
            h_ts: features de TST1 (B, dim_ts)
            h_fc: features de TST2 (B, dim_fc)
            labels: (B,) diagnóstico (obligatorio si objective=diagnosis_site)
            sites:  (B,) sitio; str/list/tensor (obligatorio si diagnosis_site)

        Returns:
            (loss, align) — ambos escalares
        """
        loss, _, align = self.contrastive_module(
            h_ts, h_fc, labels=labels, sites=sites
        )
        return loss, align
