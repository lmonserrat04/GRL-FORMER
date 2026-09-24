"""Contrastive learning task — pérdida InfoNCE sobre features ya extraídas.

La task NO ejecuta el modelo. Recibe (h_ts, h_fc) ya computados por el
trainer y devuelve (loss, align).

Diseño:
  - h_ts viene de TST1 (features en modo finetune)
  - h_fc viene de TST2 (features en modo finetune)
  - El trainer es responsable del freezing/no_grad de TST1
"""
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


class ContrastiveWrapper(nn.Module):
    """Proj heads + InfoNCE. Devuelve (loss, z_ts, align)."""

    def __init__(self, dim_ts, dim_fc, hidden_dim=256, output_dim=128,
                 temperature=0.07):
        super().__init__()
        self.proj_ts = ProjectionHead(dim_ts, hidden_dim, output_dim)
        self.proj_fc = ProjectionHead(dim_fc, hidden_dim, output_dim)
        self.criterion = InfoNCELoss(temperature)

    def forward(self, h_ts, h_fc):
        z_ts = self.proj_ts(h_ts)
        z_fc = self.proj_fc(h_fc)
        loss = self.criterion(z_ts, z_fc)

        # Diagnóstico: cosine sim de positivos
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
        loss, align = task.execution_step(h_ts, h_fc)
    """

    def __init__(self, dim_ts, dim_fc, hidden_dim=256, output_dim=128,
                 temperature=0.07):
        super().__init__()
        self.contrastive_module = ContrastiveWrapper(
            dim_ts=dim_ts, dim_fc=dim_fc,
            hidden_dim=hidden_dim, output_dim=output_dim,
            temperature=temperature,
        )

    def execution_step(self, h_ts: torch.Tensor, h_fc: torch.Tensor):
        """
        Args:
            h_ts: features de TST1 (B, dim_ts)
            h_fc: features de TST2 (B, dim_fc)

        Returns:
            (loss, align) — ambos escalares
        """
        loss, _, align = self.contrastive_module(h_ts, h_fc)
        return loss, align