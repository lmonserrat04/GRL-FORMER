"""
Contrastive learning task — configuración óptima del paper (Sec. 3.3 + Table 2):
    - InfoNCE loss, τ = 0.07
    - Projection head: input → 256 → 128  (BatchNorm + ReLU)
    - Los dos encoders (TST1 + TST2) se descongelan durante esta fase
      (Sec. 4.3.2: unfreeze both > freeze TST1 > freeze TST2 > freeze both)

ContrastiveWrapper.forward devuelve (loss, z_ts, align) donde:
    loss  : InfoNCE
    z_ts  : proyección de TST1 (batch, output_dim)
    align : similitud coseno media de pares positivos (escalar, diagnóstico)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Pérdida contrastiva InfoNCE bidireccional (τ default 0.07)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_ts: torch.Tensor, h_fc: torch.Tensor) -> torch.Tensor:
        batch_size = h_ts.shape[0]

        # Normalización L2
        h_ts = F.normalize(h_ts, p=2, dim=1)
        h_fc = F.normalize(h_fc, p=2, dim=1)

        # Matriz de similitud (batch, batch); positivos en la diagonal
        sim = torch.matmul(h_ts, h_fc.T) / self.temperature
        labels = torch.arange(batch_size, device=h_ts.device)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveWrapper(nn.Module):
    """
    Cabezales de proyección + pérdida InfoNCE.

    Config óptima (Table 2): hidden_dim=256, output_dim=128, temperature=0.07.
    """

    def __init__(
        self,
        dim_ts: int,
        dim_fc: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.proj_ts = ProjectionHead(dim_ts, hidden_dim, output_dim)
        self.proj_fc = ProjectionHead(dim_fc, hidden_dim, output_dim)
        self.criterion = InfoNCELoss(temperature)

    def forward(self, h_ts: torch.Tensor, h_fc: torch.Tensor):
        z_ts = self.proj_ts(h_ts)
        z_fc = self.proj_fc(h_fc)
        loss = self.criterion(z_ts, z_fc)

        # Diagnóstico: similitud coseno media de pares positivos
        z_ts_n = F.normalize(z_ts, p=2, dim=1)
        z_fc_n = F.normalize(z_fc, p=2, dim=1)
        align = (z_ts_n * z_fc_n).sum(dim=1).mean()

        return loss, z_ts, align

        
class ContrastiveTask(nn.Module):
    def __init__(self, dim_ts, dim_fc, hidden_dim=256, output_dim=128,
                 temperature=0.07, device=None):
        super().__init__()
        self.contrastive_module = ContrastiveWrapper(
            dim_ts=dim_ts, dim_fc=dim_fc,
            hidden_dim=hidden_dim, output_dim=output_dim,
            temperature=temperature,
        )
        self._device = device or torch.device("cpu")

    def execution_step(self, model, timeseries, pcc_vector):
        timeseries = timeseries.to(self._device)
        pcc_vector = pcc_vector.to(self._device)
        h_ts, h_fc = model.get_features(timeseries, pcc_vector)
        loss, _, _ = self.contrastive_module(h_ts, h_fc)
        return loss
# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    B, D_TS, D_FC = 16, 512, 256

    # ─── TEST 1: InfoNCELoss ──────────────────────────────────────────
    print("── TEST 1: InfoNCELoss ──────────────────────────────────────")
    h_ts = torch.randn(B, D_TS)   # salidas de encoder (antes de proyectar)
    h_fc = torch.randn(B, D_FC)
    D_PROJ = 128
    z_ts = torch.randn(B, D_PROJ)  # proyecciones
    z_fc = torch.randn(B, D_PROJ)

    crit = InfoNCELoss(temperature=0.07)
    loss = crit(z_ts, z_fc)
    assert loss.dim() == 0 and loss.item() > 0
    print(f"  ✓ loss escalar: {loss.item():.4f}")

    # Pares perfectos → menor loss que aleatorios
    same = torch.randn(B, D_PROJ)
    loss_perfect = crit(same, same.clone())
    loss_random = crit(torch.randn(B, D_PROJ), torch.randn(B, D_PROJ))
    assert loss_perfect.item() < loss_random.item()
    print(f"  ✓ pares perfectos {loss_perfect.item():.4f} < aleatorios {loss_random.item():.4f}\n")

    # ─── TEST 2: ProjectionHead ───────────────────────────────────────
    print("── TEST 2: ProjectionHead ───────────────────────────────────")
    head = ProjectionHead(D_TS, hidden_dim=256, output_dim=128)
    head.train()  # BatchNorm con batch>1
    z = head(h_ts)
    assert z.shape == (B, 128)
    print(f"  ✓ output {tuple(z.shape)}\n")

    # ─── TEST 3: ContrastiveWrapper (firma exacta de train_finetune_min) ──
    print("── TEST 3: ContrastiveWrapper ───────────────────────────────")
    wrapper = ContrastiveWrapper(dim_ts=D_TS, dim_fc=D_FC,
                                 temperature=0.07, hidden_dim=256, output_dim=128)
    
    loss, z_ts, align = wrapper(h_ts, h_fc)

    assert loss.dim() == 0
    assert z_ts.shape == (B, 128)
    assert align.dim() == 0
    assert -1.0 <= align.item() <= 1.0
    print(f"  ✓ loss={loss.item():.4f}  z_ts {tuple(z_ts.shape)}  align={align.item():.4f}\n")

    # ─── TEST 4: 3-tuple unpacking como en train_finetune_min ─────────
    print("── TEST 4: unpacking (loss, _, align) ───────────────────────")
    loss, _, align = wrapper(h_ts, h_fc)
    print(f"  ✓ unpacking OK → loss={loss.item():.4f}  align={align.item():.4f}\n")

    # ─── TEST 5: defaults coinciden con la config del paper ───────────
    print("── TEST 5: defaults == config del paper ─────────────────────")
    w = ContrastiveWrapper(dim_ts=D_TS, dim_fc=D_FC)
    assert w.criterion.temperature == 0.07
    assert w.proj_ts.net[0].out_features == 256
    assert w.proj_ts.net[-1].out_features == 128
    print(f"  ✓ τ=0.07  hidden=256  output=128\n")

    print("✅ Todos los tests de contrastive.py pasaron.")