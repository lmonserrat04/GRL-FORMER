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


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Modelo dummy: devuelve el input tal cual ─────────────────────
    class IdentityPretrainModel(nn.Module):
        def forward(self, x, mode='pretrain'):
            return x

    # ─── Modelo dummy: devuelve ceros ─────────────────────────────────
    class ZeroPretrainModel(nn.Module):
        def forward(self, x, mode='pretrain'):
            return torch.zeros_like(x)

    task = ReconstructionTask(device)

    # ─── TEST 1: modelo identidad → loss = 0 sobre la máscara ─────────
    # ─── TEST 1: modelo identidad → loss = mean(target[mask]²) ───────
    print("── TEST 1: modelo identidad → loss = mean(target[mask]²) ───")
    B, T, R = 4, 100, 200
    x = torch.randn(B, T, R)
    mask = torch.zeros(B, T, R, dtype=torch.bool)
    mask[:, ::4, :] = True   # 25% de timesteps
    masked_x = x.clone()
    masked_x[mask] = 0.0

    loss = task.execution_step(IdentityPretrainModel(), masked_x, mask, x)
    expected = (x[mask] ** 2).mean()
    assert loss.dim() == 0
    assert torch.isclose(loss, expected, atol=1e-6), \
        f"loss={loss.item():.6f} vs expected={expected.item():.6f}"
    print(f"  ✓ loss={loss.item():.6f} == mean(target²)={expected.item():.6f}\n")



    # ─── TEST 2: modelo cero → loss = MSE(0, target[mask]) ────────────
    print("── TEST 2: modelo cero → loss = mean(target[mask]²) ─────────")
    loss = task.execution_step(ZeroPretrainModel(), masked_x, mask, x)
    expected = (x[mask] ** 2).mean()
    assert torch.isclose(loss, expected, atol=1e-6), \
        f"loss={loss.item():.6f} vs expected={expected.item():.6f}"
    print(f"  ✓ loss={loss.item():.6f} == mean(target²)={expected.item():.6f}\n")

    # ─── TEST 3: forma TST2 (B, D) ────────────────────────────────────
    print("── TEST 3: forma TST2 (B, D) ────────────────────────────────")
    B, D = 4, 19900
    x_fc = torch.randn(B, D)
    mask_fc = torch.rand(B, D) > 0.85
    masked_fc = x_fc.clone()
    masked_fc[mask_fc] = 0.0

    loss = task.execution_step(IdentityPretrainModel(), masked_fc, mask_fc, x_fc)
    expected = (x_fc[mask_fc] ** 2).mean()
    assert torch.isclose(loss, expected, atol=1e-6), \
        f"loss={loss.item():.6f} vs expected={expected.item():.6f}"
    print(f"  ✓ loss={loss.item():.6f} == mean(target²)={expected.item():.6f} (forma FC)\n")

    # ─── TEST 4: la loss NO depende de posiciones no enmascaradas ─────
    print("── TEST 4: loss ignora posiciones no enmascaradas ──────────")
    x2 = x.clone()
    x2[~mask] = 999.0

    loss_a = task.execution_step(IdentityPretrainModel(), masked_x, mask, x)
    loss_b = task.execution_step(IdentityPretrainModel(), masked_x, mask, x2)
    # masked_x=0 en posiciones enmascaradas; x[mask] y x2[mask] son idénticas.
    # Como la loss solo mira posiciones enmascaradas, ambas deben coincidir.
    assert torch.isclose(loss_a, loss_b), f"{loss_a.item()} vs {loss_b.item()}"
    print(f"  ✓ loss no cambia al perturbar posiciones no enmascaradas\n")

    # ─── TEST 5: gradient flow ────────────────────────────────────────
    print("── TEST 5: gradient flow ────────────────────────────────────")

    class LearnablePretrain(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(0.5))

        def forward(self, x, mode='pretrain'):
            return x + self.bias   # bias afecta también posiciones enmascaradas

    m = LearnablePretrain()
    loss = task.execution_step(m, masked_x, mask, x)
    loss.backward()
    assert m.bias.grad is not None and m.bias.grad.abs().item() > 0, \
        f"grad={m.bias.grad}"
    print(f"  ✓ gradiente fluye, grad={m.bias.grad.item():.4f}\n")

    print("✅ Todos los tests de reconstruction.py pasaron.")