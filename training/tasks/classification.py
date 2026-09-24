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

    def __init__(self, device):
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

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


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Modelo dummy: logits deterministas ───────────────────────────
    class DummyDualStream(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.num_classes = num_classes

        def forward(self, ts, pcc):
            # logits derivados del input (para que el gradiente fluya)
            s = ts.mean(dim=(1, 2)) + pcc.mean(dim=1)
            return torch.stack([s, -s], dim=1)[:, : self.num_classes]

    task = ClassificationTask(device)
    B, T, R, D = 8, 100, 200, 19900

    # ─── TEST 1: loss escalar y positiva ──────────────────────────────
    print("── TEST 1: loss escalar y positiva ─────────────────────────")
    ts = torch.randn(B, T, R)
    pcc = torch.randn(B, D)
    y = torch.randint(0, 2, (B,))
    loss = task.execution_step(DummyDualStream(), ts, pcc, y)
    assert loss.dim() == 0
    assert loss.item() > 0
    print(f"  ✓ loss={loss.item():.4f}\n")

    # ─── TEST 2: predicciones perfectas → loss baja ───────────────────
    print("── TEST 2: predicciones perfectas vs aleatorias ────────────")

    class FixedLogits(nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits

        def forward(self, ts, pcc):
            return self.logits

    # Logits muy confiados y correctos
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    logits_perfect = torch.tensor([[10.0, -10.0], [-10.0, 10.0]] * 4)
    loss_perfect = task.execution_step(
        FixedLogits(logits_perfect), ts[:len(y)], pcc[:len(y)], y
    )

    # Logits al azar
    logits_random = torch.randn(len(y), 2)
    loss_random = task.execution_step(
        FixedLogits(logits_random), ts[:len(y)], pcc[:len(y)], y
    )
    assert loss_perfect.item() < loss_random.item()
    print(f"  ✓ loss perfecta={loss_perfect.item():.4f}  <  "
          f"aleatoria={loss_random.item():.4f}\n")

    # ─── TEST 3: gradient flow ────────────────────────────────────────
    print("── TEST 3: gradient flow ────────────────────────────────────")

    class LearnableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, ts, pcc):
            s = (ts.mean(dim=(1, 2)) + pcc.mean(dim=1)) * self.scale
            return torch.stack([s, -s], dim=1)

    m = LearnableModel()
    loss = task.execution_step(m, ts, pcc, y)
    loss.backward()
    assert m.scale.grad is not None and m.scale.grad.abs().item() > 0
    print(f"  ✓ gradiente fluye, grad={m.scale.grad.item():.4f}\n")

    # ─── TEST 4: entrena y baja la loss ───────────────────────────────
    print("── TEST 4: entrena y baja la loss en 20 pasos ───────────────")
    m = LearnableModel()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.1)

    # Objetivo trivial: que la loss media baje
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = task.execution_step(m, ts, pcc, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"{losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"  ✓ loss {losses[0]:.4f} → {losses[-1]:.4f}\n")

    print("✅ Todos los tests de classification.py pasaron.")