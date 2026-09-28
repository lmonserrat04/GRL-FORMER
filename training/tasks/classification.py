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
        tag_targets: torch.Tensor,
        domain_targets: torch.Tensor | None = None,
        return_tag_logits: bool = False,
        return_domain_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Ejecuta un paso de forward + cálculo de pérdidas para DualStreamModel.

        Args:
            model:
                Modelo DualStreamModel.
            ts_batch:
                Tensor (B, T, R) con series temporales.
            pcc_batch:
                Tensor (B, D) con vectores PCC.
            tag_targets:
                Tensor (B,2) con etiquetas enteras para clasificación de tag/diagnóstico.
            domain_targets:
                Tensor (B,N_SITES) con etiquetas enteras para clasificación de dominio/sitio.
                Si es None, no se calcula pérdida de dominio.
            return_tag_logits:
                Si es True, incluye los logits de tag en la salida.
            return_domain_logits:
                Si es True, incluye los logits de dominio en la salida.
                Requiere que domain_targets no sea None; en caso contrario
                se lanza ValueError.

        Returns:
            Dependiendo de los flags y de si hay domain_targets, retorna:
            - tag_loss (Tensor escalar) si no se pide nada más.
            - (tag_logits, tag_loss) si return_tag_logits=True y no hay domain_targets.
            - (tag_loss, domain_loss) si hay domain_targets y no se piden logits.
            - (tag_logits, tag_loss, domain_loss) si return_tag_logits=True y hay domain_targets.
            - (tag_loss, domain_loss, domain_logits) si return_domain_logits=True.
            - (tag_logits, tag_loss, domain_loss, domain_logits) si ambos flags son True.

            Orden de la tupla:
                (tag_logits?, tag_loss, domain_loss?, domain_logits?)
            donde '?' indica que el elemento solo aparece si se solicita.

        Raises:
            ValueError:
                Si return_domain_logits=True y domain_targets es None.
        """
        if return_domain_logits and domain_targets is None:
            raise ValueError(
                "return_domain_logits=True requiere domain_targets; "
                "no tiene sentido devolver logits de dominio sin etiquetas de dominio."
            )

        ts_batch = ts_batch.to(self.device)
        pcc_batch = pcc_batch.to(self.device)
        tag_targets = tag_targets.to(self.device)

        if domain_targets is not None:
            domain_targets = domain_targets.to(self.device)

        need_domain_logits = domain_targets is not None

        if need_domain_logits:
            tag_logits, domain_logits = model(
                ts_batch,
                pcc_batch,
                return_domain_logits=True,
            )
        else:
            tag_logits = model(
                ts_batch,
                pcc_batch,
                return_domain_logits=False,
            )
            domain_logits = None

        tag_loss = self.criterion(tag_logits, tag_targets)

        outputs: list[torch.Tensor] = []

        if return_tag_logits:
            outputs.append(tag_logits)

        outputs.append(tag_loss)

        if domain_targets is not None:
            domain_loss = self.criterion(domain_logits, domain_targets)
            outputs.append(domain_loss)

        if return_domain_logits:
            outputs.append(domain_logits)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)


            
            


        
   

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
    loss = task.domain_execution_step(DummyDualStream(), ts, pcc, y)
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
    loss_perfect = task.domain_execution_step(
        FixedLogits(logits_perfect), ts[:len(y)], pcc[:len(y)], y
    )

    # Logits al azar
    logits_random = torch.randn(len(y), 2)
    loss_random = task.domain_execution_step(
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
    loss = task.domain_execution_step(m, ts, pcc, y)
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
        loss = task.domain_execution_step(m, ts, pcc, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"{losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"  ✓ loss {losses[0]:.4f} → {losses[-1]:.4f}\n")

    print("✅ Todos los tests de classification.py pasaron.")