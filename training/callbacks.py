# training/callbacks.py
"""
EarlyStopping basado en validación.
Guarda el state_dict del mejor modelo (en CPU, para no ocupar VRAM) y
expone `restore(model)` para volver a él al terminar el entrenamiento.
"""

import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, model: nn.Module, config: dict):
        self.patience = config["PATIENCE"]
        self.min_delta = config["MIN_DELTA"]
        self.counter = 0
        self.min_val_loss = float("inf")
        self.best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    def __call__(self, model: nn.Module, avg_val_loss: float) -> bool:
        """
        Returns:
            True si se debe detener el entrenamiento (paciencia agotada).
            False en caso contrario.
        """
        delta = self.min_val_loss - avg_val_loss
        if delta >= self.min_delta:
            self.counter = 0
            self.min_val_loss = avg_val_loss
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Carga en `model` los pesos del mejor checkpoint visto."""
        model.load_state_dict(self.best_state)


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    # ─── Modelo dummy ─────────────────────────────────────────────────
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

    config = {"PATIENCE": 3, "MIN_DELTA": 0.01}
    model = Dummy()
    es = EarlyStopping(model, config)

    # ─── TEST 1: primeras mejoras no disparan parada ──────────────────
    print("── TEST 1: mejoras no disparan parada ───────────────────────")
    for loss in [1.0, 0.9, 0.8]:
        stop = es(model, loss)
        assert stop is False, f"loss={loss} disparó parada incorrectamente"
    print(f"  ✓ 3 mejoras seguidas → stop=False, min_val_loss={es.min_val_loss:.2f}\n")

    # ─── TEST 2: paciencia agotada dispara parada ─────────────────────
    print("── TEST 2: paciencia agotada ────────────────────────────────")
    stops = []
    for loss in [0.85, 0.84, 0.83]:  # empeoramientos < min_delta
        stops.append(es(model, loss))
    assert stops[-1] is True, f"stops={stops}"
    print(f"  ✓ 3 empeoramientos → stop={stops}\n")

    # ─── TEST 3: restore devuelve los pesos del mejor loss ────────────
    print("── TEST 3: restore recupera pesos del mejor loss ────────────")
    # Después de entrenar un poco, `restore` debe devolver el estado anterior
    with torch.no_grad():
        model.fc.weight.add_(1.0)   # modifica el modelo

    before = {k: v.clone() for k, v in model.state_dict().items()}
    es.restore(model)
    after = model.state_dict()

    # El state_dict restaurado debe coincidir con es.best_state
    for k in es.best_state:
        assert torch.allclose(after[k].cpu(), es.best_state[k]), f"{k} no restaurado"
    print(f"  ✓ restore() recupera el mejor state_dict\n")

    # ─── TEST 4: best_state en CPU, no en GPU ─────────────────────────
    print("── TEST 4: best_state en CPU ────────────────────────────────")
    for k, v in es.best_state.items():
        assert not v.is_cuda, f"{k} está en GPU"
    print(f"  ✓ tensores almacenados en CPU\n")

    print("✅ Todos los tests de callbacks.py pasaron.")