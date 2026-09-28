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


