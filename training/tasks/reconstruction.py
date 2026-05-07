import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true, mask):
    
        return nn.functional.mse_loss(y_pred[mask], y_true[mask], reduction=self.reduction)
    

class ReconstructionTask:
    """
    Estrategia de Tarea para el Pre-entrenamiento.
    Desacopla el loop de entrenamiento de la lógica del Transformer.
    """
    def __init__(self, device):
        self.criterion = MaskedMSELoss().to(device)

    def execution_step(self, model, masked_batch, mask, targets):
        """
        Ejecuta un paso de forward para la tarea de reconstrucción.
        """
        
        pred = model(masked_batch, mode = 'pretrain')
        
        loss = self.criterion(pred, targets, mask)
        
        return loss