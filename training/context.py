# training/context.py
from dataclasses import dataclass
from typing import Union
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# Importamos las clases reales de tarea (ajusta las rutas si son diferentes)
from training.tasks.reconstruction import ReconstructionTask
from training.tasks.contrastive import ContrastiveTask
from training.tasks.classification import ClassificationTask

# Tipo que agrupa cualquier tarea posible
TaskType = Union[ReconstructionTask, ContrastiveTask, ClassificationTask]

@dataclass
class ExperimentContext:
    model: nn.Module
    task: TaskType                # ← ahora sí tiene tipo concreto
    optimizer: Optimizer
    scheduler: LRScheduler
    train_loader: DataLoader
    val_loader: DataLoader
    device: torch.device