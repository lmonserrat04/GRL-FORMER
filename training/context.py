# training/context.py
"""
Contexto de un experimento: agrupa modelo, tarea, optimizer, scheduler y loaders.
El tipo `TaskType` es solo para anotación — en runtime no valida nada.
"""

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from training.tasks.reconstruction import ReconstructionTask
from training.tasks.contrastive import ContrastiveTask
from training.tasks.classification import ClassificationTask


TaskType = Union[ReconstructionTask, ContrastiveTask, ClassificationTask]


@dataclass
class ExperimentContext:
    model: nn.Module
    task: TaskType
    optimizer: Optimizer
    scheduler: LRScheduler
    train_loader: DataLoader
    val_loader: DataLoader
    device: torch.device