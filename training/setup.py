import torch
import torch.nn as nn

def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.get("LR", 1e-4),
        weight_decay=config.get("WEIGHT_DECAY", 1e-2)
    )

def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get("T_0", 10)
    )

def build_criterion(config: dict) -> nn.Module:
    return nn.CrossEntropyLoss(
        label_smoothing=config["LABEL_SMOOTHING"]
    )