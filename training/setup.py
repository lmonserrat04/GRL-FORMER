import torch
import torch.nn as nn

def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["LR"]),
        weight_decay=float(config["WEIGHT_DECAY"])
    )

def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config.get("T_0", 10))
    )

def build_criterion(config: dict) -> nn.Module:
    return nn.CrossEntropyLoss(
        label_smoothing=float(config["LABEL_SMOOTHING"])
    )