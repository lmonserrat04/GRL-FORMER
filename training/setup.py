import torch
import torch.nn as nn

from data.loaders.dataloader import get_dataloader
from data.preprocessing.harmonization import GlobalNormalizer, ResidualHarmonizer
from model.models import build_model

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

def compute_loss(outputs, labels, criterion, config):
    return criterion(outputs, labels)


def build_dataloaders(config, df_train, df_val, df_test, harmonizer, normalizer):
    train_loader = get_dataloader(config, df_train, split='train', 
                                  normalizer=normalizer, harmonizer=harmonizer)
    val_loader   = get_dataloader(config, df_val,   split='val',   
                                  normalizer=normalizer, harmonizer=harmonizer)
    test_loader  = get_dataloader(config, df_test,  split='test',  
                                  normalizer=normalizer, harmonizer=harmonizer)
    return train_loader, val_loader, test_loader

def build_experiment(config, df_train, df_val, df_test):
    harmonizer = ResidualHarmonizer(config["FACTORS"])
    normalizer = GlobalNormalizer()
    
    train_loader, val_loader, test_loader = build_dataloaders(
        config, df_train, df_val, df_test, harmonizer, normalizer
    )
    model     = build_model(config).to(config["DEVICE"])
    optimizer = build_optimizer(model, config)
    criterion = build_criterion(config)
    scheduler = build_scheduler(optimizer, config)
    
    return model, optimizer, criterion, scheduler, train_loader, val_loader, test_loader