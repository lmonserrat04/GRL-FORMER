"""
Contrastive learning training and validation functions.
Uses ExperimentContext for clean argument passing.
"""

import torch
from training.context import ExperimentContext
from training.tasks.contrastive import ContrastiveTask


def train_one_epoch(ctx: ExperimentContext):
    """Train one epoch for contrastive learning.

    Args:
        ctx: Experiment context containing model, task, optimizer, loaders, device.

    Returns:
        Total loss over the training set.
    """
    model = ctx.model
    task: ContrastiveTask = ctx.task
    optimizer = ctx.optimizer
    train_loader = ctx.train_loader
    device = ctx.device

    total_loss = 0.0

    model.train()
    task.contrastive_module.train()

    for batch, _ in train_loader:
        timeseries = batch['timeseries'].to(device)
        pcc_vector = batch['pcc_vector'].to(device)

        optimizer.zero_grad()
        loss = task.execution_step(model, timeseries, pcc_vector)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def validate(ctx: ExperimentContext):
    """Validate contrastive learning.

    Args:
        ctx: Experiment context.

    Returns:
        Total loss over the validation set.
    """
    model = ctx.model
    task: ContrastiveTask = ctx.task
    val_loader = ctx.val_loader
    device = ctx.device

    total_loss = 0.0

    model.eval()
    task.contrastive_module.eval()

    with torch.no_grad():
        for batch, _ in val_loader:
            timeseries = batch['timeseries'].to(device)
            pcc_vector = batch['pcc_vector'].to(device)

            loss = task.execution_step(model, timeseries, pcc_vector)

            total_loss += loss.item()

    return total_loss