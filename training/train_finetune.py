"""
Finetuning training and validation functions.
Uses ExperimentContext for clean argument passing.
"""

import torch
import torch.nn as nn
from training.context import ExperimentContext
from training.tasks.classification import ClassificationTask


def train_one_epoch(ctx: ExperimentContext):
    """Train one epoch for finetuning.

    Args:
        ctx: Experiment context containing model, task, optimizer, loaders, device.

    Returns:
        Total loss over the training set.
    """
    model = ctx.model
    task: ClassificationTask = ctx.task
    optimizer = ctx.optimizer
    train_loader = ctx.train_loader
    device = ctx.device

    total_loss = 0.0

    for batch in train_loader:
        ts = batch['timeseries'].to(device)
        pcc = batch['pcc_vector'].to(device)
        y = batch['label'].to(device)

        optimizer.zero_grad()

        loss = task.execution_step(model, ts_batch=ts, pcc_batch=pcc, targets=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def validate(ctx: ExperimentContext):
    """Validate finetuning.

    Args:
        ctx: Experiment context.

    Returns:
        Total loss over the validation set.
    """
    model = ctx.model
    task: ClassificationTask = ctx.task
    val_loader = ctx.val_loader
    device = ctx.device

    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            ts = batch['timeseries'].to(device)
            pcc = batch['pcc_vector'].to(device)
            y = batch['label'].to(device)

            loss = task.execution_step(model, ts_batch=ts, pcc_batch=pcc, targets=y)

            total_loss += loss.item()

    return total_loss