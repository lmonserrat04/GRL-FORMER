"""Finetune — usa build_experiment con ckpt_contrastive.
Projections congeladas del contrastive global.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from training.setup import build_experiment
from data.loaders.dataloader import get_finetune_loaders
from utils.metrics import compute_metrics, aggregate_window_predictions_to_subject_level


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in loader:
        ts = batch["timeseries"].to(device)
        pcc = batch["pcc_vector"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(ts, pcc)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def validate(model, loader, criterion, device):
    model.eval(); total = 0.0
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)
            logits = model(ts, pcc)
            total += criterion(logits, y).item()
            p = torch.softmax(logits, dim=1)[:, 1]
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())
    return total / len(loader), compute_metrics(np.array(labels), np.array(preds), np.array(probs))


def finetune_fold(config, fold_idx, save_dir=None):
    config["EXPERIMENT_TYPE"] = "finetune"

    # ─── Todo el modelo lo construye la factory ──────────────────────
    exp = build_experiment(
        config, fold_idx=fold_idx,
        ckpt_contrastive=config.get("CKPT_CONTRASTIVE"),
    )
    
    model = exp.model
    optimizer = exp.optimizer
    scheduler = exp.scheduler
    train_loader = exp.train_loader
    val_loader = exp.val_loader
    device = exp.device

    if config.get("CKPT_CONTRASTIVE"):
        print(f"  ✓ Projections ← {config['CKPT_CONTRASTIVE']}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\n── Finetune fold {fold_idx} ──")
    print(f"  Trainable: {sum(p.numel() for p in trainable):,}/{total_p:,}")

    # ─── Test loader (la factory solo devuelve train/val) ────────────
    phase = config["FINETUNING"]
    _, _, test_loader, split_info = get_finetune_loaders(
        config, batch_size=phase["BATCH_SIZE"],
        num_workers=config.get("NUM_WORKERS", 0),
        fold_idx=fold_idx, n_folds=config.get("N_FOLDS", 5),
        seed=config.get("SEED", 42),
        eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
    )

    epochs = phase["N_EPOCHS"]
    criterion = nn.CrossEntropyLoss()
    patience = phase.get("PATIENCE", 20)
    best_auc, best_state, pc = -1.0, None, 0

    with tqdm(range(1, epochs + 1), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Finetune fold {fold_idx} | Epoch {epoch}")
            tl = train_epoch(model, train_loader, optimizer, criterion, device)
            vl, vm = validate(model, val_loader, criterion, device)
            scheduler.step()
            auc = vm["auc"]
            tepoch.set_postfix(train=f"{tl:.4f}", val=f"{vl:.4f}", auc=f"{auc:.4f}")
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                pc = 0
            else:
                pc += 1
                if pc >= patience:
                    tqdm.write(f"  ⏹ Early stopping epoch {epoch} (best AUC={best_auc:.4f})")
                    break

    if best_state: model.load_state_dict(best_state)

    # ─── Test + agregación subject-level ─────────────────────────────
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)
            logits = model(ts, pcc)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    si = split_info["subject_indices"]; ti = split_info["test_idx"]
    n_subj = len(np.unique(si[ti]))
    if len(ti) > n_subj:
        yt, yp, ypr = aggregate_window_predictions_to_subject_level(
            labels, preds, probs, ti, si, strategy=config.get("SUBJECT_AGG", "majority_vote"))
        print(f"\n  Fold {fold_idx}: subject-level ({len(ti)} → {n_subj} sujetos)")
    else:
        yt, yp, ypr = np.array(labels), np.array(preds), np.array(probs)

    m = compute_metrics(yt, yp, ypr)
    print(f"  AUC={m['auc']:.4f}  ACC={m['accuracy']:.4f}  Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  F1={m['f1']:.4f}")

    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "metrics": m},
                   save_dir / f"best_finetune_fold_{fold_idx}.pt")
        print(f"  💾 best_finetune_fold_{fold_idx}.pt")
    return m