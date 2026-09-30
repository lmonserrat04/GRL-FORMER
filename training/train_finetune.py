"""Finetune — usa build_experiment con ckpt_contrastive.
Projections congeladas del contrastive global.
Guarda predicciones crudas (labels, probs, umbral óptimo) por fold.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from training.setup import build_experiment
from data.loaders.dataloader import get_finetune_loaders
from utils.metrics import compute_metrics, aggregate_window_predictions_to_subject_level
from models.grl import ganin_lambda


def train_epoch(model, loader, optimizer, task, device):
    model.train()
    total = 0.0
    for batch in loader:
        ts = batch["timeseries"].to(device)
        pcc = batch["pcc_vector"].to(device)
        y = batch["label"].to(device)
        site = batch["site_id"].to(device)

        optimizer.zero_grad()
        tag_loss, domain_loss = task.execution_step(
            model, ts, pcc, y, domain_targets=site,
        )
        
        w = model._domain_weight if hasattr(model, "_domain_weight") else 1.0
        loss = tag_loss + w * domain_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def validate(model, loader, task, device):
    model.eval()
    total = 0.0
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)
            site = batch["site_id"].to(device)

            tag_logits, tag_loss, domain_loss = task.execution_step(
                model, ts, pcc, y,
                domain_targets=site,
                return_tag_logits=True,
            )
            total += (tag_loss + domain_loss).item()

            p = torch.softmax(tag_logits, dim=1)[:, 1]
            preds.extend(torch.argmax(tag_logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    return total / len(loader), compute_metrics(
        np.array(labels), np.array(preds), np.array(probs)
    )


def finetune_fold(config, fold_idx, save_dir=None):
    config["EXPERIMENT_TYPE"] = "finetune"

    exp = build_experiment(
        config, fold_idx=fold_idx,
        ckpt_contrastive=config.get("CKPT_CONTRASTIVE"),
    )

    model = exp.model
    task = exp.task
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

    phase = config["FINETUNING"]
    _, _, test_loader, split_info = get_finetune_loaders(
        config, batch_size=phase["BATCH_SIZE"],
        num_workers=config.get("NUM_WORKERS", 0),
        fold_idx=fold_idx, n_folds=config.get("N_FOLDS", 5),
        seed=config.get("SEED", 42),
        eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
    )

    epochs = phase["N_EPOCHS"]
    patience = phase.get("PATIENCE", 20)
    best_auc, best_state, pc = -1.0, None, 0

    use_schedule = bool(phase.get("GRL_SCHEDULE", False))
    gamma = float(phase.get("GRL_GAMMA", 10.0))
    warmup = int(phase.get("GRL_WARMUP", 0))
    total_epochs = phase["N_EPOCHS"]
    print(f"  GRL: {'schedule' if use_schedule else 'fijo'} "
          f"(lambda_0={model.grl_lambda:.3f}, gamma={gamma}, warmup={warmup})")

    with tqdm(range(1, epochs + 1), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Finetune fold {fold_idx} | Epoch {epoch}")

            if use_schedule:
                if epoch <= warmup:
                    model.grl_lambda = 0.0
                else:
                    progress = (epoch - warmup - 1) / max(total_epochs - warmup - 1, 1)
                    model.grl_lambda = ganin_lambda(progress, gamma=gamma)

            tl = train_epoch(model, train_loader, optimizer, task, device)
            vl, vm = validate(model, val_loader, task, device)
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

    if best_state:
        model.load_state_dict(best_state)

    # ─── Predicciones en VAL para umbral óptimo ──────────────────────
    model.eval()
    val_labels, val_probs = [], []
    with torch.no_grad():
        for batch in val_loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)
            site = batch["site_id"].to(device)
            tag_logits, _, _ = task.execution_step(
                model, ts, pcc, y,
                domain_targets=site,
                return_tag_logits=True,
            )
            probs = torch.softmax(tag_logits, dim=1)[:, 1]
            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    val_labels = np.array(val_labels)
    val_probs = np.array(val_probs)

    from sklearn.metrics import roc_curve
    if len(np.unique(val_labels)) > 1:
        fpr, tpr, thr = roc_curve(val_labels, val_probs)
        j = tpr - fpr
        optimal_thr = float(thr[j.argmax()])
    else:
        optimal_thr = 0.5
    print(f"  Umbral óptimo (val, Youden J): {optimal_thr:.4f}")

    # ─── Evaluación test ─────────────────────────────────────────────
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)
            tag_logits = model(ts, pcc, return_domain_logits=False)
            probs.extend(torch.softmax(tag_logits, dim=1)[:, 1].cpu().numpy())
            preds.extend(torch.argmax(tag_logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    labels = np.array(labels)
    probs = np.array(probs)
    preds = np.array(preds)

    # ─── Guardar predicciones crudas ─────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_dir / f"preds_fold_{fold_idx}.npz",
            labels=labels,
            probs=probs,
            preds=preds,
            val_labels=val_labels,
            val_probs=val_probs,
            optimal_thr=np.array([optimal_thr]),
            test_idx=split_info["test_idx"],
            subject_indices=split_info["subject_indices"],
        )

    # ─── Agregación subject-level ────────────────────────────────────
    si = split_info["subject_indices"]; ti = split_info["test_idx"]
    n_subj = len(np.unique(si[ti]))
    if len(ti) > n_subj:
        yt, yp, ypr = aggregate_window_predictions_to_subject_level(
            labels.tolist(), preds.tolist(), probs.tolist(),
            ti, si, strategy=config.get("SUBJECT_AGG", "majority_vote"))
        print(f"\n  Fold {fold_idx}: subject-level ({len(ti)} → {n_subj} sujetos)")
    else:
        yt, yp, ypr = labels, preds, probs

    m = compute_metrics(yt, yp, ypr)
    print(f"  AUC={m['auc']:.4f}  ACC={m['accuracy']:.4f}  "
          f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  "
          f"F1={m['f1']:.4f}")

    if save_dir is not None:
        torch.save(
            {"model_state_dict": model.state_dict(), "metrics": m},
            save_dir / f"best_finetune_fold_{fold_idx}.pt",
        )
        print(f"  💾 best_finetune_fold_{fold_idx}.pt")
    return m
