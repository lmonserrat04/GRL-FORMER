"""Finetune — projections congeladas del contrastive global."""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data.loaders.dataloader import get_finetune_loaders
from models.dual_stream import create_dual_stream_model
from training.tasks.contrastive import ProjectionHead
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


def _load_projection_heads(config, device):
    ckpt_path = config.get("CKPT_CONTRASTIVE")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"No existe CKPT_CONTRASTIVE={ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    phase = config["T_CONTRASTIVE"]
    p1 = ProjectionHead(config["TST1"]["D_MODEL"], phase["PROJ_HIDDEN_DIM"], phase["PROJ_OUTPUT_DIM"]).to(device)
    p2 = ProjectionHead(config["TST2"]["D_MODEL"], phase["PROJ_HIDDEN_DIM"], phase["PROJ_OUTPUT_DIM"]).to(device)
    p1.load_state_dict(ckpt["proj_head_1_state_dict"])
    p2.load_state_dict(ckpt["proj_head_2_state_dict"])
    for p in p1.parameters(): p.requires_grad = False
    for p in p2.parameters(): p.requires_grad = False
    return p1, p2


def finetune_fold(config, fold_idx, save_dir=None):
    device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    phase = config["FINETUNING"]; ds = config["DUAL_STREAM"]
    fh = config.get("FUSION", {}).get("ATTENTION_POOLING", {}).get("HIDDEN_DIM")

    train_loader, val_loader, test_loader, split_info = get_finetune_loaders(
        config, batch_size=phase["BATCH_SIZE"], num_workers=config.get("NUM_WORKERS", 0),
        fold_idx=fold_idx, n_folds=config.get("N_FOLDS", 5),
        seed=config.get("SEED", 42), eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
    )

    p1, p2 = _load_projection_heads(config, device)
    print(f"  ✓ Projections ← {config['CKPT_CONTRASTIVE']}")

    model = create_dual_stream_model(
        n_rois=config["N_ROIS"], time_points=config["MAX_SEQ_LEN"],
        pcc_dim=config["TST2"]["PCC_DIM"],
        tst1_emb_dim=config["TST1"]["D_MODEL"], tst2_d_model=config["TST2"]["D_MODEL"],
        fusion_type=ds["FUSION_TYPE"], fusion_hidden_dim=fh,
        num_classes=ds["NUM_CLASSES"], dropout=ds["CLASSIFIER_DROPOUT"],
        mlp_dims=ds.get("MLP_DIMS"), proj_head_1=p1, proj_head_2=p2,
    ).to(device)

    if config.get("CKPT_TST1"): model.load_pretrained_tst1(config["CKPT_TST1"], strict=False)
    if config.get("CKPT_TST2"): model.load_pretrained_tst2(config["CKPT_TST2"], strict=False)

    model.unfreeze_encoders()
    for p in model.proj_head_1.parameters(): p.requires_grad = False
    for p in model.proj_head_2.parameters(): p.requires_grad = False

    print(f"\n── Finetune fold {fold_idx} ──")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable: {sum(p.numel() for p in trainable):,}/{sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(trainable, lr=float(phase["LR"]), weight_decay=float(phase["WEIGHT_DECAY"]))
    epochs = phase["N_EPOCHS"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(phase["LR"])*0.01)
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
        torch.save({"model_state_dict": model.state_dict(), "metrics": m}, save_dir / f"best_finetune_fold_{fold_idx}.pt")
        print(f"  💾 best_finetune_fold_{fold_idx}.pt")
    return m
