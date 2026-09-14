"""
Finetune — clasificación ASD/TC con projections congeladas del contrastive global.

Pipeline por fold:
  1. Cargar TST1 + TST2 + projection heads del checkpoint contrastive global.
  2. Congelar projection heads.
  3. Finetune con attention_pooling, encoders descongelados, lr=5e-5.
  4. Evaluación test con agregación a nivel de sujeto.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data.loaders.dataloader import get_finetune_loaders
from models.dual_stream import create_dual_stream_model
from training.tasks.contrastive import ProjectionHead
from utils.metrics import (
    compute_metrics,
    aggregate_window_predictions_to_subject_level,
)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        ts = batch["timeseries"].to(device)
        pcc = batch["pcc_vector"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ts, pcc)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)

            logits = model(ts, pcc)
            total_loss += criterion(logits, y).item()

            p = torch.softmax(logits, dim=1)[:, 1]
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    metrics = compute_metrics(np.array(labels), np.array(preds), np.array(probs))
    return total_loss / len(loader), metrics


def _load_projection_heads(config, device):
    """Crea y carga proj heads del checkpoint contrastive. Las devuelve congeladas."""
    ckpt_path = config.get("CKPT_CONTRASTIVE")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"No se encontró CKPT_CONTRASTIVE={ckpt_path}. "
            "Corre run_contrastive_global() antes del finetuning."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    phase = config["T_CONTRASTIVE"]

    proj_1 = ProjectionHead(
        input_dim=config["TST1"]["D_MODEL"],
        hidden_dim=phase["PROJ_HIDDEN_DIM"],
        output_dim=phase["PROJ_OUTPUT_DIM"],
    ).to(device)
    proj_2 = ProjectionHead(
        input_dim=config["TST2"]["D_MODEL"],
        hidden_dim=phase["PROJ_HIDDEN_DIM"],
        output_dim=phase["PROJ_OUTPUT_DIM"],
    ).to(device)

    proj_1.load_state_dict(ckpt["proj_head_1_state_dict"])
    proj_2.load_state_dict(ckpt["proj_head_2_state_dict"])

    for p in proj_1.parameters():
        p.requires_grad = False
    for p in proj_2.parameters():
        p.requires_grad = False

    return proj_1, proj_2


def finetune_fold(config, fold_idx, save_dir=None):
    """Entrena y evalúa un fold. Projections congeladas del contrastive global."""
    device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    phase = config["FINETUNING"]
    ds = config["DUAL_STREAM"]
    fusion_hidden = config.get("FUSION", {}).get("ATTENTION_POOLING", {}).get("HIDDEN_DIM")

    # 1. Loaders
    train_loader, val_loader, test_loader, split_info = get_finetune_loaders(
        config,
        batch_size=phase["BATCH_SIZE"],
        num_workers=config.get("NUM_WORKERS", 0),
        fold_idx=fold_idx,
        n_folds=config.get("N_FOLDS", 5),
        seed=config.get("SEED", 42),
        eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
    )

    # 2. Projections congeladas del contrastive
    proj_1, proj_2 = _load_projection_heads(config, device)
    print(f"  ✓ Projections ← {config['CKPT_CONTRASTIVE']}")

    # 3. Modelo dual stream CON projections
    model = create_dual_stream_model(
        n_rois=config["N_ROIS"],
        time_points=config["MAX_SEQ_LEN"],
        pcc_dim=config["TST2"]["PCC_DIM"],
        tst1_emb_dim=config["TST1"]["D_MODEL"],
        tst2_d_model=config["TST2"]["D_MODEL"],
        fusion_type=ds["FUSION_TYPE"],
        fusion_hidden_dim=fusion_hidden,
        num_classes=ds["NUM_CLASSES"],
        dropout=ds["CLASSIFIER_DROPOUT"],
        mlp_dims=ds.get("MLP_DIMS"),
        proj_head_1=proj_1,
        proj_head_2=proj_2,
    ).to(device)

    if config.get("CKPT_TST1"):
        model.load_pretrained_tst1(config["CKPT_TST1"], strict=False)
    if config.get("CKPT_TST2"):
        model.load_pretrained_tst2(config["CKPT_TST2"], strict=False)

    # Encoders descongelados (paper Table 4)
    model.unfreeze_encoders()
    # Garantizar projections congeladas
    for p in model.proj_head_1.parameters():
        p.requires_grad = False
    for p in model.proj_head_2.parameters():
        p.requires_grad = False

    # 4. Finetune
    print(f"\n── Finetune fold {fold_idx} (encoders unfrozen, proj frozen) ──")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_train:,}/{n_total:,}")

    optimizer = torch.optim.Adam(
        trainable,
        lr=float(phase["LR"]),
        weight_decay=float(phase["WEIGHT_DECAY"]),
    )
    epochs = phase["N_EPOCHS"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(phase["LR"]) * 0.01
    )
    criterion = nn.CrossEntropyLoss()

    patience = phase.get("PATIENCE", 20)
    best_val_auc = -1.0
    best_state = None
    patience_counter = 0

    with tqdm(range(1, epochs + 1), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Finetune fold {fold_idx} | Epoch {epoch}")

            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()

            val_auc = val_metrics["auc"]
            tepoch.set_postfix(train=f"{train_loss:.4f}",
                               val=f"{val_loss:.4f}",
                               auc=f"{val_auc:.4f}")

            if val_auc > best_val_auc + 1e-4:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"  ⏹ Early stopping en epoch {epoch} "
                               f"(best val AUC={best_val_auc:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 5. Evaluación test
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)

            logits = model(ts, pcc)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    subj_indices = split_info["subject_indices"]
    test_idx = split_info["test_idx"]
    n_test_subjects = len(np.unique(subj_indices[test_idx]))

    if len(test_idx) > n_test_subjects:
        yt, yp, yprob = aggregate_window_predictions_to_subject_level(
            all_labels, all_preds, all_probs,
            test_idx, subj_indices,
            strategy=config.get("SUBJECT_AGG", "majority_vote"),
        )
        print(f"\n  Fold {fold_idx}: subject-level "
              f"({len(test_idx)} muestras → {n_test_subjects} sujetos)")
    else:
        yt, yp, yprob = np.array(all_labels), np.array(all_preds), np.array(all_probs)

    test_metrics = compute_metrics(yt, yp, yprob)
    print(f"  AUC={test_metrics['auc']:.4f}  ACC={test_metrics['accuracy']:.4f}  "
          f"Sens={test_metrics['sensitivity']:.4f}  Spec={test_metrics['specificity']:.4f}  "
          f"F1={test_metrics['f1']:.4f}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"best_finetune_fold_{fold_idx}.pt"
        torch.save({"model_state_dict": model.state_dict(),
                    "metrics": test_metrics}, path)
        print(f"  💾 {path}")

    return test_metrics