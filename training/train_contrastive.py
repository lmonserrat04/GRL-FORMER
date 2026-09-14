"""Contrastive GLOBAL — paper Sec. 3.3 + README del repo.

Estrategia: freeze TST1, unfreeze TST2 (mejor combinación empírica y del repo).
InfoNCE τ=0.07, proj 256→128, Adam lr=1e-4 wd=1e-4.
"""
from pathlib import Path
import torch
from tqdm import tqdm
from training.tasks.contrastive import ContrastiveWrapper
from models.transformer_ts import create_transformer_ts
from models.transformer_fc import create_transformer_fc
from data.loaders.dataloader import get_single_split_loaders


def _tst1_cfg(c):
    t = c["TST1"]
    return {"n_rois": c["N_ROIS"], "emb_dim": t["D_MODEL"],
            "n_layers": t["NUM_ENCODER_LAYERS"], "n_heads": t["N_HEADS"],
            "dim_feedforward": t["DIM_FEEDFORWARD"], "dropout": t["ENC_DROP"],
            "max_seq_len": c["MAX_SEQ_LEN"], "use_cls_token": t["USE_CLS_TOKEN"]}


def _tst2_cfg(c):
    t = c["TST2"]
    return {"pcc_dim": t["PCC_DIM"], "d_model": t["D_MODEL"],
            "n_layers": t["NUM_ENCODER_LAYERS"], "n_heads": t["N_HEADS"],
            "dim_feedforward": t["DIM_FEEDFORWARD"], "dropout": t["ENC_DROP"]}


def _train_epoch(tst1, tst2, cm, loader, optimizer, device, trainable):
    tst1.eval(); tst2.train(); cm.train()
    tl, ta, n = 0.0, 0.0, 0
    for batch in loader:
        ts = batch["timeseries"].to(device)
        pcc = batch["pcc_vector"].to(device)
        with torch.no_grad():
            h_ts = tst1(ts, mode='finetune')
        h_fc = tst2(pcc, mode='finetune')
        loss, _, align = cm(h_ts, h_fc)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        tl += loss.item(); ta += align.item(); n += 1
    return tl / max(n, 1), ta / max(n, 1)


@torch.no_grad()
def _validate(tst1, tst2, cm, loader, device):
    tst1.eval(); tst2.eval(); cm.eval()
    vl, va, n = 0.0, 0.0, 0
    for batch in loader:
        ts = batch["timeseries"].to(device)
        pcc = batch["pcc_vector"].to(device)
        h_ts = tst1(ts, mode='finetune')
        h_fc = tst2(pcc, mode='finetune')
        loss, _, align = cm(h_ts, h_fc)
        vl += loss.item(); va += align.item(); n += 1
    return vl / max(n, 1), va / max(n, 1)


def run_contrastive_global(config, save_dir=None):
    device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    phase = config["T_CONTRASTIVE"]

    tst1 = create_transformer_ts(_tst1_cfg(config)).to(device)
    tst2 = create_transformer_fc(_tst2_cfg(config)).to(device)
    if config.get("CKPT_TST1"): tst1.load_pretrained(config["CKPT_TST1"], strict=False)
    if config.get("CKPT_TST2"): tst2.load_pretrained(config["CKPT_TST2"], strict=False)

    cm = ContrastiveWrapper(
        dim_ts=tst1.emb_dim, dim_fc=tst2.d_model,
        hidden_dim=phase["PROJ_HIDDEN_DIM"], output_dim=phase["PROJ_OUTPUT_DIM"],
        temperature=phase["TEMPERATURE"],
    ).to(device)

    # freeze TST1, unfreeze TST2 (README del repo + empíricamente mejor)
    for p in tst1.parameters(): p.requires_grad = False
    for p in tst2.parameters(): p.requires_grad = True

    trainable = ([p for p in tst2.parameters() if p.requires_grad]
                 + list(cm.parameters()))
    optimizer = torch.optim.Adam(
        trainable, lr=float(phase["LR"]),
        weight_decay=float(phase["WEIGHT_DECAY"]),
    )

    train_loader, val_loader, _, _ = get_single_split_loaders(
        config, batch_size=phase["BATCH_SIZE"],
        num_workers=config.get("NUM_WORKERS", 0), seed=config.get("SEED", 42),
    )

    epochs = phase["N_EPOCHS"]
    patience = phase.get("PATIENCE", 20)
    min_delta = phase.get("MIN_DELTA", 1e-4)

    best_val = float("inf")
    best_tst1_state = {k: v.detach().cpu().clone() for k, v in tst1.state_dict().items()}
    best_tst2_state = {k: v.detach().cpu().clone() for k, v in tst2.state_dict().items()}
    best_cm_state   = {k: v.detach().cpu().clone() for k, v in cm.state_dict().items()}
    counter = 0

    with tqdm(range(1, epochs + 1), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Contrastive GLOBAL | Epoch {epoch}")
            tl, ta = _train_epoch(tst1, tst2, cm, train_loader, optimizer, device, trainable)
            vl, va = _validate(tst1, tst2, cm, val_loader, device)
            tepoch.set_postfix(train=f"{tl:.4f}", val=f"{vl:.4f}",
                               align_tr=f"{ta:.3f}", align_val=f"{va:.3f}")

            if vl < best_val - min_delta:
                best_val = vl
                best_tst1_state = {k: v.detach().cpu().clone() for k, v in tst1.state_dict().items()}
                best_tst2_state = {k: v.detach().cpu().clone() for k, v in tst2.state_dict().items()}
                best_cm_state   = {k: v.detach().cpu().clone() for k, v in cm.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    tqdm.write(f"  ⏹ Early stopping contrastive en epoch {epoch} "
                               f"(best val={best_val:.4f})")
                    break

    tst1.load_state_dict(best_tst1_state)
    tst2.load_state_dict(best_tst2_state)
    cm.load_state_dict(best_cm_state)

    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "contrastive_global.pt"
        torch.save({
            "tst1_state_dict":        tst1.state_dict(),
            "tst2_state_dict":        tst2.state_dict(),
            "proj_head_1_state_dict": cm.proj_ts.state_dict(),
            "proj_head_2_state_dict": cm.proj_fc.state_dict(),
        }, path)
        print(f"  💾 {path}")
