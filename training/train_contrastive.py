"""Contrastive GLOBAL — paper Sec. 3.3 + README del repo.

Estrategia: freeze TST1, unfreeze TST2.
InfoNCE τ=0.07, proj 256→128, Adam lr=1e-4 wd=1e-4.

El objetivo contrastivo se selecciona con ``T_CONTRASTIVE.OBJECTIVE``:

  - ``subject`` (paper): positivo = mismo sujeto en la otra rama.
  - ``diagnosis_site``: positivo = mismo diagnóstico de otro sitio;
    negativos = diagnóstico opuesto (cualquier sitio); se excluyen los
    pares mismo diagnóstico + mismo sitio.
"""
import json
from pathlib import Path

import torch
from tqdm import tqdm

from training.tasks.contrastive import (
    ContrastiveTask,
    positive_coverage,
)
from models.transformer_ts import create_transformer_ts
from models.transformer_fc import create_transformer_fc
from data.loaders.dataloader import (
    get_single_split_loaders,
    get_contrastive_loaders,
)


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


def _phase_objective(config) -> str:
    return config["T_CONTRASTIVE"].get("OBJECTIVE", "subject")


def build_contrastive_components(config, device):
    """
    Construye (tst1, tst2, task, trainable_params).

    Freezing: TST1 frozen, TST2 unfrozen.
    """
    phase = config["T_CONTRASTIVE"]

    tst1 = create_transformer_ts(_tst1_cfg(config)).to(device)
    tst2 = create_transformer_fc(_tst2_cfg(config)).to(device)

    if config.get("CKPT_TST1"):
        tst1.load_pretrained(config["CKPT_TST1"], strict=False)
    if config.get("CKPT_TST2"):
        tst2.load_pretrained(config["CKPT_TST2"], strict=False)

    task = ContrastiveTask(
        dim_ts=tst1.emb_dim, dim_fc=tst2.d_model,
        hidden_dim=phase["PROJ_HIDDEN_DIM"],
        output_dim=phase["PROJ_OUTPUT_DIM"],
        temperature=phase["TEMPERATURE"],
        objective=phase.get("OBJECTIVE", "subject"),
    ).to(device)

    for p in tst1.parameters(): p.requires_grad = False
    for p in tst2.parameters(): p.requires_grad = True

    trainable = ([p for p in tst2.parameters() if p.requires_grad]
                 + list(task.parameters()))
    return tst1, tst2, task, trainable


def _batch_inputs(batch, device):
    ts = batch["timeseries"].to(device)
    pcc = batch["pcc_vector"].to(device)
    labels = batch.get("label")
    sites = batch.get("site")
    if labels is not None:
        labels = labels.to(device)
    return ts, pcc, labels, sites


def _train_epoch(tst1, tst2, task, loader, optimizer, device, trainable,
                 objective):
    tst1.eval(); tst2.train(); task.train()
    tl, ta, tc, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        ts, pcc, labels, sites = _batch_inputs(batch, device)

        with torch.no_grad():
            h_ts = tst1(ts, mode='finetune')
        h_fc = tst2(pcc, mode='finetune')

        loss, align = task.execution_step(h_ts, h_fc, labels, sites)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        tl += loss.item(); ta += align.item()
        if objective == "diagnosis_site" and labels is not None:
            tc += float(positive_coverage(labels, sites).item())
        n += 1
    denom = max(n, 1)
    return tl / denom, ta / denom, (tc / denom if objective == "diagnosis_site" else float("nan"))


@torch.no_grad()
def _validate(tst1, tst2, task, loader, device, objective):
    tst1.eval(); tst2.eval(); task.eval()
    vl, va, vc, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        ts, pcc, labels, sites = _batch_inputs(batch, device)
        h_ts = tst1(ts, mode='finetune')
        h_fc = tst2(pcc, mode='finetune')
        loss, align = task.execution_step(h_ts, h_fc, labels, sites)
        vl += loss.item(); va += align.item()
        if objective == "diagnosis_site" and labels is not None:
            vc += float(positive_coverage(labels, sites).item())
        n += 1
    denom = max(n, 1)
    return vl / denom, va / denom, (vc / denom if objective == "diagnosis_site" else float("nan"))


def _set_sampler_epoch(loader, epoch: int) -> None:
    sampler = getattr(loader, "batch_sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _build_loaders(config, phase, objective):
    if objective == "diagnosis_site":
        return get_contrastive_loaders(
            config,
            batch_size=phase["BATCH_SIZE"],
            num_workers=config.get("NUM_WORKERS", 0),
            seed=config.get("SEED", 42),
            min_sites_per_class=phase.get("MIN_SITES_PER_CLASS", 4),
        )
    return get_single_split_loaders(
        config, batch_size=phase["BATCH_SIZE"],
        num_workers=config.get("NUM_WORKERS", 0), seed=config.get("SEED", 42),
    )


def _snapshot(tst1, tst2, task):
    return {
        "tst1": {k: v.detach().cpu().clone() for k, v in tst1.state_dict().items()},
        "tst2": {k: v.detach().cpu().clone() for k, v in tst2.state_dict().items()},
        "task": {k: v.detach().cpu().clone() for k, v in task.state_dict().items()},
    }


def run_contrastive_global(config, save_dir=None):
    device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    phase = config["T_CONTRASTIVE"]
    objective = _phase_objective(config)

    tst1, tst2, task, trainable = build_contrastive_components(config, device)

    optimizer = torch.optim.Adam(
        trainable, lr=float(phase["LR"]),
        weight_decay=float(phase["WEIGHT_DECAY"]),
    )

    train_loader, val_loader, _, split_info = _build_loaders(
        config, phase, objective
    )

    epochs = phase["N_EPOCHS"]
    patience = phase.get("PATIENCE", 20)
    min_delta = phase.get("MIN_DELTA", 1e-4)

    best_val = float("inf")
    best_epoch = 0
    best_state = _snapshot(tst1, tst2, task)
    counter = 0
    history = []

    print(f"  Objetivo contrastivo: {objective}")
    if objective == "diagnosis_site":
        probs = split_info.get("class_probs") or []
        probs_txt = " / ".join(f"{p:.3f}" for p in probs)
        print(f"  Batch sampler: estratificado por diagnóstico "
              f"(p = {probs_txt}), >= {split_info.get('min_sites_per_class')} "
              f"sitios por clase, positivos validados")

    with tqdm(range(1, epochs + 1), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Contrastive GLOBAL | Epoch {epoch}")
            _set_sampler_epoch(train_loader, epoch)
            _set_sampler_epoch(val_loader, 0)

            tl, ta, tc = _train_epoch(
                tst1, tst2, task, train_loader, optimizer, device,
                trainable, objective,
            )
            vl, va, vc = _validate(
                tst1, tst2, task, val_loader, device, objective
            )
            tepoch.set_postfix(train=f"{tl:.4f}", val=f"{vl:.4f}",
                               align_tr=f"{ta:.3f}", align_val=f"{va:.3f}")
            history.append({
                "epoch": epoch,
                "train_loss": tl,
                "val_loss": vl,
                "align_train": ta,
                "align_val": va,
                "positive_coverage_train": tc,
                "positive_coverage_val": vc,
            })

            if vl < best_val - min_delta:
                best_val = vl
                best_epoch = epoch
                best_state = _snapshot(tst1, tst2, task)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    tqdm.write(f"  ⏹ Early stopping contrastive en epoch {epoch} "
                               f"(best val={best_val:.4f})")
                    break

    tst1.load_state_dict(best_state["tst1"])
    tst2.load_state_dict(best_state["tst2"])
    task.load_state_dict(best_state["task"])

    metrics = {
        "objective": objective,
        "n_epochs_requested": int(epochs),
        "n_epochs_run": len(history),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "batch_size": int(phase["BATCH_SIZE"]),
        "temperature": float(phase["TEMPERATURE"]),
        "sampler": split_info.get("sampler"),
        "min_sites_per_class": split_info.get("min_sites_per_class"),
        "class_probs": split_info.get("class_probs"),
        "split_sizes": {
            "train": int(len(split_info["train_idx"])),
            "val": int(len(split_info["val_idx"])),
            "test": int(len(split_info["test_idx"])),
        },
        "history": history,
    }

    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "contrastive_global.pt"
        torch.save({
            "tst1_state_dict":        tst1.state_dict(),
            "tst2_state_dict":        tst2.state_dict(),
            "proj_head_1_state_dict": task.contrastive_module.proj_ts.state_dict(),
            "proj_head_2_state_dict": task.contrastive_module.proj_fc.state_dict(),
            "objective":              objective,
            "best_epoch":             int(best_epoch),
        }, path)
        print(f"  💾 {path}")

        history_path = save_dir / "contrastive_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  💾 {history_path}")

    return metrics
