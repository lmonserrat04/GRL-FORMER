# training/setup.py
"""
Factory de experimento. Construye el ExperimentContext para las fases
pretrain_ts, pretrain_fc y finetune según config["EXPERIMENT_TYPE"].

La fase contrastive NO se construye aquí: vive dentro del finetune
(config["T_CONTRASTIVE"] se lee en train_finetune.py).

Los loaders se construyen leyendo .1D + CSV en runtime (sin .pkl intermedio).
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from training.tasks.contrastive import ContrastiveTask
from data.loaders.dataloader import get_pretrain_loaders, get_finetune_loaders
from models.transformer_ts import create_transformer_ts
from models.transformer_fc import create_transformer_fc
from models.dual_stream import create_dual_stream_model

from training.context import ExperimentContext
from training.tasks.reconstruction import ReconstructionTask
from training.tasks.classification import ClassificationTask


# ──────────────────────────────────────────────────────────────────────
# Registries
# ──────────────────────────────────────────────────────────────────────

OPTIMIZER_REGISTRY = {
    "Adam":  optim.Adam,
    "AdamW": optim.AdamW,
}

SCHEDULER_REGISTRY = {
    "CosineAnnealingLR":           lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "StepLR":                      lr_scheduler.StepLR,
    "ExponentialLR":               lr_scheduler.ExponentialLR,
    "ReduceLROnPlateau":           lr_scheduler.ReduceLROnPlateau,
}


def build_optimizer(params, phase_config: dict) -> torch.optim.Optimizer:
    name = phase_config.get("OPTIMIZER", "Adam")
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizador '{name}' no soportado. "
                         f"Disponibles: {list(OPTIMIZER_REGISTRY)}")

    kwargs = {"lr": float(phase_config["LR"])}
    if "WEIGHT_DECAY" in phase_config:
        kwargs["weight_decay"] = float(phase_config["WEIGHT_DECAY"])

    return OPTIMIZER_REGISTRY[name](params, **kwargs)


def build_scheduler(optimizer, phase_config: dict):
    name = phase_config.get("SCHEDULER", "CosineAnnealingLR")
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{name}' no soportado. "
                         f"Disponibles: {list(SCHEDULER_REGISTRY)}")

    raw_params = dict(phase_config.get("SCHEDULER_PARAMS", {}))

    # Sanitizar: yaml puede devolver strings para notación científica
    params = {}
    for k, v in raw_params.items():
        if isinstance(v, str):
            try:
                fv = float(v)
                params[k] = int(fv) if (k == "T_max" and fv.is_integer()) else fv
                continue
            except ValueError:
                pass
        params[k] = v

    return SCHEDULER_REGISTRY[name](optimizer, **params)
# ──────────────────────────────────────────────────────────────────────
# Traducción de claves UPPERCASE del config → vocabulario de cada modelo
# ──────────────────────────────────────────────────────────────────────

def _tst1_cfg(config: dict) -> dict:
    t = config["TST1"]
    return {
        "n_rois":          config["N_ROIS"],
        "emb_dim":         t["D_MODEL"],
        "n_layers":        t["NUM_ENCODER_LAYERS"],
        "n_heads":         t["N_HEADS"],
        "dim_feedforward": t["DIM_FEEDFORWARD"],
        "dropout":         t["ENC_DROP"],
        "max_seq_len":     config["MAX_SEQ_LEN"],
        "use_cls_token":   t["USE_CLS_TOKEN"],
    }


def _tst2_cfg(config: dict) -> dict:
    t = config["TST2"]
    return {
        "pcc_dim":         t["PCC_DIM"],
        "d_model":         t["D_MODEL"],
        "n_layers":        t["NUM_ENCODER_LAYERS"],
        "n_heads":         t["N_HEADS"],
        "dim_feedforward": t["DIM_FEEDFORWARD"],
        "dropout":         t["ENC_DROP"],
    }


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

def build_experiment(config: dict, fold_idx: int = 0) -> ExperimentContext:
    """
    Construye el ExperimentContext para la fase indicada en
    config["EXPERIMENT_TYPE"] ∈ {"pretrain_ts", "pretrain_fc", "finetune"}.

    Returns:
        ExperimentContext con (model, task, optimizer, scheduler,
                               train_loader, val_loader, device).
    """
    exp_type = config["EXPERIMENT_TYPE"]
    device = torch.device(
        config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    )
    num_workers = config.get("NUM_WORKERS", 0)
    seed = config.get("SEED", 42)

    # ─── Pretrain TST1 ────────────────────────────────────────────────
    if exp_type == "pretrain_ts":
        phase = config["PT_TST1"]

        model = create_transformer_ts(_tst1_cfg(config)).to(device)
        task = ReconstructionTask(device)

        train_ts, val_ts, _, _ = get_pretrain_loaders(
            config,
            batch_size=phase["BATCH_SIZE"],
            num_workers=num_workers,
            seed=seed,
        )

        optimizer = build_optimizer(model.parameters(), phase)
        scheduler = build_scheduler(optimizer, phase)

        return ExperimentContext(
            model=model, task=task,
            optimizer=optimizer, scheduler=scheduler,
            train_loader=train_ts, val_loader=val_ts,
            device=device,
        )

    # ─── Pretrain TST2 ────────────────────────────────────────────────
    if exp_type == "pretrain_fc":
        phase = config["PT_TST2"]

        model = create_transformer_fc(_tst2_cfg(config)).to(device)
        task = ReconstructionTask(device)

        _, _, train_fc, val_fc = get_pretrain_loaders(
            config,
            batch_size=phase["BATCH_SIZE"],
            num_workers=num_workers,
            seed=seed,
        )

        optimizer = build_optimizer(model.parameters(), phase)
        scheduler = build_scheduler(optimizer, phase)

        return ExperimentContext(
            model=model, task=task,
            optimizer=optimizer, scheduler=scheduler,
            train_loader=train_fc, val_loader=val_fc,
            device=device,
        )
        # ─── Contrastive (paper Sec. 3.3 + Sec. 4.3.2) ────────────────────
    if exp_type == "contrastive":
        phase = config["T_CONTRASTIVE"]

        model = create_dual_stream_model(
            n_rois=config["N_ROIS"],
            time_points=config["MAX_SEQ_LEN"],
            pcc_dim=config["TST2"]["PCC_DIM"],
            tst1_emb_dim=config["TST1"]["D_MODEL"],
            tst2_d_model=config["TST2"]["D_MODEL"],
            fusion_type=config["DUAL_STREAM"]["FUSION_TYPE"],
            num_classes=config["DUAL_STREAM"]["NUM_CLASSES"],
            dropout=config["DUAL_STREAM"]["CLASSIFIER_DROPOUT"],
            mlp_dims=config["DUAL_STREAM"].get("MLP_DIMS"),
        ).to(device)

        if config.get("CKPT_TST1"):
            model.load_pretrained_tst1(config["CKPT_TST1"], strict=False)
        if config.get("CKPT_TST2"):
            model.load_pretrained_tst2(config["CKPT_TST2"], strict=False)

        # Ambos encoders descongelados (unfreeze both, Sec. 4.3.2)
        model.unfreeze_encoders()

        task = ContrastiveTask(
            dim_ts=model.dim_ts,
            dim_fc=model.dim_fc,
            hidden_dim=phase["PROJ_HIDDEN_DIM"],
            output_dim=phase["PROJ_OUTPUT_DIM"],
            temperature=phase["TEMPERATURE"],
            device=device,
        ).to(device)

        # Optimizador ve modelo + cabezales de proyección
        params = list(model.parameters()) + list(task.contrastive_module.parameters())

        # Loaders: reusamos los de finetune (mismo split por fold)
        train_loader, val_loader, _, _ = get_finetune_loaders(
            config,
            batch_size=phase["BATCH_SIZE"],
            num_workers=num_workers,
            fold_idx=fold_idx,
            n_folds=config.get("N_FOLDS", 5),
            seed=seed,
            eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
        )

        optimizer = build_optimizer(params, phase)
        scheduler = build_scheduler(optimizer, phase)

        return ExperimentContext(
            model=model, task=task,
            optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=val_loader,
            device=device,
        )

    # ─── Finetune ─────────────────────────────────────────────────────
    if exp_type == "finetune":
        phase = config["FINETUNING"]
        ds = config["DUAL_STREAM"]
        fusion_hidden = config.get("FUSION", {}).get("ATTENTION_POOLING", {}).get("HIDDEN_DIM")

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
        ).to(device)

        # Cargar checkpoints pretrain si están disponibles
        if config.get("CKPT_TST1"):
            model.load_pretrained_tst1(config["CKPT_TST1"], strict=False)
        if config.get("CKPT_TST2"):
            model.load_pretrained_tst2(config["CKPT_TST2"], strict=False)

        task = ClassificationTask(device)

        train_loader, val_loader, _, _ = get_finetune_loaders(
            config,
            batch_size=phase["BATCH_SIZE"],
            num_workers=num_workers,
            fold_idx=fold_idx,
            n_folds=config.get("N_FOLDS", 5),
            seed=seed,
            eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
        )

        optimizer = build_optimizer(model.parameters(), phase)
        scheduler = build_scheduler(optimizer, phase)

        return ExperimentContext(
            model=model, task=task,
            optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=val_loader,
            device=device,
        )

    raise ValueError(f"EXPERIMENT_TYPE no soportado: {exp_type}")


# ──────────────────────────────────────────────────────────────────────
# Tests (generan .1D + CSV sintéticos en disco)
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import tempfile
    import shutil
    from pathlib import Path

    torch.manual_seed(0)

    # ─── Estructura sintética ─────────────────────────────────────────
    tmp = Path(tempfile.mkdtemp())
    interp_dir = tmp / "interp"
    interp_dir.mkdir()

    N, T, R = 30, 100, 200
    D = R * (R - 1) // 2
    sites = ["SITE_0", "SITE_1", "SITE_2"]

    rng = np.random.default_rng(0)
    rows = []
    for i in range(N):
        fid = f"S{1000 + i}"
        arr = rng.standard_normal((T, R)).astype(np.float32)
        np.savetxt(interp_dir / f"interp_{fid}_rois_cc200.1D", arr)
        rows.append({
            "FILE_ID": fid, "SUB_ID": i,
            "SITE_ID": sites[i % 3],
            "DX_GROUP": int(rng.integers(0, 2)),
        })

    csv_path = tmp / "meta.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ─── Config mínimo (sintético, vocab UPPERCASE) ───────────────────
    base_config = {
        "DEVICE": "cpu",
        "NUM_WORKERS": 0,
        "SEED": 0,
        "RAW_PATH": str(interp_dir),          # no se usa en los tests
        "INTERP_PATH": str(interp_dir),
        "CSV_PATH": str(csv_path),
        "ATLAS": "cc200",
        "PREFIX": "interp_",
        "N_ROIS": R,
        "MAX_SEQ_LEN": T,
        "LABEL_COL": "DX_GROUP",
        "EVAL_PROTOCOL": "kfold",
        "N_FOLDS": 3,
        "TST1": {
            "D_MODEL": 64, "DIM_FEEDFORWARD": 128,
            "NUM_ENCODER_LAYERS": 2, "N_HEADS": 4,
            "ENC_DROP": 0.1, "USE_CLS_TOKEN": True,
        },
        "TST2": {
            "PCC_DIM": D, "D_MODEL": 64,
            "N_HEADS": 4, "NUM_ENCODER_LAYERS": 2,
            "DIM_FEEDFORWARD": 128, "ENC_DROP": 0.1,
        },
        "DUAL_STREAM": {
            "FUSION_TYPE": "attention_pooling",
            "NUM_CLASSES": 2,
            "CLASSIFIER_DROPOUT": 0.3,
            "MLP_DIMS": [256, 64, 2],
        },
        "FUSION": {"ATTENTION_POOLING": {"HIDDEN_DIM": 128}},
        "PT_TST1": {
            "N_EPOCHS": 2, "BATCH_SIZE": 4, "LR": 1e-4,
            "WEIGHT_DECAY": 1e-4, "MASK_RATIO": None,
            "OPTIMIZER": "Adam", "SCHEDULER": "CosineAnnealingLR",
            "SCHEDULER_PARAMS": {"T_max": 2, "eta_min": 1e-6},
        },
        "PT_TST2": {
            "N_EPOCHS": 2, "BATCH_SIZE": 4, "LR": 1e-4,
            "WEIGHT_DECAY": 1e-4, "MASK_RATIO": 0.15,
            "OPTIMIZER": "Adam", "SCHEDULER": "CosineAnnealingLR",
            "SCHEDULER_PARAMS": {"T_max": 2, "eta_min": 1e-6},
        },
        "FINETUNING": {
            "N_EPOCHS": 2, "BATCH_SIZE": 4, "LR": 5e-5,
            "WEIGHT_DECAY": 1e-4, "OPTIMIZER": "Adam",
            "SCHEDULER": "CosineAnnealingLR",
            "SCHEDULER_PARAMS": {"T_max": 2, "eta_min": 5e-7},
        },
    }

    # ─── TEST 1: pretrain_ts ──────────────────────────────────────────
    print("── TEST 1: build_experiment (pretrain_ts) ───────────────────")
    cfg = {**base_config, "EXPERIMENT_TYPE": "pretrain_ts"}
    exp = build_experiment(cfg, fold_idx=0)

    assert exp.model.__class__.__name__ == "TransformerTS"
    assert exp.task.__class__.__name__ == "ReconstructionTask"
    batch = next(iter(exp.train_loader))
    assert batch.shape == (4, T, R), batch.shape
    print(f"  ✓ model={exp.model.__class__.__name__}  batch {tuple(batch.shape)}")

    mask = torch.zeros_like(batch, dtype=torch.bool)
    mask[:, ::4, :] = True
    m = batch.clone(); m[mask] = 0.0
    loss = exp.task.execution_step(exp.model, m, mask, batch)
    print(f"  ✓ loss={loss.item():.4f}\n")

    # ─── TEST 2: pretrain_fc ──────────────────────────────────────────
    print("── TEST 2: build_experiment (pretrain_fc) ───────────────────")
    cfg = {**base_config, "EXPERIMENT_TYPE": "pretrain_fc"}
    exp = build_experiment(cfg, fold_idx=0)

    assert exp.model.__class__.__name__ == "TransformerFC"
    batch = next(iter(exp.train_loader))
    assert batch.shape == (4, D), batch.shape
    print(f"  ✓ model={exp.model.__class__.__name__}  batch {tuple(batch.shape)}")

    mask = torch.rand_like(batch) > 0.85
    m = batch.clone(); m[mask] = 0.0
    loss = exp.task.execution_step(exp.model, m, mask, batch)
    print(f"  ✓ loss={loss.item():.4f}\n")

    # ─── TEST 3: finetune ─────────────────────────────────────────────
    print("── TEST 3: build_experiment (finetune) ──────────────────────")
    cfg = {**base_config, "EXPERIMENT_TYPE": "finetune"}
    exp = build_experiment(cfg, fold_idx=0)

    assert exp.model.__class__.__name__ == "DualStreamModel"
    assert exp.task.__class__.__name__ == "ClassificationTask"
    batch = next(iter(exp.train_loader))
    assert set(batch.keys()) == {"timeseries", "pcc_vector", "label"}
    assert batch["timeseries"].shape == (4, T, R)
    assert batch["pcc_vector"].shape == (4, D)
    print(f"  ✓ model={exp.model.__class__.__name__}  batch keys={list(batch.keys())}")

    loss = exp.task.execution_step(
        exp.model, batch["timeseries"], batch["pcc_vector"], batch["label"]
    )
    print(f"  ✓ loss={loss.item():.4f}\n")

    # ─── TEST 4: MLP_DIMS se aplica ───────────────────────────────────
    print("── TEST 4: MLP_DIMS del config se aplica al classifier ──────")
    cfg = {**base_config, "EXPERIMENT_TYPE": "finetune"}
    exp = build_experiment(cfg, fold_idx=0)
    # Última capa debe salir a NUM_CLASSES=2
    assert exp.model.classifier[-1].out_features == 2
    # Debe haber 3 capas lineales (256, 64, 2)
    linear_layers = [m for m in exp.model.classifier if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3, f"capas={len(linear_layers)}"
    print(f"  ✓ 3 capas lineales, salida={linear_layers[-1].out_features}\n")

    # ─── TEST 5: EXPERIMENT_TYPE inválido ─────────────────────────────
    print("── TEST 5: EXPERIMENT_TYPE inválido ─────────────────────────")
    try:
        build_experiment({**base_config, "EXPERIMENT_TYPE": "no_existe"}, fold_idx=0)
        raise AssertionError("Debió lanzar ValueError")
    except ValueError as e:
        print(f"  ✓ ValueError: {str(e)[:60]}\n")

    # ─── TEST 6: optimizer actualiza pesos ────────────────────────────
    print("── TEST 6: un paso de optimizer actualiza pesos ─────────────")
    cfg = {**base_config, "EXPERIMENT_TYPE": "finetune"}
    exp = build_experiment(cfg, fold_idx=0)
    batch = next(iter(exp.train_loader))
    w_before = exp.model.classifier[-1].weight.clone()

    exp.optimizer.zero_grad()
    loss = exp.task.execution_step(
        exp.model, batch["timeseries"], batch["pcc_vector"], batch["label"]
    )
    loss.backward()
    exp.optimizer.step()
    w_after = exp.model.classifier[-1].weight
    assert not torch.allclose(w_before, w_after), "El optimizer no actualizó pesos"
    print(f"  ✓ pesos del classifier cambiaron\n")

    shutil.rmtree(tmp)
    print("✅ Todos los tests de setup.py pasaron.")