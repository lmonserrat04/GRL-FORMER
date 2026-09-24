"""Factory — solo pretrain_ts, pretrain_fc, finetune. Contrastive vive en train_contrastive.py."""
from pathlib import Path
import sys
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from data.loaders.dataloader import get_pretrain_loaders, get_finetune_loaders
from models.transformer_ts import create_transformer_ts
from models.transformer_fc import create_transformer_fc
from models.dual_stream import create_dual_stream_model

from .context import ExperimentContext
from .tasks.reconstruction import ReconstructionTask
from .tasks.classification import ClassificationTask
from .tasks.contrastive import ProjectionHead
from .train_contrastive import build_contrastive_components


OPTIMIZER_REGISTRY = {"Adam": optim.Adam, "AdamW": optim.AdamW}
SCHEDULER_REGISTRY = {
    "CosineAnnealingLR":           lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "StepLR":                      lr_scheduler.StepLR,
    "ExponentialLR":               lr_scheduler.ExponentialLR,
    "ReduceLROnPlateau":           lr_scheduler.ReduceLROnPlateau,
}


def build_optimizer(params, phase_config):
    name = phase_config.get("OPTIMIZER", "Adam")
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizador '{name}' no soportado.")
    kwargs = {"lr": float(phase_config["LR"])}
    if "WEIGHT_DECAY" in phase_config:
        kwargs["weight_decay"] = float(phase_config["WEIGHT_DECAY"])
    return OPTIMIZER_REGISTRY[name](params, **kwargs)


def build_scheduler(optimizer, phase_config):
    name = phase_config.get("SCHEDULER", "CosineAnnealingLR")
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{name}' no soportado.")
    raw = dict(phase_config.get("SCHEDULER_PARAMS", {}))
    params = {}
    for k, v in raw.items():
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
# Traductores config → dict de modelo
# ──────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────
# Carga de projection heads (contrastive → finetune)
# ──────────────────────────────────────────────────────────────────────

def _load_projection_heads(ckpt_path, config, device):
    """Carga proj heads del checkpoint contrastive. Las devuelve congeladas."""
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"No existe CKPT_CONTRASTIVE={ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    phase = config["T_CONTRASTIVE"]

    p1 = ProjectionHead(config["TST1"]["D_MODEL"],
                        phase["PROJ_HIDDEN_DIM"], phase["PROJ_OUTPUT_DIM"]).to(device)
    p2 = ProjectionHead(config["TST2"]["D_MODEL"],
                        phase["PROJ_HIDDEN_DIM"], phase["PROJ_OUTPUT_DIM"]).to(device)

    p1.load_state_dict(ckpt["proj_head_1_state_dict"])
    p2.load_state_dict(ckpt["proj_head_2_state_dict"])

    for p in p1.parameters(): p.requires_grad = False
    for p in p2.parameters(): p.requires_grad = False
    return p1, p2


def build_experiment(config, fold_idx=0, ckpt_contrastive=None):
    exp_type = config["EXPERIMENT_TYPE"]
    device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    nw = config.get("NUM_WORKERS", 0)
    seed = config.get("SEED", 42)

    # ─── Pretrain TST1 ────────────────────────────────────────────────

    if exp_type == "pretrain_ts":
        phase = config["PT_TST1"]
        model = create_transformer_ts(_tst1_cfg(config)).to(device)
        task = ReconstructionTask(device)
        train_loader, val_loader, _, _ = get_pretrain_loaders(config, batch_size=phase["BATCH_SIZE"], num_workers=nw, seed=seed)
        opt = build_optimizer(model.parameters(), phase)
        sch = build_scheduler(opt, phase)
        return ExperimentContext(model=model, task=task, optimizer=opt, scheduler=sch,
                                  train_loader=train_loader, val_loader=val_loader, device=device)

     # ─── Pretrain TST2 ────────────────────────────────────────────────

    if exp_type == "pretrain_fc":
        phase = config["PT_TST2"]
        model = create_transformer_fc(_tst2_cfg(config)).to(device)
        task = ReconstructionTask(device)
        _, _, train_loader, val_loader = get_pretrain_loaders(config, batch_size=phase["BATCH_SIZE"], num_workers=nw, seed=seed)
        opt = build_optimizer(model.parameters(), phase)
        sch = build_scheduler(opt, phase)
        return ExperimentContext(model=model, task=task, optimizer=opt, scheduler=sch,
                                  train_loader=train_loader, val_loader=val_loader, device=device)


    # ─── Finetune ─────────────────────────────────────────────────────

    if exp_type == "finetune":
        phase = config["FINETUNING"]; 
        ds = config["DUAL_STREAM"]
        fh = config.get("FUSION", {}).get("ATTENTION_POOLING", {}).get("HIDDEN_DIM")
        mlp_classiffier = config.get("MLP_HEAD")

        # Projections congeladas (si se pasa ckpt contrastive)
        if ckpt_contrastive is not None:
            p1, p2 = _load_projection_heads(ckpt_contrastive, config, device)
        else:
            p1, p2 = None, None

        model = create_dual_stream_model(
            tst1_config=_tst1_cfg(config),
            tst2_config= _tst2_cfg(config),
            fusion_type = ds["FUSION_TYPE"],
            fusion_hidden_dim = fh,
            num_classes = ds["NUM_CLASSES"],
            dropout = mlp_classiffier["DROPOUT"],
            mlp_dims = list(mlp_classiffier["MLP_DIMS"]),
            proj_head_1=p1,
            proj_head_2=p2,
        ).to(device)


        if config.get("CKPT_TST1"):
            model.load_pretrained_tst1(config["CKPT_TST1"], strict=False)
        if config.get("CKPT_TST2"):
            model.load_pretrained_tst2(config["CKPT_TST2"], strict=False)

        # Encoders entrenables, proj heads congeladas
        model.unfreeze_encoders()
        if p1 is not None:
            for p in model.proj_head_1.parameters(): p.requires_grad = False
        if p2 is not None:
            for p in model.proj_head_2.parameters(): p.requires_grad = False

        task = ClassificationTask(device)

        train_loader, val_loader, _, _ = get_finetune_loaders(
            config, batch_size=phase["BATCH_SIZE"], num_workers=nw,
            fold_idx=fold_idx, n_folds=config.get("N_FOLDS", 5), seed=seed,
            eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
        )


        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = build_optimizer(model.parameters(), phase)
        sch = build_scheduler(opt, phase)
        return ExperimentContext(model=model, task=task, optimizer=opt, scheduler=sch,
                                  train_loader=train_loader, val_loader=val_loader, device=device)

    raise ValueError(f"EXPERIMENT_TYPE no soportado: {exp_type}")
