"""Factory — solo pretrain_ts, pretrain_fc, finetune. Contrastive vive en train_contrastive.py."""
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from data.loaders.dataloader import get_pretrain_loaders, get_finetune_loaders
from models.transformer_ts import create_transformer_ts
from models.transformer_fc import create_transformer_fc
from models.dual_stream import create_dual_stream_model

from training.context import ExperimentContext
from training.tasks.reconstruction import ReconstructionTask
from training.tasks.classification import ClassificationTask


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


def build_experiment(config, fold_idx=0):
    exp_type = config["EXPERIMENT_TYPE"]
    device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    nw = config.get("NUM_WORKERS", 0)
    seed = config.get("SEED", 42)

    if exp_type == "pretrain_ts":
        phase = config["PT_TST1"]
        model = create_transformer_ts(_tst1_cfg(config)).to(device)
        task = ReconstructionTask(device)
        tr, va, _, _ = get_pretrain_loaders(config, batch_size=phase["BATCH_SIZE"], num_workers=nw, seed=seed)
        opt = build_optimizer(model.parameters(), phase)
        sch = build_scheduler(opt, phase)
        return ExperimentContext(model=model, task=task, optimizer=opt, scheduler=sch,
                                  train_loader=tr, val_loader=va, device=device)

    if exp_type == "pretrain_fc":
        phase = config["PT_TST2"]
        model = create_transformer_fc(_tst2_cfg(config)).to(device)
        task = ReconstructionTask(device)
        _, _, tr, va = get_pretrain_loaders(config, batch_size=phase["BATCH_SIZE"], num_workers=nw, seed=seed)
        opt = build_optimizer(model.parameters(), phase)
        sch = build_scheduler(opt, phase)
        return ExperimentContext(model=model, task=task, optimizer=opt, scheduler=sch,
                                  train_loader=tr, val_loader=va, device=device)

    if exp_type == "finetune":
        phase = config["FINETUNING"]; ds = config["DUAL_STREAM"]
        fh = config.get("FUSION", {}).get("ATTENTION_POOLING", {}).get("HIDDEN_DIM")
        mlp_classiffier = config.get("MLP_HEAD")

        model = create_dual_stream_model(
            tst1_config=_tst1_cfg(config),
            tst2_config= _tst2_cfg(config),
            fusion_type = ds["FUSION_TYPE"],
            fusion_hidden_dim = fh,
            num_classes = ds["NUM_CLASSES"],
            dropout = mlp_classiffier["DROPOUT"],
            mlp_dims = list(mlp_classiffier["MLP_DIMS"]),
            proj_head_1=None,
            proj_head_2=None,
        ).to(device)


        if config.get("CKPT_TST1"): model.load_pretrained_tst1(config["CKPT_TST1"], strict=False)
        if config.get("CKPT_TST2"): model.load_pretrained_tst2(config["CKPT_TST2"], strict=False)
        task = ClassificationTask(device)

        tr, va, _, _ = get_finetune_loaders(
            config, batch_size=phase["BATCH_SIZE"], num_workers=nw,
            fold_idx=fold_idx, n_folds=config.get("N_FOLDS", 5), seed=seed,
            eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
        )

        opt = build_optimizer(model.parameters(), phase)
        sch = build_scheduler(opt, phase)
        return ExperimentContext(model=model, task=task, optimizer=opt, scheduler=sch,
                                  train_loader=tr, val_loader=va, device=device)

    raise ValueError(f"EXPERIMENT_TYPE no soportado: {exp_type}")
