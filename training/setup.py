import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler   # al inicio del archivo
from data.loaders.dataloader import get_dataloader
from data.preprocessing.harmonization import GlobalNormalizer, ResidualHarmonizer
from models.transformer_ts import build_model_ts
from models.transformer_fc import build_model_fc
from models.dual_stream import create_dual_stream_model
from training.tasks.classification import ClassificationTask
from training.tasks.reconstruction import ReconstructionTask
from training.tasks.contrastive import ContrastiveTask, ContrastiveWrapper

def build_optimizer(params, config: dict) -> torch.optim.Optimizer:
    

    return torch.optim.AdamW(
        params,
        lr=float(config["LR"]),
        weight_decay=float(config["WEIGHT_DECAY"])
    )

def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config.get("T_0", 10))
    )


def build_criterion(config: dict) -> nn.Module:
    return nn.CrossEntropyLoss(
        label_smoothing=float(config["LABEL_SMOOTHING"])
    )

def compute_loss(outputs, labels, criterion, config):
    return criterion(outputs, labels)


def build_dataloaders(config, df_train, df_val, df_test, harmonizer, normalizer):
    train_loader = get_dataloader(config, df_train, split='train', 
                                  normalizer=normalizer, harmonizer=harmonizer)
    val_loader   = get_dataloader(config, df_val,   split='val',   
                                  normalizer=normalizer, harmonizer=harmonizer)
    test_loader  = get_dataloader(config, df_test,  split='test',  
                                  normalizer=normalizer, harmonizer=harmonizer)
    return train_loader, val_loader, test_loader



def build_experiment(config, df_train, df_val, df_test,
                     chkpt_ts=None,    # pretrain TS  → contrastive
                     chkpt_fc=None,    # pretrain FC  → contrastive
                     chkpt_cont=None):
    """
    Instancia solo los componentes necesarios según el tipo de experimento.
    """
    device = config["DEVICE"]
    exp_type = config["EXPERIMENT_TYPE"] # Ej: 'pretrain_ts', 'pretrain_fc', 'contrastive', 'finetune'

    # 1. Dataloaders (comunes o específicos según config)
    train_loader, val_loader, test_loader = build_dataloaders(config, df_train, df_val, df_test)


    # 2. Selección de Arquitectura y Tarea
    phase_config = {}

    if exp_type == "pretrain_ts":
        
        model = build_model_ts(config).to(device)
        task = ReconstructionTask(device)
        params = model.parameters()
        phase_config = config["PT_TST1"]
       

    elif exp_type == "pretrain_fc":
        model = build_model_fc(config).to(device)
        task = ReconstructionTask(device)
        params = model.parameters()
        phase_config = config["PT_TST2"]

    elif exp_type == "contrastive":
        model = create_dual_stream_model(
            config,
            name_chkpt_pt_ts=chkpt_ts,   # llega como parámetro de build_experiment
            name_chkpt_pt_fc=chkpt_fc,   # llega como parámetro de build_experiment
        ).to(device)


        task = ContrastiveTask(
            dim_ts=model.dim_ts,            # ya expuesto en DualStreamModel
            dim_fc=model.dim_fc,
            proj_hidden_dim=config["T_CONTRASTIVE"]["PROJ_HIDDEN_DIM"],
            proj_output_dim=config["T_CONTRASTIVE"]["PROJ_OUTPUT_DIM"],
            temperature=config["T_CONTRASTIVE"]["TEMPERATURE"],
            device=device
)

        # El optimizador necesita parámetros del modelo Y de los Projection Heads[cite: 3]
        params = [
            {'params': model.parameters()},
            {'params': task.contrastive_module.parameters()}
        ]
        phase_config = config["T_CONTRASTIVE"]


    elif exp_type == "finetune":
        model = create_dual_stream_model(config,
                                         name_chkpt_cont=chkpt_cont,
                                        ).to(device)
        if chkpt_cont:
            model.load_state_dict(torch.load(chkpt_cont))
        task = ClassificationTask(device)
        params = model.parameters()
        phase_config = config["FINETUNING"]

    else:
        raise ValueError(f"Unknown EXPERIMENT_TYPE: {exp_type}")

    # 3. Componentes de entrenamiento finales
    optimizer = build_optimizer(params, phase_config)
    scheduler = build_scheduler(optimizer, phase_config)

    

    return {
        "model": model,
        "task": task,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "device": device
    }