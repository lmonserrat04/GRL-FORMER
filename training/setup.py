import torch
import torch.nn as nn
from data.loaders.dataloader1 import build_dataloaders  
from models.transformer_ts import build_model_ts
from models.transformer_fc import build_model_fc
from models.dual_stream import create_dual_stream_model
from training.tasks.classification import ClassificationTask
from training.tasks.reconstruction import ReconstructionTask
from training.tasks.contrastive import ContrastiveTask
from training.context import ExperimentContext
import torch.optim as optim

# Registry: nombre → clase de optimizador
OPTIMIZER_REGISTRY = {
    "Adam":   optim.Adam,
    "AdamW":  optim.AdamW,
}

def build_optimizer(params, config: dict) -> torch.optim.Optimizer:
    """
    params : iterable de parámetros o lista de dicts (grupos)
    config : diccionario de fase (ej. config["PT_TST1"] o config["T_CONTRASTIVE"])
             Debe contener "OPTIMIZER" (opcional, por defecto "AdamW"),
             "LR", y opcionalmente "WEIGHT_DECAY", "MOMENTUM", etc.
    """
    optimizer_name = config.get("OPTIMIZER", "AdamW")
    optimizer_cls = OPTIMIZER_REGISTRY.get(optimizer_name)
    if optimizer_cls is None:
        raise ValueError(f"Optimizador '{optimizer_name}' no soportado. "
                         f"Disponibles: {list(OPTIMIZER_REGISTRY.keys())}")

    # Preparamos los kwargs genéricos
    kwargs = {"lr": float(config["LR"])}
    if "WEIGHT_DECAY" in config:
        kwargs["weight_decay"] = float(config["WEIGHT_DECAY"])
    
    # Si en el futuro necesitas betas, eps... añádelos aquí

    return optimizer_cls(params, **kwargs) # **kwargs -> Keyword Arguments, desempaqueta el dict 


import torch.optim.lr_scheduler as lr_scheduler

# Registry: nombre → clase de scheduler
SCHEDULER_REGISTRY = {
    "CosineAnnealingLR":          lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "StepLR":                     lr_scheduler.StepLR,
    "ExponentialLR":              lr_scheduler.ExponentialLR,
    "ReduceLROnPlateau":          lr_scheduler.ReduceLROnPlateau,  # requiere métrica aparte
}

def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Construye el scheduler según la clave 'SCHEDULER' y 'SCHEDULER_PARAMS'.
    Si no se especifica, por defecto CosineAnnealingWarmRestarts con T_0 = T_0 de config, o 10.
    """
    scheduler_name = config.get("SCHEDULER", "CosineAnnealingWarmRestarts")
    scheduler_cls = SCHEDULER_REGISTRY.get(scheduler_name)
    if scheduler_cls is None:
        raise ValueError(f"Scheduler '{scheduler_name}' no soportado. "
                         f"Disponibles: {list(SCHEDULER_REGISTRY.keys())}")

    # Parámetros específicos del scheduler (todos opcionales, el scheduler usará sus defaults)
    scheduler_params = config.get("SCHEDULER_PARAMS", {})

    # Retrocompatibilidad: si no hay SCHEDULER_PARAMS y no se definió SCHEDULER, usar T_0 antiguo
    if not scheduler_params and scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler_params = {"T_0": int(config.get("T_0", 10))}

    return scheduler_cls(optimizer, **scheduler_params)




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

    

    return ExperimentContext(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )