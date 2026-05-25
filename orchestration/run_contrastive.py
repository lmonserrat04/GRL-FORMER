from pathlib import Path
import torch
#from training.callbacks import EarlyStopping
from training.setup import build_experiment
from training.tasks.contrastive import ContrastiveTask
from tqdm import tqdm
from utils.checkpoint import get_checkpoint_path
from training.context import ExperimentContext


def run_contrastive(
    config: dict, df_train, df_val, df_test, fold, chkpt_ts, chkpt_fc
):
    """
    Fase de aprendizaje contrastivo.
    Utiliza el modelo dual‑stream y proyecta ambas modalidades.
    Guarda el checkpoint del modelo contrastivo al finalizar.
    """
    config["EXPERIMENT_TYPE"] = "contrastive"

    # build_experiment ahora devuelve un ExperimentContext
    exp: ExperimentContext = build_experiment(
        config, df_train, df_val, df_test,
        chkpt_ts=chkpt_ts, chkpt_fc=chkpt_fc
    )

    model                        = exp.model
    task: ContrastiveTask        = exp.task          # ContrastiveTask
    optimizer                    = exp.optimizer
    train_loader                 = exp.train_loader
    device                       = exp.device

    epochs = config["T_CONTRASTIVE"]["N_EPOCHS"]

    model.train()
    task.contrastive_module.train()
    losses = []

    with tqdm(range(epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Contrastive Epoch {epoch+1}")
            epoch_loss = 0.0
            num_batches = 0

            for batch, _ in train_loader:
                timeseries = batch['timeseries'].to(device)
                pcc_vector = batch['pcc_vector'].to(device)

                optimizer.zero_grad()
                loss = task.execution_step(model, timeseries, pcc_vector)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            tepoch.set_postfix(contr_loss=f"{avg_loss:.4f}")
            losses.append(avg_loss)

        save_path = get_checkpoint_path(config, "CONT", fold)
        torch.save(model.state_dict(), save_path)

    return losses