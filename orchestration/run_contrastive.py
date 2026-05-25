from pathlib import Path
import torch
from tqdm import tqdm
from training.setup import build_experiment
from training.train_contrastive import train_one_epoch
from utils.checkpoint import get_checkpoint_path
from training.context import ExperimentContext


def run_contrastive(
    config: dict, df_train, df_val, df_test, fold, chkpt_ts, chkpt_fc
):
    """
    Fase de aprendizaje contrastivo (sin validación, como en el paper original).
    Guarda el checkpoint al final de todas las épocas.
    """
    config["EXPERIMENT_TYPE"] = "contrastive"

    exp: ExperimentContext = build_experiment(
        config, df_train, df_val, df_test,
        chkpt_ts=chkpt_ts, chkpt_fc=chkpt_fc
    )

    epochs = config["T_CONTRASTIVE"]["N_EPOCHS"]
    losses = []

    with tqdm(range(epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Contrastive Epoch {epoch+1}")

            train_loss = train_one_epoch(exp)
            avg_loss = train_loss / len(exp.train_loader)

            # Scheduler step (si lo hubiera; actualmente no se usa en contrastive)
            # exp.scheduler.step()

            losses.append(train_loss)
            tepoch.set_postfix(loss=f"{avg_loss:.4f}")

    save_path = get_checkpoint_path(config, "CONT", fold)
    torch.save(exp.model.state_dict(), save_path)

    return losses