from tqdm import tqdm
from training.setup import build_experiment
from pretraining.pretrain_fc import train_one_epoch, validate
from training.callbacks import EarlyStopping
import torch.nn as nn
from utils.checkpoint import get_checkpoint_path
from training.context import ExperimentContext
import torch


def run_pretrain_fc(config: dict, df_train, df_val, df_test, fold):
    config["EXPERIMENT_TYPE"] = "pretrain_fc"
    exp = build_experiment(config, df_train, df_val, df_test)  # ExperimentContext

    model = exp.model
    epochs = config["PT_TST2"]["N_EPOCHS"]
    early_stopping = EarlyStopping(model, config)

    train_losses = []
    val_losses = []

    with tqdm(range(epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            train_running_loss = 0.0
            val_running_loss = 0.0
            model.train()

            # Train
            train_running_loss += train_one_epoch(
                exp,
                mask_ratio=config["PT_TST2"]["MASK_RATIO"]
            )

            model.eval()
            # Validate
            val_running_loss += validate(
                exp,
                mask_ratio=config["PT_TST2"]["MASK_RATIO"]
            )

            # Update learning rate
            exp.scheduler.step()

            avg_train_loss = train_running_loss / len(exp.train_loader)
            avg_val_loss = val_running_loss / len(exp.val_loader)

            train_losses.append(train_running_loss)
            val_losses.append(val_running_loss)

            best: nn.Module | None = early_stopping(model, avg_val_loss)

            if best:
                model.load_state_dict(early_stopping.best_model.state_dict())
                print(f"Early stopping pretraining pcc, Epoca: {epoch + 1}. Mejor loss: {early_stopping.min_val_loss:4f}")
                break

            tepoch.set_postfix(v_loss=f"{avg_val_loss:.4f}")

            print(f"Train Loss: {train_running_loss:.4f}, Val Loss: {val_running_loss:.4f}")

        save_path = get_checkpoint_path(config, "PT_FC", fold)
        torch.save(model.state_dict(), save_path)