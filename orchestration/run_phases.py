from tqdm import tqdm
from training.setup import build_experiment
from pretraining.pretrain_ts import train_one_epoch, validate
from training.callbacks import EarlyStopping
import torch.nn as nn


def run_pretrain_ts(config: dict, df_train, df_val, df_test):
    config["EXPERIMENT_TYPE"] = "pretrain_ts"
    exp_ts = build_experiment(config, df_train, df_val, df_test) # diccionario que devuelve : "model", "task", "optimizer"
                                                                    # "scheduler", "train_loader","val_loader"

    model = exp_ts['model']
    epochs = config["PT_TST1"]["N_EPOCHS"]
    early_stopping = EarlyStopping(model,config)

    train_losses = []
    val_losses = []

    

    
    with tqdm(range(epochs), unit="epoch") as tepoch:
        
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            train_running_loss = 0.0
            val_running_loss = 0.0
            model.train()
            
            # Train
            train_running_loss+= train_one_epoch(
                **exp_ts
            )
            model.eval()
            # Validate
            val_running_loss+= validate(**exp_ts)

            # Update learning rate
            exp_ts['scheduler'].step()

            avg_train_loss = train_running_loss/ len(exp_ts["train_loader"])
            avg_val_loss = val_running_loss / len(exp_ts["val_loader"])
            
            # Record losses
            train_losses.append(train_running_loss)
            val_losses.append(val_running_loss)

            best: nn.Module | None = early_stopping(model,avg_val_loss)

            if best:
                    model.load_state_dict(early_stopping.best_model.state_dict())
                    print(f"Early stopping pretraining timeseries, Epoca: {epoch + 1}. Mejor loss: {early_stopping.min_val_loss:4f}")
                    break


            tepoch.set_postfix(v_loss=f"{avg_val_loss:.4f}")
            
            
            
            
            print(f"Train Loss: {train_running_loss:.4f}, Val Loss: {val_running_loss:.4f}")
            # print(f"LR: {current_lr:.6f}")
            