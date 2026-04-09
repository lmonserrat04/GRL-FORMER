import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from pathlib import Path
import torch
from training.train import Trainer
from test.test import test as run_test
from utils.logger import Logger
from setup import build_experiment
from torch.utils.data import  DataLoader


class CrossValidator:
    def __init__(self, config: dict):
        self.config = config
        self.skf = StratifiedKFold(
            n_splits=config["N_SPLITS_CV"],
            shuffle=True,
            random_state=config["SEED"]
        )

    def run(self):
        df = pd.read_csv(Path(self.config["CSV_PATH"]).resolve())

        vals = []
        accs = []
        val_losses = []

        table_cols = ['Fold', 'Accuracy', 'Precision', 'Recall',
                      'F1', 'AUC', 'AP', 'FPR', 'FNR', 'TPR', 'TNR']

        df["STRATIFY"] = df['SITE_ID'].astype(str) + '_' + df['DX_GROUP'].astype(str)

        for fold, (train_idx, test_idx) in enumerate(self.skf.split(df, df["STRATIFY"])):

            print(f"\n{'='*80}")
            print(f"  FOLD {fold + 1} / {self.config['N_SPLITS_CV']}")
            print(f"{'='*80}")

            df_train, df_val = train_test_split(
                df.iloc[train_idx],
                test_size=0.20,
                stratify=df.iloc[train_idx]["STRATIFY"],
                random_state=self.config["SEED"]
            )

            df_test = df.iloc[test_idx]

            
            model, optimizer, criterion, scheduler, \
            train_loader, val_loader, test_loader = build_experiment(self.config, df_train, df_val, df_test)



            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=self.config,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                fold=fold
            )

            best_val_loss = trainer.fit()
            val_losses.append(best_val_loss)

            run_test(model, fold, accs, vals, self.config, test_loader, criterion) # Pudieras retornar vals y acc en vez de modificar estos
                                                                                   # objetos mutables inplace

            torch.save(model.state_dict(), Path(self.config["CHECKPOINTS_PATH"]).resolve() / f"best_model_fold{fold+1}.pth")

        
        results = pd.DataFrame(vals, columns=table_cols)
        results.to_csv(self.config["RESULTS_PATH"], index=False)

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        mean_val_loss = np.mean(val_losses)

        print(f"\n{'='*80}")
        print(results.to_string(index=False))
        print(f"{'='*80}")
        print(f"Mean accuracy : {mean_acc:.4f}")
        print(f"Std  accuracy : {std_acc:.4f}")

       
        summary_log_path = Path(self.config["LOGS_PATH"]) / "summary.txt"
        summary_logger   = Logger(summary_log_path, mode='w')
        summary_logger.log_summary(
            pt_val_loss=mean_val_loss,
            t_val_loss=mean_val_loss,
            mean_acc=mean_acc,
            std_acc=std_acc
        )