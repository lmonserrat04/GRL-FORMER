
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from training.train import Trainer
from model.models import build_model
import pandas as pd
from data.loaders.dataloader import get_dataloader
from setup import *

class CrossValidator:
    def __init__(self, config: dict):
        self.config = config
        self.skf = StratifiedKFold(n_splits=config["N_SPLITS_CV"], shuffle=True, 
                                   random_state=config["SEED"])

    def run(self, df: pd.DataFrame):

        df["STRATIFY"] = df['SITE_ID'].astype(str) + '_' + df['DX_GROUP'].astype(str)
        
        for fold, (train_idx, test_idx) in enumerate(
            self.skf.split(df, df["STRATIFY"])
        ):  
            df_train, df_val = train_test_split(
                df.iloc[train_idx],
                test_size=0.20,
                stratify=df.iloc[train_idx]["STRATIFY"],
                random_state=self.config["SEED"]
            )

            df_test = df.iloc[test_idx]

            train_loader = get_dataloader(self.config, df_train, split='train')
            val_loader   = get_dataloader(self.config, df_val,   split='val')
            test_loader  = get_dataloader(self.config, df_test,  split='test')

            model     = build_model(self.config)
            optimizer = build_optimizer(model, self.config)
            criterion = build_criterion(self.config)
            scheduler = build_scheduler(optimizer, self.config)

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=self.config,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler
            )
            trainer.fit()