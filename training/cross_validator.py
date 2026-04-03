# cross_validator.py
from sklearn.model_selection import StratifiedKFold
from training.train import Trainer

class CrossValidator:
    def __init__(self, config, n_splits=5):
        self.config = config
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                   random_state=config["SEED"])

    def run(self, df, model_fn):
        for fold, (train_idx, val_idx) in enumerate(
            self.skf.split(df, df["STRATIFY"])
        ):
            df_train = df.iloc[train_idx]
            df_val   = df.iloc[val_idx]

            model   = model_fn()        # modelo fresco cada fold
            trainer = Trainer(model, self.config)
            trainer.fit(df_train, df_val)