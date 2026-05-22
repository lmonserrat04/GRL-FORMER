import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
import torch.nn.functional as F
from utils.logger import *
from pathlib import Path


def test(model, fold, accs: list, vals: list, config: dict, test_loader: DataLoader, criterion):
    model.eval()
    y_true, y_preds, y_probs = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            # batch es un diccionario: {'timeseries': Tensor, 'pcc_vector': Tensor, 'label': Tensor}
            ts = batch['timeseries'].to(config["DEVICE"])
            pcc = batch['pcc_vector'].to(config["DEVICE"])
            labels = batch['label'].to(config["DEVICE"])

            # El modelo dual stream espera dos entradas
            outputs = model(ts, pcc)
            loss = criterion(outputs, labels)

            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            y_true.extend(labels.detach().cpu().numpy())
            y_preds.extend(np.argmax(probs, axis=1))
            y_probs.extend(probs[:, 1])

        # Cálculo de métricas
        acc = accuracy_score(y_true, y_preds)
        accs.append(acc)
        cm = confusion_matrix(y_true, y_preds)

        # Manejo de divisiones por cero en caso de clases ausentes
        if cm.shape == (2, 2):
            precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0.0
            recall    = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
            fpr       = cm[0, 1] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) > 0 else 0.0
            fnr       = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0
            tpr       = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
            tnr       = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
        else:
            precision = recall = fpr = fnr = tpr = tnr = 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
        ap = average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0

        vals.append([fold, acc, precision, recall, f1, auc, ap, fpr, fnr, tpr, tnr])

        print(f"Fold {fold}: \nAccuracy: {acc:.4f}")
        print(f"Confusion matrix:\n{cm}")
        print(f"AUC: {auc:.4f}")
        print(f"F1: {f1:.4f}")

        log_path = Path(config["LOGS_PATH"]).resolve() / f"test_log_fold{fold+1}.txt"
        log_test = Logger(log_path, mode='a')
        log_test.log_test(config, fold, acc, precision, recall, f1, auc, ap, fpr, fnr, tpr, tnr)