import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
import torch.nn.functional as F
from utils.logger import *
from pathlib import Path


def test(model,fold, accs: list,vals:list, config:dict, test_loader: DataLoader, criterion):
    model.eval()
    y_true, y_preds, y_probs = [], [], []
    
    with torch.no_grad():
        for _, (windows,labels) in enumerate(test_loader):
            inputs: torch.Tensor = windows.to(config["DEVICE"])

            # expected_shape = (config["BATCH_SIZE"], config["WINDOWS_SIZE"], config["NRO_ROIS"])
            # if inputs.shape != expected_shape:
            #     raise ValueError(f"Error de shapes antes de pasar datos al model, se esperaba {expected_shape}, recibido {inputs.shape} ")
            
            labels = labels.to(config["DEVICE"])

            outputs = model(inputs)
            loss = criterion(outputs,labels)
            #test_running_loss+=loss.item()
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()  # [B, num_classes]
            y_true.extend(labels.detach().cpu().numpy())               # labels ya son índices
            y_preds.extend(np.argmax(probs, axis=1))                   # clase con mayor prob
            y_probs.extend(probs[:, 1])                                # prob de clase positiva


        acc       = accuracy_score(y_true, y_preds)
        accs.append(acc)
        cm        = confusion_matrix(y_true, y_preds)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        recall    = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        f1        = 2 * (precision * recall) / (precision + recall)
        auc       = roc_auc_score(y_true, y_probs)
        ap        = average_precision_score(y_true, y_probs)
        fpr       = cm[0, 1] / (cm[0, 1] + cm[0, 0])
        fnr       = cm[1, 0] / (cm[1, 0] + cm[1, 1])
        tpr       = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        tnr       = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        vals.append([fold, acc, precision, recall,
                     f1, auc, ap, fpr, fnr, tpr, tnr])

        print(f"Fold {fold}: \nAccuracy: {acc:4f}")
        print(f"Confusion matrix:\n{cm}")
        print(f"AUC: {auc:4f}")
        print(f"F1: {f1:4f}")

        log_path = Path(config["LOGS_PATH"]).resolve() / f"test_log_fold{fold+1}.txt"

        log_test = Logger(log_path, mode='a')

        log_test.log_test(config, fold, acc, precision, recall, f1, auc, ap, fpr, fnr, tpr, tnr)

