import torch.nn as nn


class ClassificationTask:
    def __init__(self,device):
        self.criterion = nn.CrossEntropyLoss().to(device)

    def execution_step(self, model, ts_batch, pcc_batch, targets):
        preds = model(ts_batch, pcc_batch)
        loss = self.criterion(preds, targets)
        return loss

