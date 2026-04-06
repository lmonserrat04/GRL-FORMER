import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from main import config
from callbacks import EarlyStopping

class Trainer():

    def __init__(self,model:nn.Module , train_loader: DataLoader,val_loader:DataLoader, config: dict, device, optimizer, criterion, scheduler):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.n_epochs = self.config["N_EPOCHS"]
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

    def fit(self):
        early_stopping = EarlyStopping(self.model, self.config["PATIENCE"],self.config["MIN_DELTA"])
        
        with tqdm(range(self.n_epochs), unit="epoch") as tepoch:
            for epoch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                train_running_loss = 0.0
                val_running_loss = 0.0
                self.model.train()
                train_running_loss += train_one_epoch(self.model,self.config,self.train_loader,self.device, self.optimizer, self.criterion)
                
                self.model.eval()
                val_running_loss+= evaluate_one_epoch(self.model,self.config,self.val_loader,self.device,self.criterion)
                self.scheduler.step()        

                avg_train_loss = train_running_loss/len(self.train_loader) #Propositos de logging
                avg_val_loss = val_running_loss / len(self.val_loader)    #Propositos de logging y early stopping

                best = early_stopping(self.model,avg_val_loss)
                if best:
                    #TODO
                    break


                tepoch.set_postfix(val_loss=f"{avg_val_loss:.4f}")
                



def train_one_epoch(model:nn.Module,config: dict, train_loader:DataLoader, device, optimizer, criterion):
    
    
    train_running_loss = 0.0
    
    
    for _, (windows,labels) in enumerate(train_loader):
        inputs = windows.to(device)

        expected_shape = (config["BATCH_SIZE"], config["WINDOWS_SIZE"], config["NRO_ROIS"])
        if inputs.shape != expected_shape:
            raise ValueError(f"Error de shapes antes de pasar datos al model, se esperaba {expected_shape}, recibido {inputs.shape} ")

        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
    

    return train_running_loss


def evaluate_one_epoch(model: nn.Module,config: dict, val_loader: DataLoader, device, criterion):
    val_running_loss = 0.0

    with torch.no_grad():
        for _, (windows,labels) in enumerate(val_loader):
            inputs = windows.to(device)

            expected_shape = (config["BATCH_SIZE"], config["WINDOWS_SIZE"], config["NRO_ROIS"])
            if inputs.shape != expected_shape:
                raise ValueError(f"Error de shapes antes de pasar datos al model, se esperaba {expected_shape}, recibido {inputs.shape} ")
            
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs,labels)
            val_running_loss+=loss.item()


    
    return val_running_loss
