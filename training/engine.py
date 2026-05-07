

def train_one_epoch_contrastive(model, dataloader, task, optimizer, device):
    """

    """
    model.train()
    task.contrastive_module.train() # Asegurar que los ProjectionHeads estén en train
    
    running_loss = 0.0
    
    for batch in dataloader:
        # 1. Preparación de datos
        timeseries = batch['timeseries'].to(device)
        pcc_vector = batch['pcc_vector'].to(device)
        
        optimizer.zero_grad()
        
        # 2. Uso de la TASK (aquí delegas la lógica que ya escribiste)
        # Tu ContrastiveTask ya hace el model.get_features y el cálculo de pérdida
        loss = task.execution_step(model, timeseries, pcc_vector)
        
        # 3. Optimización
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)