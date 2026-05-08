from training.tasks.contrastive import ContrastiveTask

def run_contrastive(
    dual_stream_model,
    dataloader,
    contrastive_module,
    optimizer,
    device,
    epochs=50
):
    """
    Entrenamiento de aprendizaje contrastivo
    
    Args:
        model: Modelo Dual-Stream
        dataloader: Cargador de datos
        contrastive_module: Módulo de aprendizaje contrastivo
        optimizer: Optimizador
        device: Dispositivo (CPU/GPU)
        epochs: Número de épocas de entrenamiento
    
    Returns:
        losses: Lista de pérdidas por cada época
    """
    from tqdm import tqdm
    
    dual_stream_model.train()
    contrastive_module.train()
    losses = []
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Contrastive Epoch {epoch}/{epochs}")
        for batch in pbar:
            timeseries = batch['timeseries'].to(device)
            pcc_vector = batch['pcc_vector'].to(device)
            
            # Obtener características
            h_ts, h_fc = dual_stream_model.get_features(timeseries, pcc_vector)
            
            # Calcular pérdida contrastiva
            optimizer.zero_grad()
            loss, z_ts, z_fc = contrastive_module(h_ts, h_fc)
            
            # Retropropagación
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    return losses
