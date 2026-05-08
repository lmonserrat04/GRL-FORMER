"""
Módulo de Aprendizaje Contrastivo
Implementa la pérdida InfoNCE para alinear las representaciones de características de TST1 y TST2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoNCELoss(nn.Module):
    """
    Pérdida contrastiva InfoNCE
    Las características TST1 y TST2 de la misma muestra forman pares positivos
    Las de muestras diferentes forman pares negativos
    """
    
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Parámetro de temperatura que controla la nitidez de la distribución
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: Características TST1 (batch, dim_ts)
            h_fc: Características TST2 (batch, dim_fc)
        
        Returns:
            loss: Pérdida InfoNCE
        """
        batch_size = h_ts.shape[0]
        
        # Normalización L2
        h_ts = F.normalize(h_ts, p=2, dim=1)
        h_fc = F.normalize(h_fc, p=2, dim=1)
        
        # Calcular matriz de similitud
        # sim[i, j] = h_ts[i] · h_fc[j]
        sim_matrix = torch.matmul(h_ts, h_fc.T) / self.temperature  # (batch, batch)
        
        # Los pares positivos están en la diagonal
        labels = torch.arange(batch_size, device=h_ts.device)
        
        # Pérdida contrastiva bidireccional
        # TST1 -> TST2
        loss_ts2fc = F.cross_entropy(sim_matrix, labels)
        # TST2 -> TST1
        loss_fc2ts = F.cross_entropy(sim_matrix.T, labels)
        
        # Pérdida promedio
        loss = (loss_ts2fc + loss_fc2ts) / 2
        
        return loss


class ProjectionHead(nn.Module):
    """
    Cabezal de Proyección
    Proyecta las características al espacio de aprendizaje contrastivo
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        """
        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión de la capa oculta
            output_dim: Dimensión de salida
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ContrastiveWrapper(nn.Module):
    """
    Clase envolvente (wrapper) de aprendizaje contrastivo
    Contiene los cabezales de proyección y la pérdida contrastiva
    """
    
    def __init__(
        self,
        dim_ts,
        dim_fc,
        proj_hidden_dim=256,
        proj_output_dim=128,
        temperature=0.07,
        loss_type='infonce'
    ):
        """
        Args:
            dim_ts: Dimensión de características de TST1
            dim_fc: Dimensión de características de TST2
            proj_hidden_dim: Dimensión oculta del cabezal de proyección
            proj_output_dim: Dimensión de salida del cabezal de proyección
            temperature: Parámetro de temperatura
            loss_type: Tipo de pérdida ('infonce' o 'ntxent')
        """
        super().__init__()
        
        # Cabezales de proyección
        self.proj_ts = ProjectionHead(dim_ts, proj_hidden_dim, proj_output_dim)
        self.proj_fc = ProjectionHead(dim_fc, proj_hidden_dim, proj_output_dim)
        
        # Pérdida contrastiva
        if loss_type == 'infonce':
            self.criterion = InfoNCELoss(temperature)
        #elif loss_type == 'ntxent':
        #    self.criterion = NTXentLoss(temperature)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.loss_type = loss_type
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: Características TST1 (batch, dim_ts)
            h_fc: Características TST2 (batch, dim_fc)
        
        Returns:
            loss: Pérdida contrastiva
            z_ts: Características proyectadas de TST1
            z_fc: Características proyectadas de TST2
        """
        # Proyección
        z_ts = self.proj_ts(h_ts)
        z_fc = self.proj_fc(h_fc)
        
        # Calcular pérdida
        loss = self.criterion(z_ts, z_fc)
        
        return loss, z_ts, z_fc


class ContrastiveTask:

    def __init__(
        self,
        dim_ts,
        dim_fc,
        proj_hidden_dim=256,
        proj_output_dim=128,
        temperature=0.07,
        loss_type='infonce',
        device = 'cuda'
    ):
        

        self.contrastive_module = ContrastiveWrapper(dim_ts,dim_fc,proj_hidden_dim,proj_output_dim,temperature,loss_type).to(device)


    def execution_step(self, model,batch_ts, batch_pcc):

        h_ts, h_fc = model.get_features(batch_ts, batch_pcc)

        loss,_,_ = self.contrastive_module(h_ts,h_fc)

        return loss




if __name__ == '__main__':
    # Prueba del módulo de aprendizaje contrastivo
    print("Testing contrastive learning modules...")
    
    batch_size = 32
    dim_ts = 512
    dim_fc = 256
    
    h_ts = torch.randn(batch_size, dim_ts)
    h_fc = torch.randn(batch_size, dim_fc)
    
    # Prueba de InfoNCE
    print("\nTesting InfoNCELoss:")
    criterion = InfoNCELoss(temperature=0.07)
    loss = criterion(h_ts, h_ts)
    print(f"  Loss: {loss.item():.4f}")
    
    # Prueba de NT-Xent
    # print("\nTesting NTXentLoss:")
    # criterion = NTXentLoss(temperature=0.5)
    # loss = criterion(h_ts, h_fc)
    # print(f"  Loss: {loss.item():.4f}")
    
    # Prueba de ContrastiveWrapper
    print("\nTesting ContrastiveWrapper:")
    wrapper = ContrastiveWrapper(dim_ts, dim_fc)
    loss, z_ts, z_fc = wrapper(h_ts, h_fc)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Projected TST1 shape: {z_ts.shape}")
    print(f"  Projected TST2 shape: {z_fc.shape}")
    
    print("\nAll tests passed!")