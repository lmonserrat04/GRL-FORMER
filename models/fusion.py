"""
Módulo de fusión
Implementa múltiples estrategias de fusión de características: concatenación, puerta (gated) y atención cruzada.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConcatFusion(nn.Module):
    """
    Fusión por concatenación simple
    Concatena directamente los dos vectores de características
    """
    
    def __init__(self, dim_ts, dim_fc, output_dim=None):
        """
        Args:
            dim_ts: Dimensión de características de TST1
            dim_fc: Dimensión de características de TST2
            output_dim: Dimensión de salida (si es None, se concatenan directamente)
        """
        super().__init__()
        
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        self.output_dim = output_dim or (dim_ts + dim_fc)
        
        if output_dim is not None:
            self.proj = nn.Linear(dim_ts + dim_fc, output_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: Características TST1 (batch, dim_ts)
            h_fc: Características TST2 (batch, dim_fc)
        
        Returns:
            fused: Características fusionadas (batch, output_dim)
        """
        concat = torch.cat([h_ts, h_fc], dim=-1)
        return self.proj(concat)


class GatedFusion(nn.Module):
    """
    Fusión por compuerta (Gated)
    Utiliza un mecanismo de compuerta aprendible para fusionar ponderadamente las dos características
    gate * h_ts + (1 - gate) * h_fc
    """
    
    def __init__(self, dim_ts, dim_fc, hidden_dim=None):
        """
        Args:
            dim_ts: Dimensión de características de TST1
            dim_fc: Dimensión de características de TST2
            hidden_dim: Dimensión de la capa oculta
        """
        super().__init__()
        
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        
        # Proyectar ambas características a la misma dimensión
        self.output_dim = max(dim_ts, dim_fc)
        
        self.proj_ts = nn.Linear(dim_ts, self.output_dim)
        self.proj_fc = nn.Linear(dim_fc, self.output_dim)
        
        # Red de compuerta
        hidden_dim = hidden_dim or self.output_dim
        self.gate_net = nn.Sequential(
            nn.Linear(dim_ts + dim_fc, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: Características TST1 (batch, dim_ts)
            h_fc: Características TST2 (batch, dim_fc)
        
        Returns:
            fused: Características fusionadas (batch, output_dim)
        """
        # Proyectar a la misma dimensión
        h_ts_proj = self.proj_ts(h_ts)
        h_fc_proj = self.proj_fc(h_fc)
        
        # Calcular pesos de la compuerta
        concat = torch.cat([h_ts, h_fc], dim=-1)
        gate = self.gate_net(concat)
        
        # Fusión por compuerta
        fused = gate * h_ts_proj + (1 - gate) * h_fc_proj
        return fused


class CrossAttentionFusion(nn.Module):
    """
    Fusión por atención cruzada (Cross-Attention)
    Utiliza un mecanismo de atención cruzada para fusionar las dos características
    """
    
    def __init__(self, dim_ts, dim_fc, n_heads=8, dropout=0.1):
        """
        Args:
            dim_ts: Dimensión de características de TST1
            dim_fc: Dimensión de características de TST2
            n_heads: Número de cabezales de atención
            dropout: Ratio de Dropout
        """
        super().__init__()
        
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        
        # Unificar a la dimensión mayor
        self.d_model = max(dim_ts, dim_fc)
        
        # Capas de proyección
        self.proj_ts = nn.Linear(dim_ts, self.d_model)
        self.proj_fc = nn.Linear(dim_fc, self.d_model)
        
        # Atención cruzada: TS -> FC
        self.cross_attn_ts2fc = nn.MultiheadAttention(
            self.d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Atención cruzada: FC -> TS
        self.cross_attn_fc2ts = nn.MultiheadAttention(
            self.d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Red de alimentación hacia adelante (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        
        # Normalización por capas
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        
        self.output_dim = self.d_model
    
    def forward(self, h_ts, h_fc, return_attention=False):
        """
        Args:
            h_ts: Características TST1 (batch, dim_ts)
            h_fc: Características TST2 (batch, dim_fc)
            return_attention: Si devuelve los pesos de atención
        
        Returns:
            fused: Características fusionadas (batch, d_model)
            attention_weights (opcional): Diccionario de pesos de atención
        """
        # Proyectar a la misma dimensión y añadir dimensión de secuencia
        h_ts = self.proj_ts(h_ts).unsqueeze(1)  # (batch, 1, d_model)
        h_fc = self.proj_fc(h_fc).unsqueeze(1)  # (batch, 1, d_model)
        
        # Atención cruzada
        # TS actúa como Query, FC como Key/Value
        attn_ts, attn_weights_ts2fc = self.cross_attn_ts2fc(h_ts, h_fc, h_fc) # attn_ts.shape : (batch,1,1)
        h_ts = self.norm1(h_ts + attn_ts)
        
        # FC actúa como Query, TS como Key/Value
        attn_fc, attn_weights_fc2ts = self.cross_attn_fc2ts(h_fc, h_ts, h_ts)
        h_fc = self.norm2(h_fc + attn_fc)
        
        # Concatenar y pasar por la red FFN
        concat = torch.cat([h_ts, h_fc], dim=-1)  # (batch, 1, d_model*2)
        fused = self.ffn(concat)  # (batch, 1, d_model)
        fused = self.norm3(fused)
        
        fused_out = fused.squeeze(1)  # (batch, d_model)
        
        if return_attention:
            attention_weights = {
                'ts2fc': attn_weights_ts2fc,  # (batch, n_heads, 1, 1)
                'fc2ts': attn_weights_fc2ts   # (batch, n_heads, 1, 1)
            }
            return fused_out, attention_weights
        else:
            return fused_out


class BilinearFusion(nn.Module):
    """
    Fusión bilineal
    Utiliza una transformación bilineal para fusionar las dos características
    """
    
    def __init__(self, dim_ts, dim_fc, output_dim=256):
        """
        Args:
            dim_ts: Dimensión de características de TST1
            dim_fc: Dimensión de características de TST2
            output_dim: Dimensión de salida
        """
        super().__init__()
        
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        self.output_dim = output_dim
        
        # Capa bilineal
        self.bilinear = nn.Bilinear(dim_ts, dim_fc, output_dim)
        
        # Proyecciones para conexión residual
        self.proj_ts = nn.Linear(dim_ts, output_dim)
        self.proj_fc = nn.Linear(dim_fc, output_dim)
        
        # Normalización por capas
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: Características TST1 (batch, dim_ts)
            h_fc: Características TST2 (batch, dim_fc)
        
        Returns:
            fused: Características fusionadas (batch, output_dim)
        """
        # Transformación bilineal
        bilinear_out = self.bilinear(h_ts, h_fc)
        
        # Residual
        residual = self.proj_ts(h_ts) + self.proj_fc(h_fc)
        
        # Fusión
        fused = self.norm(bilinear_out + residual)
        return fused


class AttentionPoolingFusion(nn.Module):
    """
    Fusión por pooling de atención
    Utiliza un mecanismo de atención para realizar una fusión ponderada de ambas características
    """
    
    def __init__(self,config, dim_ts, dim_fc):
        """
        Args:
            dim_ts: Dimensión de características de TST1
            dim_fc: Dimensión de características de TST2
            hidden_dim: Dimensión de la capa oculta
        """
        super().__init__()
        
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        self.output_dim = max(dim_ts, dim_fc)
        
        # Proyectar a la misma dimensión
        self.proj_ts = nn.Linear(dim_ts, self.output_dim)
        self.proj_fc = nn.Linear(dim_fc, self.output_dim)
        
        # Cálculo de pesos de atención
        hidden_dim = config["ATTENTION_POOLING"]["HIDDEN_DIM"] or self.output_dim
        self.attention = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: Características TST1 (batch, dim_ts)
            h_fc: Características TST2 (batch, dim_fc)
        
        Returns:
            fused: Características fusionadas (batch, output_dim)
        """
        # Proyectar a la misma dimensión
        h_ts = self.proj_ts(h_ts)
        h_fc = self.proj_fc(h_fc)
        
        # Apilar como secuencia (batch, 2, output_dim)
        features = torch.stack([h_ts, h_fc], dim=1)
        
        # Calcular pesos de atención
        attn_scores = self.attention(features)  # (batch, 2, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Fusión ponderada
        fused = (features * attn_weights).sum(dim=1)  # (batch, output_dim)
        return fused


class GatedMultiFusion(nn.Module):
    """
    Fusión múltiple por compuerta (Gated Multi-Fusion)
    Ejecuta en paralelo 5 estrategias de fusión (concat, gated, cross_attention, bilinear, attention_pooling),
    aprende una matriz de pesos de compuerta para elegir flexiblemente una o varias estrategias.
    Mejora: Profundización de la compuerta, inicialización sesgada hacia attention_pooling,
    almacenamiento de gate_weights para regularización de entropía.
    """

    def __init__(self, dim_ts, dim_fc, output_dim=256, hidden_dim=128, dropout=0.1,
                 init_favor_idx=4):
        """
        Args:
            dim_ts: Dimensión de características de TST1
            dim_fc: Dimensión de características de TST2
            output_dim: Dimensión final de la salida fusionada
            hidden_dim: Dimensión oculta interna de cada estrategia
            dropout: Ratio de Dropout
            init_favor_idx: Índice de la estrategia favorecida inicialmente (4=attention_pooling, óptima individualmente)
        """
        super().__init__()
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        self.output_dim = output_dim
        self.n_strategies = 5
        self.init_favor_idx = init_favor_idx
        self.last_gate_weights = None  # Para uso en regularización de entropía

        # 5 estrategias de fusión
        self.fusions = nn.ModuleList([
            ConcatFusion(dim_ts, dim_fc, output_dim=hidden_dim),
            GatedFusion(dim_ts, dim_fc, hidden_dim=hidden_dim),
            CrossAttentionFusion(dim_ts, dim_fc, n_heads=8, dropout=dropout),
            BilinearFusion(dim_ts, dim_fc, output_dim=hidden_dim),
            AttentionPoolingFusion(dim_ts, dim_fc, hidden_dim=hidden_dim),
        ])

        # Las dimensiones de salida pueden variar, unificarlas a output_dim
        fusion_dims = [
            hidden_dim,  # concat
            max(dim_ts, dim_fc),  # gated
            max(dim_ts, dim_fc),  # cross_attention
            hidden_dim,  # bilinear
            max(dim_ts, dim_fc),  # attention_pooling
        ]
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
            for d in fusion_dims
        ])

        # Red de compuerta profundizada
        self.gate_net = nn.Sequential(
            nn.Linear(dim_ts + dim_fc, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_strategies),
        )
        self._init_gate_favor_strategy()

    def _init_gate_favor_strategy(self):
        """Inicializa la compuerta: hace que la estrategia init_favor_idx (ej. attention_pooling) tenga mayor peso inicial"""
        last_linear = self.gate_net[-1]
        with torch.no_grad():
            last_linear.weight.data *= 0.1
            bias = torch.zeros(self.n_strategies)
            bias[self.init_favor_idx] = 2.0
            last_linear.bias.data.copy_(bias)

    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: Características TST1 (batch, dim_ts)
            h_fc: Características TST2 (batch, dim_fc)

        Returns:
            fused: Características fusionadas (batch, output_dim)
        """
        # Calcular en paralelo las 5 salidas de fusión
        fusion_outputs = []
        for i, fusion in enumerate(self.fusions):
            out = fusion(h_ts, h_fc)
            out = self.projections[i](out)
            fusion_outputs.append(out)

        # apilar: (batch, 5, output_dim)
        stacked = torch.stack(fusion_outputs, dim=1)

        # Pesos de compuerta: (batch, 5)
        gate_input = torch.cat([h_ts, h_fc], dim=-1)
        gate_logits = self.gate_net(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)
        self.last_gate_weights = gate_weights  # Para uso en regularización de entropía

        # Suma ponderada: (batch, 5, output_dim) * (batch, 5, 1) -> (batch, output_dim)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)

        return fused


def create_fusion_module(config, dim_ts, dim_fc):
    """
    Función de fábrica para crear el módulo de fusión
    
    Args:
        fusion_type: Tipo de fusión ('concat', 'gated', 'cross_attention', 'bilinear', 'attention_pooling')
        dim_ts: Dimensión de características de TST1
        dim_fc: Dimensión de características de TST2
        **kwargs: Parámetros adicionales
    
    Returns:
        fusion_module: Instancia del módulo de fusión
    """
    fusion_classes = {
        'concat': ConcatFusion,
        'gated': GatedFusion,
        'cross_attention': CrossAttentionFusion,
        'bilinear': BilinearFusion,
        'attention_pooling': AttentionPoolingFusion,
        'gated_multi': GatedMultiFusion,
    }

    fusion_type = config["DUAL_STREAM"]["FUSION_TYPE"]
    
    if fusion_type not in fusion_classes:
        raise ValueError(f"Unknown fusion type: {fusion_type}. "
                        f"Available: {list(fusion_classes.keys())}")
    
   
    
    return fusion_classes[fusion_type](config["FUSION"], dim_ts, dim_fc)


if __name__ == '__main__':
    config = {
        "FUSION": {
            "ATTENTION_POOLING": {
                "HIDDEN_DIM": 512
            }
        },
        "DUAL_STREAM": {
            "FUSION_TYPE": "attention_pooling"
        }
    }

    batch_size = 4
    dim_ts = 256  # TST1.D_MODEL
    dim_fc = 512  # TST2.D_MODEL

    h_ts = torch.randn(batch_size, dim_ts)
    h_fc = torch.randn(batch_size, dim_fc)

    print("Testing AttentionPoolingFusion...")
    fusion = AttentionPoolingFusion(config["FUSION"], dim_ts, dim_fc)
    output = fusion(h_ts, h_fc)
    print(f"Input  h_ts : {h_ts.shape}")
    print(f"Input  h_fc : {h_fc.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dim  : {fusion.output_dim}")
    print("\nTest passed!")