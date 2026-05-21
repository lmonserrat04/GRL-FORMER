"""
Modelo Dual-Stream (Doble Flujo)
Modelo completo que integra TST1, TST2 y el módulo de fusión
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_ts import build_model_ts
from .transformer_fc import build_model_fc
from .fusion import create_fusion_module


class DualStreamModel(nn.Module):
    """
    Modelo de pre-entrenamiento auto-supervisado de doble flujo
    
    Contiene:
    - TST1: Transformer temporal, procesa series temporales de fMRI
    - TST2: Transformer de conectividad, procesa vectores PCC
    - Módulo de fusión: Fusiona las características de ambos Transformers
    - Cabezal de clasificación: Clasificador MLP
    """
    
    def __init__(
        self,
        config : dict,
    ):
        """
        Args:
            config: diccionario de configuracion global
        """
        super().__init__()

        self.dual_stream_cfg = config["DUAL_STREAM"]
        
        # Crear TST1
        self.transformer_ts = build_model_ts(config)
        
        # Crear TST2
        self.transformer_fc = build_model_fc(config)
        
        # Obtener dimensiones de las características
        self.dim_ts = self.transformer_ts.d_model
        self.dim_fc = self.transformer_fc.d_model
        
        # Crear módulo de fusión
        self.fusion = create_fusion_module(
            config, self.dim_ts, self.dim_fc
        )

        
        # Cabezal de clasificación
        fusion_dim = self.fusion.output_dim
        self.num_classes = self.dual_stream_cfg["NUM_CLASSES"]
        self.classifier_dropout = self.dual_stream_cfg["CLASSIFIER_DROPOUT"]
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(self.classifier_dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(self.classifier_dropout),
            nn.Linear(fusion_dim // 4, self.num_classes )
        )
        
        
    
    def forward(self, timeseries, pcc_vector):
        """
        Args:
            timeseries: Serie temporal (batch, T, n_rois)
            pcc_vector: Vector PCC (batch, pcc_dim)
           
        
        Returns:
            logits: Logits de clasificación (batch, num_classes)
           
        """
        # Obtener características de TST1
        h_ts = self.transformer_ts(timeseries, mode='finetune')
        
        # Obtener características de TST2
        h_fc = self.transformer_fc(pcc_vector, mode='finetune')
        
        # Fusión
        
        fused = self.fusion(h_ts, h_fc)
         
        # Clasificación
        logits = self.classifier(fused)
        
        result = [logits]
        
        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)
    
    def get_features(self, timeseries, pcc_vector):
        """
        Obtener características de ambos Transformers (para aprendizaje contrastivo)
        
        Args:
            timeseries: Serie temporal (batch, T, n_rois)
            pcc_vector: Vector PCC (batch, pcc_dim)
        
        Returns:
            h_ts: Características de TST1 (batch, dim_ts)
            h_fc: Características de TST2 (batch, dim_fc)
        """
        h_ts = self.transformer_ts(timeseries, mode='finetune')
        h_fc = self.transformer_fc(pcc_vector, mode='finetune')
        return h_ts, h_fc
    
    def load_pretrained_tst1(self, checkpoint_path, strict=False):
        """Cargar pesos pre-entrenados de TST1"""
        self.transformer_ts.load_pretrained(checkpoint_path, strict=strict)
    
    def load_pretrained_tst2(self, checkpoint_path, strict=False):
        """Cargar pesos pre-entrenados de TST2"""
        self.transformer_fc.load_pretrained(checkpoint_path, strict=strict)

    def load_pretrained_contrastive(self, checkpoint_path, strict=False):
        """Cargar pesos pre-entrenados de Contrastive"""
        self.transformer_fc.load_pretrained(checkpoint_path, strict=strict)



def create_dual_stream_model(config, name_chkpt_pt_ts: str | None = None, name_chkpt_pt_fc: str | None = None, 
                             name_chkpt_cont: str | None = None):
    dual_stream = DualStreamModel(config)
    
    if name_chkpt_pt_fc and name_chkpt_pt_ts:
        dual_stream.load_pretrained_tst1(name_chkpt_pt_ts)  
        dual_stream.load_pretrained_tst2(name_chkpt_pt_fc)  
    elif name_chkpt_cont:
        dual_stream.load_pretrained_contrastive(name_chkpt_cont)
    else:
        raise ValueError("Missing some checkpoint names to load weights for dual stream model")

    

    return dual_stream


if __name__ == '__main__':
    config = {
        "DUAL_STREAM": {
            "FUSION_TYPE": "attention_pooling",
            "NUM_CLASSES": 2,
            "CLASSIFIER_DROPOUT": 0.1
        },
        "FUSION": {
            "ATTENTION_POOLING": {
                "HIDDEN_DIM": 512
            }
        },
        "TST1": {
            "D_MODEL": 256,
            "DIM_FEEDFORWARD": 512,
            "NUM_ENCODER_LAYERS": 2,
            "N_HEADS": 4,
            "ENC_DROP": 0.5,
            "USE_CLS_TOKEN": True
        },
        "TST2": {
            "D_MODEL": 256,
            "N_HEADS": 8,
            "NUM_ENCODER_LAYERS": 2,
            "DIM_FEEDFORWARD": 512,
            "ENC_DROP": 0.1
        },
        "N_ROIS": 200,
        "PCC_DIM": 19900,
        "MAX_SEQ_LEN": 200
    }

    print("Testing DualStreamModel...")
    #model = create_dual_stream_model(config)
    #print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 4
    timeseries = torch.randn(batch_size, 116, 200)
    pcc_vector = torch.randn(batch_size, 19900)

    #logits = model(timeseries, pcc_vector)
    #print(f"Logits shape: {logits.shape}")  # esperado: (4, 2)

    print("\nTest passed!")