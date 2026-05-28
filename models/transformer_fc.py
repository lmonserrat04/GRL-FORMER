import math
import torch
import torch.nn.functional as F
import torch.nn as nn


def build_model_fc(config: dict):
    model_cfg = config["TST2"] 
    
    return TST2(
        pcc_dim            = model_cfg["PCC_DIM"],           
        d_model            = model_cfg["D_MODEL"],
        dim_feedforward    = model_cfg["DIM_FEEDFORWARD"],
        num_encoder_layers = model_cfg["NUM_ENCODER_LAYERS"],
        n_heads            = model_cfg["N_HEADS"],
        enc_dropout        = model_cfg["ENC_DROP"]
    )


class TST2(nn.Module):
    def __init__(self,
                pcc_dim = 19900,
                d_model = 256, 
                dim_feedforward = 512,
                num_encoder_layers = 2,
                n_heads = 8,
                enc_dropout = 0.1):
        
        super().__init__()

        self.pcc_dim = pcc_dim
        self.d_model = d_model
    
        
        # Input Embedding Layer: Maps ROI features at each time point to the embedding space
        self.input_embedding = nn.Linear(pcc_dim, d_model)


        self.transformer_encoder = TransformerEncoderBlock(d_model, dim_feedforward,
                                    num_encoder_layers, n_heads, enc_dropout)
        
        self.pretraining_decoder = PretrainingDecoder(d_model,pcc_dim,enc_dropout)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=enc_dropout)

        # Layer Normalization
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        
        

    def forward(self, x: torch.Tensor, mode='pretrain'):
        """
        Args:
            x: Tensor, shape (batch, pcc_dim)
            mode: 'pretrain' or 'finetune'
        
        Returns:
            pretrain mode: Reconstructed PCC vector (batch, pcc_dim)
            finetune mode: Feature vector (batch, d_model)
        """
        
        # Input embedding
        x = self.input_embedding(x) * math.sqrt(self.d_model)

       
        # Add dummy sequence dimension (batch, 1, d_model)
        # Añade una dimensión de secuencia ficticia (batch, 1, d_model)
        x = x.unsqueeze(1)

       
        # Transformer Encoding
        x = self.transformer_encoder(x)
        

        # Remove sequence dimension (batch, d_model)
        # Elimina la dimensión de secuencia (batch, d_model)
        x = x.squeeze(1)


        # Activation and Normalization / Activación y Normalización
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        
        if mode == 'finetune':
            # Returns feature vector / Devuelve el vector de características
            return x  # (batch, d_model)
        else:
            # pretrain mode: Reconstruct original PCC vector 
            # modo pretrain: Reconstruye el vector PCC original
            output = self.pretraining_decoder(x)  # (batch, pcc_dim)
            return output
        
    def get_features(self, x):
        """
        Get feature representation (used for contrastive learning)
        Obtiene la representación de características (usada para aprendizaje contrastivo)
        
        Args:
            x: Tensor, shape (batch, pcc_dim)
        
        Returns:
            features: Tensor, shape (batch, d_model)
        """
        return self.forward(x, mode='finetune')
    
    def load_pretrained(self, checkpoint_path, strict=True):
        """
        Load pretrained weights / Carga pesos pre-entrenados
        
        Args:
            checkpoint_path: Path to weights / Ruta a los pesos
            strict: Whether to match strictly / Si debe coincidir estrictamente
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only= True)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out decoder weights (not needed for fine-tuning)
        # Filtra los pesos del decodificador (no necesarios para el ajuste fino)
        if not strict:
            state_dict = {
                k: v for k, v in state_dict.items() 
                if 'pretrain_decoder' not in k
            }
        
        self.load_state_dict(state_dict, strict=strict)
        print(f"Loaded pretrained weights from {checkpoint_path}")
        

class TransformerEncoderBlock(nn.Module):
    def __init__(self,d_model, dim_feedforward, 
                 num_encoder_layers, n_heads, dropout):  
        super().__init__()

        self.d_model = d_model
       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first= True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, x: torch.Tensor):
        # x: (B, PCC)
        x = self.encoder(x) 
        return x
    

class PretrainingDecoder(nn.Module):
    
    def __init__(self,d_model, pcc_dim, dropout):
        super().__init__()


        self.pretrain_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pcc_dim)
        )

    def forward(self,x: torch.Tensor):
        return self.pretrain_decoder(x)
    
    

if __name__ == "__main__":
    import torch

    # ── Config con los nombres de TST1 ─────────────────────────────────────
    config = {
        "PCC_DIM":              19900,   
        "TST2" : {
            "D_MODEL":             128,   # d_model
            "DIM_FEEDFORWARD":     256,   # dim_feedforward
            "NUM_ENCODER_LAYERS":    2,   # num_encoder_layers
            "N_HEADS":               4,   # n_heads
            "ENC_DROP":            0.1,   # enc_dropout
        }
    }

    # ── Construcción del modelo ────────────────────────────────────────────
    model = build_model_fc(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    # ── Datos sintéticos ──────────────────────
    B, PCC = 4,  config["PCC_DIM"]
    x = torch.randn(B, PCC)

    # ── 1. Modo pretrain (reconstrucción) ─────────────────────────────────
    model.eval()
    with torch.no_grad():
        out_pretrain = model.forward(x, mode="pretrain")
    print(f"[pretrain]  input:  {tuple(x.shape)}")
    print(f"[pretrain]  output: {tuple(out_pretrain.shape)}")
    assert out_pretrain.shape == (B, PCC), "❌ pretrain shape incorrecto"
    print("✓ pretrain OK")

    # ── 2. Modo finetune ──────────────────────────────────────
    with torch.no_grad():
        out_finetune = model.forward(x, mode="finetune")
    print(f"[finetune]  output: {tuple(out_finetune.shape)}")
    assert out_finetune.shape == (B, config["TST2"]["D_MODEL"]), "❌ finetune shape incorrecto"
    print("✓ finetune OK")

    # ── 3. (máscara + loss) ───────────────────────
   

    mask = torch.rand(4, 19900) > 0.85
    masked_x = x.clone()
    masked_x[mask] = 0.0

    model.eval()
    with torch.no_grad():
        pred = model.forward(masked_x)

    print(f"[wrapper]   pred:  {tuple(pred.shape)}")
    assert pred.shape == (B, PCC), "❌ wrapper pred shape incorrecto"
    
    print("✓ TransformerFCForPretrain OK")


    print("\n✅ Todos los tests pasaron.")
            
