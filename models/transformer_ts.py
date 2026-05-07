import math
import torch
import torch.nn.functional as F
import torch.nn as nn


def build_model_ts(config: dict):
    model_cfg = config["TST1"] 
    
    return TST1(
        input_dim          = config["N_ROIS"],         
        max_seq_len        = config["MAX_SEQ_LEN"],   
        d_model            = model_cfg["D_MODEL"],
        dim_feedforward    = model_cfg["DIM_FEEDFORWARD"],
        num_encoder_layers = model_cfg["NUM_ENCODER_LAYERS"],
        n_heads            = model_cfg["N_HEADS"],
        enc_dropout        = model_cfg["ENC_DROP"],
        use_cls_token      = model_cfg["USE_CLS_TOKEN"]
    )


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    """
    
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TST1(nn.Module):
    def __init__(self, input_dim = 200, #N_ROIS
                max_seq_len = 200, #T
                d_model = 512, 
                dim_feedforward = 2048,
                num_encoder_layers = 6,
                n_heads = 8,
                enc_dropout = 0.1,
                use_cls_token = True):
        
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # CLS token (used for classification tasks)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        
        # Input Embedding Layer: Maps ROI features at each time point to the embedding space
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_seq_len + 1, dropout=enc_dropout
        )

        self.transformer_encoder = TransformerEncoderBlock(d_model, dim_feedforward,
                                    num_encoder_layers, n_heads, enc_dropout)
        
        self.pretraining_decoder = PretrainingDecoder(d_model,input_dim,enc_dropout)

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
            x: Tensor, shape (batch, T, n_rois)
            mode: 'pretrain' or 'finetune'
        
        Returns:
            pretrain mode: Reconstructed time series (batch, T, n_rois)
            finetune mode: CLS token features (batch, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        
        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer Encoding
        x = self.transformer_encoder(x)
        x = self.norm(x)
        
        if mode == 'finetune':
            # Return CLS token features
            if self.use_cls_token:
                return x[:, 0, :]  # (batch, emb_dim)
            else:
                # Use mean pooling if no CLS token is present
                return x.mean(dim=1)  # (batch, emb_dim)
        else:
            # pretrain mode: Reconstruct time series
            if self.use_cls_token:
                x = x[:, 1:, :]  # Remove CLS token
            
            # Decoding reconstruction
            output = self.pretraining_decoder(x)  # (batch, T, n_rois)
            return output
        
    def get_features(self, x):
        """
        Get feature representations (for contrastive learning)
        
        Args:
            x: Tensor, shape (batch, T, n_rois)
        
        Returns:
            features: Tensor, shape (batch, emb_dim)
        """
        return self.forward(x, mode='finetune')
    
    def load_pretrained(self, checkpoint_path, strict=True):
        """
        Load pre-trained weights
        
        Args:
            checkpoint_path: Path to pre-trained weights
            strict: Whether to perform strict matching
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out decoder weights (not needed during fine-tuning)
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
        # x: (B, T, N_ROIS)
        x = self.encoder(x) 
        return x
    

class PretrainingDecoder(nn.Module):
    
    def __init__(self,d_model, n_rois, dropout):
        super().__init__()


        self.pretrain_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_rois)
        )

    def forward(self,x: torch.Tensor):
        return self.pretrain_decoder(x)
        
 

if __name__ == "__main__":
    import torch

    config = {
        "N_ROIS":      200,
        "MAX_SEQ_LEN": 150,
        "TST1": {
            "D_MODEL":             128,
            "DIM_FEEDFORWARD":     256,
            "NUM_ENCODER_LAYERS":    2,
            "N_HEADS":               4,
            "ENC_DROP":            0.1,
            "USE_CLS_TOKEN":      True,
        }
    }

    model = build_model_ts(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    B, T, R = 4, 100, config["N_ROIS"]
    x = torch.randn(B, T, R)

    model.eval()
    with torch.no_grad():
        out_pretrain = model.forward(x, mode="pretrain")
    print(f"[pretrain]  input:  {tuple(x.shape)}")
    print(f"[pretrain]  output: {tuple(out_pretrain.shape)}")
    assert out_pretrain.shape == (B, T, R), "❌ pretrain shape incorrecto"
    print("✓ pretrain OK")

    with torch.no_grad():
        out_finetune = model.forward(x, mode="finetune")
    print(f"[finetune]  output: {tuple(out_finetune.shape)}")
    assert out_finetune.shape == (B, config["TST1"]["D_MODEL"]), "❌ finetune shape incorrecto"
    print("✓ finetune OK")

    
    mask = torch.rand(B, T, R) > 0.85
    masked_x = x.clone()
    masked_x[mask] = 0.0

    model.eval()
    with torch.no_grad():
        pred = model.forward(masked_x)
    print(f"[wrapper]   pred:  {tuple(pred.shape)}")
    assert pred.shape == (B, T, R), "❌ wrapper pred shape incorrecto"
    print("✓ TransformerTSForPretrain OK")

    print("\n✅ Todos los tests pasaron.")
