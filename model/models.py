import torch
import torch.nn.functional as F
import torch.nn as nn


def build_model(config: dict):

    return SAT(
        input_dim          = config["N_ROIS"],
        seq_len            = config["WINDOW_SIZE"],
        d_model            = config["D_MODEL"],
        dim_feedforward    = config["DIM_FEEDFORWARD"],
        num_encoder_layers = config["NUM_ENCODER_LAYERS"],
        n_heads            = config["N_HEADS"],
        enc_dropout        = config["ENC_DROP"],
        cls_dropout        = config["CLS_DROPOUT"],
    )



class SAT(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, dim_feedforward,
                 num_encoder_layers, n_heads, enc_dropout, cls_dropout):
        super().__init__()
        self.encoder = EncoderBlock(input_dim, seq_len, d_model, dim_feedforward,
                                    num_encoder_layers, n_heads, enc_dropout)
        self.classificator = Classificator(d_model, cls_dropout)

    def forward(self, x: torch.Tensor, mask=None):
        x = self.encoder(x, mask=mask)       # (B, T, N_ROIS) -> (B, T, d_model)
        x = self.classificator(x)            # (B, 2)
        return x



class EncoderBlock(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, dim_feedforward, 
                 num_encoder_layers, n_heads, dropout):  
        super().__init__()
        self.d_model = d_model
        self.inp_emb = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Embedding(seq_len, d_model)         

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, x: torch.Tensor, mask=None):
        # x: (B, T, N_ROIS)
        _, T, _ = x.shape
        
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        pos_enc = self.pos_enc(positions)                           # [1, T, d_model]
        x = self.inp_emb(x) + pos_enc                              # [B, T, d_model]
        x = self.encoder(x, src_key_padding_mask=mask)
        
        return x
    

class Classificator(nn.Module):
    def __init__(self, d_model,dropout):
        super().__init__()
        self.d_model = d_model
        
        self.mlp_layer_1 = nn.Linear(d_model,64)
        self.mlp_layer_2 = nn.Linear(64,2)
        

    def forward(self, x: torch.Tensor):
        x = x.mean(dim = 1)
        x = self.mlp_layer_1(x)
        x = F.relu(x)
        x = self.mlp_layer_2(x)

        return x
         


            
