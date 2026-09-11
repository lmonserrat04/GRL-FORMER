"""
TST2: Connectivity Transformer / Transformer de Conectividad
Processes the PCC upper triangular vector using an element-wise masking strategy for pre-training.
Procesa el vector del triángulo superior de la matriz PCC empleando una estrategia de
enmascaramiento a nivel de elemento para el pre-entrenamiento.
"""

import math
import torch
import torch.nn as nn


class TransformerFC(nn.Module):
    """
    Connectivity Transformer (TST2)

    Input: (batch, pcc_dim) - PCC upper triangular vector
    Output:
        - pretrain mode: Reconstructed PCC vector (batch, pcc_dim)
        - finetune mode: Feature vector (batch, d_model)
    """

    def __init__(
        self,
        pcc_dim=19900,
        d_model=256,
        n_heads=8,
        n_layers=2,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()

        self.pcc_dim = pcc_dim
        self.d_model = d_model

        # Input Embedding: Maps the PCC vector into the embedding space
        self.input_embedding = nn.Linear(pcc_dim, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Pre-training Decoder: Reconstructs the PCC vector
        self.pretrain_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pcc_dim)
        )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, mode='pretrain'):
        # Input Embedding
        x = self.input_embedding(x) / math.sqrt(self.d_model)

        # Add dummy sequence dimension (batch, 1, d_model)
        x = x.unsqueeze(1)

        # Transformer Encoding
        x = self.transformer_encoder(x)

        # Remove sequence dimension (batch, d_model)
        x = x.squeeze(1)

        # Activation and Normalization
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)

        if mode == 'finetune':
            return x  # (batch, d_model)
        else:
            return self.pretrain_decoder(x)  # (batch, pcc_dim)

    def get_features(self, x):
        """Feature vector (for contrastive learning)"""
        return self.forward(x, mode='finetune')

    def load_pretrained(self, checkpoint_path, strict=True):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        if not strict:
            state_dict = {
                k: v for k, v in state_dict.items()
                if 'pretrain_decoder' not in k
            }

        self.load_state_dict(state_dict, strict=strict)
        print(f"Loaded pretrained weights from {checkpoint_path}")


class TransformerFCForPretrain(nn.Module):
    """Wrapper for TST2 pre-training: masking + reconstruction loss."""

    def __init__(self, transformer_fc):
        super().__init__()
        self.transformer = transformer_fc

    def forward(self, x, masked_x, mask):
        pred = self.transformer(masked_x, mode='pretrain')
        loss = nn.functional.mse_loss(pred[mask], x[mask], reduction='mean')
        return loss, pred


class MaskedMSELoss(nn.Module):
    """Masked MSE: only over masked positions."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred, y_true, mask):
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)
        return self.mse_loss(masked_pred, masked_true)


def create_transformer_fc(config=None):
    """
    Factory. Acepta 'dim_feedforward' o su alias 'd_ff'.
    """
    default_config = {
        'pcc_dim': 19900,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1
    }

    if config is not None:
        cfg = dict(config)
        if 'd_ff' in cfg and 'dim_feedforward' not in cfg:
            cfg['dim_feedforward'] = cfg.pop('d_ff')
        default_config.update(cfg)

    return TransformerFC(**default_config)


if __name__ == '__main__':
    torch.manual_seed(0)

    print("── TEST 1: factory defaults ─────────────────────────────────")
    model = create_transformer_fc()
    print(f"  ✓ params: {sum(p.numel() for p in model.parameters()):,}")

    print("── TEST 2: alias d_ff ───────────────────────────────────────")
    m2 = create_transformer_fc({'pcc_dim': 100, 'd_model': 32, 'n_layers': 2,
                                'n_heads': 4, 'd_ff': 64, 'dropout': 0.1})
    assert m2.d_model == 32
    print(f"  ✓ d_model={m2.d_model}")

    print("── TEST 3: forward pretrain ─────────────────────────────────")
    x = torch.randn(4, 19900)
    with torch.no_grad():
        out = model(x, mode='pretrain')
    assert out.shape == (4, 19900)
    print(f"  ✓ output {tuple(out.shape)}")

    print("── TEST 4: forward finetune ─────────────────────────────────")
    with torch.no_grad():
        feat = model(x, mode='finetune')
    assert feat.shape == (4, 256)
    print(f"  ✓ output {tuple(feat.shape)}")

    print("── TEST 5: wrapper + MaskedMSELoss ──────────────────────────")
    wrapper = TransformerFCForPretrain(model)
    mask = torch.rand(4, 19900) > 0.85
    masked_x = x.clone(); masked_x[mask] = 0.0
    loss, pred = wrapper(x, masked_x, mask)
    assert pred.shape == (4, 19900) and loss.item() >= 0
    print(f"  ✓ loss={loss.item():.4f}")

    print("── TEST 6: get_features ─────────────────────────────────────")
    with torch.no_grad():
        feats = model.get_features(x)
    assert feats.shape == (4, 256)
    print(f"  ✓ features {tuple(feats.shape)}")

    print("\n✅ Todos los tests de transformer_fc.py pasaron.")