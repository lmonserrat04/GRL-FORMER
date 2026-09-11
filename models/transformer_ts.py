"""
TST1: Temporal Transformer
Processes raw fMRI time series using an ROI-level masking strategy for pre-training.
"""

import math
import torch
import torch.nn as nn


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


class TransformerTS(nn.Module):
    """
    Temporal Transformer (TST1)

    Input: (batch, T, n_rois) - Time series data
    Output:
        - pretrain mode: Reconstructed full time series (batch, T, n_rois)
        - finetune mode: CLS token features (batch, emb_dim)
    """

    def __init__(
        self,
        n_rois=200,
        emb_dim=512,
        n_heads=8,
        n_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=200,
        use_cls_token=True
    ):
        super().__init__()

        self.n_rois = n_rois
        self.emb_dim = emb_dim
        self.use_cls_token = use_cls_token

        # Input Embedding Layer: Maps ROI features at each time point to the embedding space
        self.input_embedding = nn.Linear(n_rois, emb_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(
            emb_dim, max_len=max_seq_len + 1, dropout=dropout
        )

        # CLS token (used for classification tasks)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
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

        # Pre-training Decoder: Reconstructs the time series
        self.pretrain_decoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, n_rois)
        )

        # Layer Normalization
        self.norm = nn.LayerNorm(emb_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, mode='pretrain'):
        """
        Args:
            x: Tensor, shape (batch, T, n_rois)
            mode: 'pretrain' or 'finetune'

        Returns:
            pretrain mode: Reconstructed time series (batch, T, n_rois)
            finetune mode: CLS token features (batch, emb_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Input embedding
        x = self.input_embedding(x) * math.sqrt(self.emb_dim)

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
            output = self.pretrain_decoder(x)  # (batch, T, n_rois)
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


class TransformerTSForPretrain(nn.Module):
    """
    Wrapper class for TST1 pre-training
    Includes masking logic and loss calculation
    """

    def __init__(self, transformer_ts):
        super().__init__()
        self.transformer = transformer_ts

    def forward(self, x, masked_x, mask):
        """
        Args:
            x: Original time series (batch, T, n_rois)
            masked_x: Masked time series (batch, T, n_rois)
            mask: Mask locations (batch, T, n_rois)

        Returns:
            loss: Reconstruction loss
            pred: Predicted time series
        """
        # Forward pass
        pred = self.transformer(masked_x, mode='pretrain')

        # Calculate MSE loss for masked positions
        loss = nn.functional.mse_loss(
            pred[mask], x[mask], reduction='mean'
        )

        return loss, pred


def create_transformer_ts(config=None):
    """
    Factory function to create a TST1 model instance.

    Acepta dos vocabularios de config para ser compatible con todos los
    call-sites del pipeline (dual_stream usa 'emb_dim'/'dim_feedforward',
    train_pretrain_min usa 'd_model'/'d_ff').

    Args:
        config: Configuration dictionary; uses defaults if None

    Returns:
        model: TransformerTS instance
    """
    default_config = {
        'n_rois': 200,
        'emb_dim': 512,
        'n_heads': 8,
        'n_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_seq_len': 200,
        'use_cls_token': True
    }

    if config is not None:
        # Aceptar alias (d_model → emb_dim, d_ff → dim_feedforward)
        cfg = dict(config)
        if 'd_model' in cfg and 'emb_dim' not in cfg:
            cfg['emb_dim'] = cfg.pop('d_model')
        if 'd_ff' in cfg and 'dim_feedforward' not in cfg:
            cfg['dim_feedforward'] = cfg.pop('d_ff')
        default_config.update(cfg)

    return TransformerTS(**default_config)


if __name__ == '__main__':
    # ── Test con vocabulario estándar (emb_dim) ────────────────────────
    print("── TEST 1: factory con 'emb_dim'/'dim_feedforward' ──────────")
    model = create_transformer_ts()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ default params: {n_params:,}")

    # ── Test con vocabulario de train_pretrain_min (d_model/d_ff) ──────
    print("── TEST 2: factory con 'd_model'/'d_ff' (alias) ──────────────")
    model_alias = create_transformer_ts({
        'n_rois': 200, 'max_seq_len': 100,
        'd_model': 512, 'n_layers': 6, 'n_heads': 8,
        'd_ff': 2048, 'dropout': 0.1,
    })
    assert model_alias.emb_dim == 512, "alias d_model no aplicado"
    print(f"  ✓ emb_dim={model_alias.emb_dim}")

    # ── Forward pretrain ───────────────────────────────────────────────
    print("── TEST 3: forward pretrain ──────────────────────────────────")
    x = torch.randn(4, 100, 200)
    model.eval()
    with torch.no_grad():
        out = model(x, mode='pretrain')
    assert out.shape == (4, 100, 200), out.shape
    print(f"  ✓ pretrain output {tuple(out.shape)}")

    # ── Forward finetune ───────────────────────────────────────────────
    print("── TEST 4: forward finetune ──────────────────────────────────")
    with torch.no_grad():
        feat = model(x, mode='finetune')
    assert feat.shape == (4, 512), feat.shape
    print(f"  ✓ finetune output {tuple(feat.shape)}")

    # ── Masking + loss wrapper ─────────────────────────────────────────
    print("── TEST 5: TransformerTSForPretrain (mask + loss) ───────────")
    wrapper = TransformerTSForPretrain(model)
    mask = torch.rand(4, 100, 200) > 0.85
    masked_x = x.clone()
    masked_x[mask] = 0.0
    loss, pred = wrapper(x, masked_x, mask)
    assert pred.shape == (4, 100, 200)
    assert loss.item() >= 0.0
    print(f"  ✓ loss={loss.item():.4f}  pred={tuple(pred.shape)}")

    # ── get_features (para contrastive) ────────────────────────────────
    print("── TEST 6: get_features ──────────────────────────────────────")
    with torch.no_grad():
        feats = model.get_features(x)
    assert feats.shape == (4, 512)
    print(f"  ✓ features {tuple(feats.shape)}")

    print("\n✅ Todos los tests de transformer_ts.py pasaron.")