"""
Modelo Dual-Stream (Doble Flujo)
Modelo completo que integra TST1, TST2 y el módulo de fusión
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_ts import TransformerTS, create_transformer_ts
from .transformer_fc import TransformerFC, create_transformer_fc
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
        tst1_config=None,
        tst2_config=None,
        fusion_type='cross_attention',
        fusion_config=None,
        num_classes=2,
        dropout=0.1,
        mlp_dims=None,
    ):
        super().__init__()

        self.transformer_ts = create_transformer_ts(tst1_config)
        self.transformer_fc = create_transformer_fc(tst2_config)

        self.dim_ts = self.transformer_ts.emb_dim
        self.dim_fc = self.transformer_fc.d_model

        fusion_config = fusion_config or {}
        self.fusion = create_fusion_module(
            fusion_type, self.dim_ts, self.dim_fc, **fusion_config
        )
        self.fusion_type = fusion_type
        self.num_classes = num_classes

        fusion_dim = self.fusion.output_dim
        dims = mlp_dims or [fusion_dim // 2, fusion_dim // 4, num_classes]
        hidden_dims = dims[:-1]
        out_dim = dims[-1]

        layers = []
        prev = fusion_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(self, timeseries, pcc_vector, return_features=False, return_attention=False):
        """
        Args:
            timeseries: Serie temporal (batch, T, n_rois)
            pcc_vector: Vector PCC (batch, pcc_dim)
            return_features: Si devuelve características intermedias
            return_attention: Si devuelve pesos de atención

        Returns:
            logits: Logits de clasificación (batch, num_classes)
            features (opcional): Características fusionadas (batch, fusion_dim)
            attention_weights (opcional): Diccionario de pesos de atención
        """
        # Obtener características de TST1
        h_ts = self.transformer_ts(timeseries, mode='finetune')

        # Obtener características de TST2
        h_fc = self.transformer_fc(pcc_vector, mode='finetune')

        # Fusión
        if return_attention and hasattr(self.fusion, 'forward'):
            # Verificar si la fusión admite devolver atención
            if 'return_attention' in self.fusion.forward.__code__.co_varnames:
                fused, attention_weights = self.fusion(h_ts, h_fc, return_attention=True)
            else:
                fused = self.fusion(h_ts, h_fc)
                attention_weights = None
        else:
            fused = self.fusion(h_ts, h_fc)
            attention_weights = None

        # Clasificación
        logits = self.classifier(fused)

        result = [logits]
        if return_features:
            result.extend([fused, h_ts, h_fc])
        if return_attention and attention_weights is not None:
            result.append(attention_weights)

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

    def freeze_encoders(self):
        """Congelar ambos codificadores Transformer"""
        for param in self.transformer_ts.parameters():
            param.requires_grad = False
        for param in self.transformer_fc.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        """Descongelar ambos codificadores Transformer"""
        for param in self.transformer_ts.parameters():
            param.requires_grad = True
        for param in self.transformer_fc.parameters():
            param.requires_grad = True


class DualStreamModelSingleBranch(nn.Module):
    """
    Modelo de rama única (utilizado para experimentos de ablación)
    Solo utiliza TST1 o TST2
    """

    def __init__(
        self,
        branch='ts',
        tst_config=None,
        num_classes=2,
        dropout=0.1
    ):
        """
        Args:
            branch: Qué rama utilizar ('ts' o 'fc')
            tst_config: Configuración del Transformer
            num_classes: Número de clases para clasificación
            dropout: Ratio de Dropout
        """
        super().__init__()

        self.branch = branch

        if branch == 'ts':
            self.transformer = create_transformer_ts(tst_config)
            feature_dim = self.transformer.emb_dim
        elif branch == 'fc':
            self.transformer = create_transformer_fc(tst_config)
            feature_dim = self.transformer.d_model
        else:
            raise ValueError(f"Unknown branch: {branch}")

        # Cabezal de clasificación
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        """
        Args:
            x: Datos de entrada
               - rama ts: (batch, T, n_rois)
               - rama fc: (batch, pcc_dim)

        Returns:
            logits: Logits de clasificación (batch, num_classes)
        """
        features = self.transformer(x, mode='finetune')
        logits = self.classifier(features)
        return logits


def create_dual_stream_model(
    n_rois=200,
    time_points=100,
    pcc_dim=19900,
    tst1_emb_dim=512,
    tst2_d_model=256,
    fusion_type='cross_attention',
    fusion_hidden_dim=None,
    num_classes=2,
    dropout=0.1,
    mlp_dims=None,
):
    tst1_config = {
        'n_rois': n_rois, 'emb_dim': tst1_emb_dim,
        'n_heads': 8, 'n_layers': 6, 'dim_feedforward': 2048,
        'dropout': dropout, 'max_seq_len': time_points,
        'use_cls_token': True,
    }

    tst2_config = {
        'pcc_dim': pcc_dim, 'd_model': tst2_d_model,
        'n_heads': 8, 'n_layers': 2, 'dim_feedforward': 512,
        'dropout': dropout,
    }

    fusion_config = {}
    if fusion_type == "attention_pooling" and fusion_hidden_dim is not None:
        fusion_config["hidden_dim"] = fusion_hidden_dim

    return DualStreamModel(
        tst1_config=tst1_config,
        tst2_config=tst2_config,
        fusion_type=fusion_type,
        fusion_config=fusion_config,
        num_classes=num_classes,
        dropout=dropout,
        mlp_dims=mlp_dims,
    )

# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(0)

    B, T, R = 4, 100, 200
    D_PCC = 19900

    # ─── TEST 1: construcción con factory ─────────────────────────────
    print("── TEST 1: create_dual_stream_model ─────────────────────────")
    model = create_dual_stream_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ params: {n_params:,}")
    assert model.dim_ts == 512, model.dim_ts
    assert model.dim_fc == 256, model.dim_fc
    print(f"  ✓ dim_ts={model.dim_ts}  dim_fc={model.dim_fc}")
    print()

    # ─── TEST 2: forward base (solo logits) ───────────────────────────
    print("── TEST 2: forward básico ───────────────────────────────────")
    ts = torch.randn(B, T, R)
    pcc = torch.randn(B, D_PCC)
    logits = model(ts, pcc)
    assert logits.shape == (B, 2), logits.shape
    print(f"  ✓ logits {tuple(logits.shape)}")
    print()

    # ─── TEST 3: forward con return_features ──────────────────────────
    print("── TEST 3: forward con return_features ──────────────────────")
    logits, fused, h_ts, h_fc = model(ts, pcc, return_features=True)
    assert logits.shape == (B, 2)
    assert fused.shape == (B, model.fusion.output_dim)
    assert h_ts.shape == (B, 512)
    assert h_fc.shape == (B, 256)
    print(f"  ✓ logits {tuple(logits.shape)}  fused {tuple(fused.shape)}")
    print(f"  ✓ h_ts {tuple(h_ts.shape)}  h_fc {tuple(h_fc.shape)}")
    print()

    # ─── TEST 4: get_features (para contrastive) ──────────────────────
    print("── TEST 4: get_features ─────────────────────────────────────")
    h_ts, h_fc = model.get_features(ts, pcc)
    assert h_ts.shape == (B, 512)
    assert h_fc.shape == (B, 256)
    print(f"  ✓ h_ts {tuple(h_ts.shape)}  h_fc {tuple(h_fc.shape)}")
    print()

    # ─── TEST 5: todas las fusiones funcionan en el dual stream ───────
    print("── TEST 5: cada fusión en el modelo completo ────────────────")
    for ft in ['concat', 'gated', 'cross_attention', 'bilinear', 'attention_pooling']:
        m = create_dual_stream_model(fusion_type=ft)
        out = m(ts, pcc)
        assert out.shape == (B, 2), f"{ft}: {out.shape}"
        print(f"  ✓ {ft:>18s} → {tuple(out.shape)}")
    print()

    # ─── TEST 6: freeze / unfreeze encoders ───────────────────────────
    print("── TEST 6: freeze_encoders / unfreeze_encoders ──────────────")
    model.freeze_encoders()
    for p in model.transformer_ts.parameters():
        assert not p.requires_grad
    for p in model.transformer_fc.parameters():
        assert not p.requires_grad
    # El clasificador debe seguir entrenable
    assert all(p.requires_grad for p in model.classifier.parameters())
    print(f"  ✓ freeze: encoders congelados, classifier entrenable")

    model.unfreeze_encoders()
    for p in model.transformer_ts.parameters():
        assert p.requires_grad
    for p in model.transformer_fc.parameters():
        assert p.requires_grad
    print(f"  ✓ unfreeze: encoders entrenables de nuevo")
    print()

    # ─── TEST 7: single-branch models (ablación) ──────────────────────
    print("── TEST 7: DualStreamModelSingleBranch ──────────────────────")
    ts_branch = DualStreamModelSingleBranch(branch='ts')
    out = ts_branch(ts)
    assert out.shape == (B, 2)
    print(f"  ✓ TS-only → {tuple(out.shape)}")

    fc_branch = DualStreamModelSingleBranch(branch='fc')
    out = fc_branch(pcc)
    assert out.shape == (B, 2)
    print(f"  ✓ FC-only → {tuple(out.shape)}")

    try:
        DualStreamModelSingleBranch(branch='invalido')
        raise AssertionError("Debió lanzar ValueError")
    except ValueError:
        print(f"  ✓ branch inválido lanza ValueError")
    print()

    # ─── TEST 8: output_dim coincide con classifier[0].in_features ────
    print("── TEST 8: coherencia output_dim ↔ classifier ───────────────")
    for ft in ['concat', 'gated', 'cross_attention', 'bilinear', 'attention_pooling']:
        m = create_dual_stream_model(fusion_type=ft)
        assert m.classifier[0].in_features == m.fusion.output_dim
    print(f"  ✓ classifier conectado correctamente a la fusión")
    print()

    print("✅ Todos los tests de dual_stream.py pasaron.")