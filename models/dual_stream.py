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
from .mlp_head import create_mlp_head



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
        proj_head_1=None,
        proj_head_2=None,
    ):
        """
        Args:
            proj_head_1: Projection head entrenada en contrastive (TST1).
                         Si se pasa, la fusión opera sobre z (128), no sobre h_ts.
            proj_head_2: Idem para TST2.
        """
        super().__init__()

        self.transformer_ts = create_transformer_ts(tst1_config)
        self.transformer_fc = create_transformer_fc(tst2_config)

        self.dim_ts = self.transformer_ts.emb_dim   # 512
        self.dim_fc = self.transformer_fc.d_model   # 256

        # Projection heads opcionales
        self.proj_head_1 = proj_head_1
        self.proj_head_2 = proj_head_2

        # La fusión opera sobre z si hay projections
        if proj_head_1 is not None and proj_head_2 is not None:
            dim_ts_fusion = proj_head_1.net[-1].out_features   # 128
            dim_fc_fusion = proj_head_2.net[-1].out_features   # 128
        else:
            dim_ts_fusion = self.dim_ts
            dim_fc_fusion = self.dim_fc

        fusion_config = fusion_config or {}
        self.fusion = create_fusion_module(
            fusion_type, dim_ts_fusion, dim_fc_fusion, **fusion_config
        )
        self.fusion_type = fusion_type
        self.num_classes = num_classes

        fusion_dim = self.fusion.output_dim
        
        self.classifier = create_mlp_head([fusion_dim] + list(mlp_dims) , dropout, act_name= 'gelu')

    def forward(self, timeseries, pcc_vector, return_features=False, return_attention=False):
        h_ts = self.transformer_ts(timeseries, mode='finetune')
        h_fc = self.transformer_fc(pcc_vector, mode='finetune')

        # Aplicar projections si están presentes (fine-tuning con projections del paper)
        if self.proj_head_1 is not None:
            h_ts = self.proj_head_1(h_ts)
        if self.proj_head_2 is not None:
            h_fc = self.proj_head_2(h_fc)

        # Fusión
        if return_attention and hasattr(self.fusion, 'forward'):
            if 'return_attention' in self.fusion.forward.__code__.co_varnames:
                fused, attention_weights = self.fusion(h_ts, h_fc, return_attention=True)
            else:
                fused = self.fusion(h_ts, h_fc)
                attention_weights = None
        else:
            fused = self.fusion(h_ts, h_fc)
            attention_weights = None

        logits = self.classifier(fused)

        result = [logits]
        if return_features:
            result.extend([fused, h_ts, h_fc])
        if return_attention and attention_weights is not None:
            result.append(attention_weights)

        if len(result) == 1:
            return result[0]
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
    tst1_config: dict,
    tst2_config: dict,
    fusion_type: str = 'attention_pooling',
    fusion_hidden_dim: int | None = None,
    num_classes: int = 2,
    dropout: float = 0.1,
    mlp_dims: list | None = None,
    proj_head_1=None,
    proj_head_2=None,
):
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
        proj_head_1=proj_head_1,
        proj_head_2=proj_head_2,
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