"""
Modelo Dual-Stream (Doble Flujo)
Modelo completo que integra TST1, TST2 y el módulo de fusión.

Soporta dos modos de domain adversarial training:

  - Modo single (multilayer=False):
      domain_classifier(GRL(fused))

  - Modo multi-layer (multilayer=True):
      domain_classifier_ts(GRL(h_ts))
      domain_classifier_fc(GRL(h_fc))
      domain_classifier_fused(GRL(fused))
      (tres GRLs independientes, tres domain classifiers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_ts import TransformerTS, create_transformer_ts
from .transformer_fc import TransformerFC, create_transformer_fc
from .fusion import create_fusion_module
from .mlp_head import create_mlp_head
from .grl import grad_reverse


class DualStreamModel(nn.Module):
    """
    Modelo de pre-entrenamiento auto-supervisado de doble flujo

    Contiene:
    - TST1: Transformer temporal, procesa series temporales de fMRI
    - TST2: Transformer de conectividad, procesa vectores PCC
    - Módulo de fusión: Fusiona las características de ambos Transformers
    - Cabezal de clasificación: Clasificador MLP
    - Domain classifiers: 1 o 3 según modo
    """

    def __init__(
        self,
        tst1_config=None,
        tst2_config=None,
        fusion_type='cross_attention',
        fusion_config=None,
        num_classes=2,
        num_domains=20,
        dropout=0.1,
        mlp_dims=None,
        proj_head_1=None,
        proj_head_2=None,
        grl_lambda=1.0,
        domain_weight=1.0,
        multilayer=False,
        grl_stream_hidden_dims=None,
    ):
        """
        Args:
            proj_head_1: Projection head entrenada en contrastive (TST1).
            proj_head_2: Idem para TST2.
            multilayer: si True, usa 3 domain classifiers (ts, fc, fused) con GRL
                        en cada stream. Si False, comportamiento original.
        """
        super().__init__()

        self.transformer_ts = create_transformer_ts(tst1_config)
        self.transformer_fc = create_transformer_fc(tst2_config)

        self.dim_ts = self.transformer_ts.emb_dim
        self.dim_fc = self.transformer_fc.d_model

        self.proj_head_1 = proj_head_1
        self.proj_head_2 = proj_head_2

        if proj_head_1 is not None and proj_head_2 is not None:
            dim_ts_fusion = proj_head_1.net[-1].out_features
            dim_fc_fusion = proj_head_2.net[-1].out_features
        else:
            dim_ts_fusion = self.dim_ts
            dim_fc_fusion = self.dim_fc

        fusion_config = fusion_config or {}
        self.fusion = create_fusion_module(
            fusion_type, dim_ts_fusion, dim_fc_fusion, **fusion_config
        )
        self.fusion_type = fusion_type
        self.num_classes = num_classes

        self.fusion_dim = self.fusion.output_dim
        self.dim_ts_fusion = dim_ts_fusion
        self.dim_fc_fusion = dim_fc_fusion

        self.grl_lambda: float = float(grl_lambda)
        self._domain_weight = float(domain_weight)
        self.multilayer = bool(multilayer)
        self._grl_stream_hidden_dims = grl_stream_hidden_dims

        # ─── Tag classifier ────────────────────────────────────────
        mlp_dims = mlp_dims or [self.fusion_dim // 2, self.fusion_dim // 4, num_classes]
        hidden = list(mlp_dims)[:-1]
        self.tag_classifier = create_mlp_head(
            [self.fusion_dim] + hidden + [num_classes],
            dropout,
            act_name="gelu",
        )

        # ─── Domain classifiers ────────────────────────────────────
        if self.multilayer:
            # Capas ocultas para los domain classifiers de los streams.
            # Si grl_stream_hidden_dims is None → usa hidden completo.
            # Si es [] → sin capas ocultas.
            # Si es [64] → una capa oculta de 64.
            if grl_stream_hidden_dims is None:
                stream_hidden = list(hidden)
            else:
                stream_hidden = list(grl_stream_hidden_dims)

            self.domain_classifier_ts = create_mlp_head(
                [dim_ts_fusion] + stream_hidden + [num_domains],
                dropout, act_name="relu",
            )
            self.domain_classifier_fc = create_mlp_head(
                [dim_fc_fusion] + stream_hidden + [num_domains],
                dropout, act_name="relu",
            )
            self.domain_classifier = create_mlp_head(
                [self.fusion_dim] + hidden + [num_domains],
                dropout, act_name="relu",
            )
        else:
            self.domain_classifier = create_mlp_head(
                [self.fusion_dim] + hidden + [num_domains],
                dropout, act_name="relu",
            )

    def forward(
        self,
        timeseries,
        pcc_vector,
        *,
        return_domain_logits: bool = False,
        return_features=False,
        return_attention=False,
    ):
        h_ts = self.transformer_ts(timeseries, mode='finetune')
        h_fc = self.transformer_fc(pcc_vector, mode='finetune')

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

        # Tag classifier (sin GRL)
        tag_logits = self.tag_classifier(fused)

        # Domain classifiers (con GRL)
        if self.multilayer:
            dom_ts = self.domain_classifier_ts(
                grad_reverse(h_ts, lambda_=self.grl_lambda)
            )
            dom_fc = self.domain_classifier_fc(
                grad_reverse(h_fc, lambda_=self.grl_lambda)
            )
            dom_fused = self.domain_classifier(
                grad_reverse(fused, lambda_=self.grl_lambda)
            )
        else:
            dom_ts = None
            dom_fc = None
            dom_fused = self.domain_classifier(
                grad_reverse(fused, lambda_=self.grl_lambda)
            )

        if return_domain_logits:
            if self.multilayer:
                return tag_logits, dom_ts, dom_fc, dom_fused
            return tag_logits, dom_fused

        return tag_logits

    def get_features(self, timeseries, pcc_vector):
        h_ts = self.transformer_ts(timeseries, mode='finetune')
        h_fc = self.transformer_fc(pcc_vector, mode='finetune')
        return h_ts, h_fc

    def load_pretrained_tst1(self, checkpoint_path, strict=False):
        self.transformer_ts.load_pretrained(checkpoint_path, strict=strict)

    def load_pretrained_tst2(self, checkpoint_path, strict=False):
        self.transformer_fc.load_pretrained(checkpoint_path, strict=strict)

    def freeze_encoders(self):
        for param in self.transformer_ts.parameters():
            param.requires_grad = False
        for param in self.transformer_fc.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        for param in self.transformer_ts.parameters():
            param.requires_grad = True
        for param in self.transformer_fc.parameters():
            param.requires_grad = True


class DualStreamModelSingleBranch(nn.Module):
    """Modelo de rama única (ablación)."""

    def __init__(self, branch='ts', tst_config=None, num_classes=2, dropout=0.1):
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

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x):
        features = self.transformer(x, mode='finetune')
        return self.classifier(features)


def create_dual_stream_model(
    tst1_config: dict,
    tst2_config: dict,
    fusion_type: str = "attention_pooling",
    fusion_hidden_dim: int | None = None,
    num_classes: int = 2,
    num_domains: int = 20,
    dropout: float = 0.1,
    mlp_dims: list | None = None,
    proj_head_1=None,
    proj_head_2=None,
    grl_lambda: float = 1.0,
    domain_weight: float = 1.0,
    multilayer: bool = False,
    grl_stream_hidden_dims=None,
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
        num_domains=num_domains,
        dropout=dropout,
        mlp_dims=mlp_dims,
        proj_head_1=proj_head_1,
        proj_head_2=proj_head_2,
        grl_lambda=grl_lambda,
        domain_weight=domain_weight,
        multilayer=multilayer,
        grl_stream_hidden_dims=grl_stream_hidden_dims,
    )
