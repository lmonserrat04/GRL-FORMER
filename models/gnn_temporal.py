"""
GNN temporal para clasificación ASD/TD desde series BOLD (ABIDE I, CC200).

Sin LSTM ni GRU. La dinámica temporal se resuelve con dos piezas:
    (1) aristas dinámicas: e_uv(t) = z_u(t) * z_v(t), resumidas en una
        ventana deslizante por su MEDIA y su DESVIACIÓN TÍPICA;
    (2) atención temporal sobre los instantes (no recurrencia).

Pipeline por sujeto (x ∈ R^{T×R}, R=200):
    1. z-score por ROI sobre el tiempo  →  z (B, T, R)
    2. Grafo disperso k-NN por sujeto:
           score_uv = |mean_t z_u(t) z_v(t)|   (= |correlación|)
           top-k por nodo + aristas inversas   →  E = 2·R·k
    3. Arista dinámica:
           e_uv(t)   = z_u(t) · z_v(t)
           mu_uv(t)  = media de e_uv en ventana W centrada en t
           sig_uv(t) = desv. típica de e_uv en esa ventana
    4. GNN por paso temporal (pesos compartidos, sin recurrencia):
           h_u      = W_in · z_u(t)
           m_u      = Σ_{v∈N(u)} φ([h_v ‖ mu_uv ‖ sig_uv]) / deg_u
           h_u      ← LayerNorm(h_u + Dropout(m_u))
           h_u      ← LayerNorm(h_u + Dropout(ψ([h_u ‖ m_u])))
           (NUM_LAYERS veces)
    5. Readout por instante:  g_t = [mean_u h_u ‖ max_u h_u] ∈ R^{2d}
    6. Atención temporal:     alpha_t = softmax_t(wᵀ tanh(W g_t))
                              context = Σ_t alpha_t g_t
    7. Clasificador:          LayerNorm → Linear → GELU → Dropout → Linear(2)

El grafo se construye por sujeto (topología distinta por muestra) pero los
pesos de la GNN se comparten entre todos los instantes y sujetos.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphMessageLayer(nn.Module):
    """Una capa de paso de mensajes sobre un grafo disperso ya construido."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_model + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.psi = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_feats, lin_u, lin_v, deg):
        """
        h:           (N, d)  con N = B·T_chunk·R
        edge_feats:  (M, 2)  con M = B·T_chunk·E
        lin_u/lin_v: (M,)    índices planos receptor / emisor
        deg:         (N, 1)  grado de entrada por nodo
        """
        h_v = h[lin_v]                                      # (M, d)
        msg = self.phi(torch.cat([h_v, edge_feats], dim=-1))
        agg = torch.zeros_like(h).index_add_(0, lin_u, msg)
        agg = agg / deg

        h = self.norm1(h + self.drop(agg))
        upd = self.psi(torch.cat([h, agg], dim=-1))
        h = self.norm2(h + self.drop(upd))
        return h


class TemporalGNN(nn.Module):
    """Clasificador ASD/TD: grafo disperso dinámico + GNN + atención temporal."""

    def __init__(
        self,
        n_rois: int = 200,
        d_model: int = 32,
        knn: int = 8,
        window: int = 9,
        num_layers: int = 2,
        dropout: float = 0.3,
        stride: int = 2,
        time_chunk: int = 8,
        num_classes: int = 2,
    ):
        super().__init__()
        self.n_rois = int(n_rois)
        self.d_model = int(d_model)
        self.knn = int(knn)
        self.window = int(window) if int(window) % 2 == 1 else int(window) + 1
        self.stride = max(int(stride), 1)
        self.time_chunk = max(int(time_chunk), 1)

        self.node_in = nn.Linear(1, d_model)
        self.node_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [GraphMessageLayer(d_model, dropout) for _ in range(int(num_layers))]
        )
        self.readout_norm = nn.LayerNorm(2 * d_model)
        self.att_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    # ──────────────────────────────────────────────────────────────────
    # Construcción del grafo disperso
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _build_graph(self, z: torch.Tensor):
        """z: (B, T, R) ya estandarizado por ROI → (u_idx, v_idx) (B, E)."""
        b, t, r = z.shape
        zc = z - z.mean(dim=1, keepdim=True)
        sd = zc.std(dim=1, keepdim=True).clamp_min(1e-5)
        zn = zc / sd
        score = torch.einsum("btr,bts->brs", zn, zn).abs() / t
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score.diagonal(dim1=-2, dim2=-1).zero_()

        k = min(self.knn, r - 1)
        nbr = score.topk(k, dim=-1).indices                      # (B, R, k)
        node = torch.arange(r, device=z.device).view(1, r, 1).expand(b, r, k)

        u_idx = torch.cat([node, nbr], dim=-1).reshape(b, -1)    # receptor
        v_idx = torch.cat([nbr, node], dim=-1).reshape(b, -1)    # emisor
        return u_idx, v_idx

    # ──────────────────────────────────────────────────────────────────
    # Aristas dinámicas: e_uv(t) + media/desv en ventana
    # ──────────────────────────────────────────────────────────────────
    def _edge_features(self, z, u_idx, v_idx):
        """z: (B,T,R) → (B, T, E, 2) con [media, desv] en ventana."""
        b, t, _ = z.shape

        zu = z.gather(2, u_idx.unsqueeze(1).expand(-1, t, -1))   # (B,T,E)
        zv = z.gather(2, v_idx.unsqueeze(1).expand(-1, t, -1))
        e = (zu * zv).transpose(1, 2)                            # (B,E,T)

        pad = self.window // 2
        e_pad = F.pad(e, (pad, pad), mode="replicate")
        mean = F.avg_pool1d(e_pad, self.window, stride=1)
        sq = F.avg_pool1d(e_pad * e_pad, self.window, stride=1)
        var = (sq - mean * mean).clamp_min(1e-6)
        std = var.sqrt()                                         # (B,E,T)

        feats = torch.stack([mean, std], dim=-1)                 # (B,E,T,2)
        return feats.permute(0, 2, 1, 3).contiguous()            # (B,T,E,2)

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """x: (B, T, R) crudo o normalizado por sujeto → logits (B, 2)."""
        if x.dim() != 3:
            raise ValueError(f"Se espera (B, T, R); recibido {tuple(x.shape)}")
        b, t, r = x.shape
        if r != self.n_rois:
            raise ValueError(f"n_rois={self.n_rois}, pero la entrada tiene R={r}")

        # 1) Estandarización por ROI (necesaria para que z_u·z_v sea comparable)
        z = x - x.mean(dim=1, keepdim=True)
        z = z / z.std(dim=1, keepdim=True).clamp_min(1e-5)

        # 2) Grafo disperso k-NN por sujeto
        u_idx, v_idx = self._build_graph(z)                      # (B,E)
        e_count = u_idx.size(1)

        # 3) Aristas dinámicas con media y desv en ventana
        edge_feats_full = self._edge_features(z, u_idx, v_idx)   # (B,T,E,2)

        # 4) Instantes a evaluar (stride) y embedding de nodo inicial
        times = torch.arange(0, t, self.stride, device=x.device)
        te = times.numel()

        z_s = z.index_select(1, times)                           # (B,Te,R)
        h0 = self.node_norm(self.node_in(z_s.unsqueeze(-1)))     # (B,Te,R,d)
        edge_feats = edge_feats_full.index_select(1, times)      # (B,Te,E,2)

        g_steps = []
        for start in range(0, te, self.time_chunk):
            stop = min(start + self.time_chunk, te)
            tc = stop - start

            h = h0[:, start:stop].reshape(b * tc, r, self.d_model)
            ef = edge_feats[:, start:stop].reshape(b * tc, e_count, 2)

            # El batch plano del chunk es (b, t_chunk), así que cada muestra
            # aporta sus propias aristas en cada instante del chunk.
            u_chunk = u_idx.unsqueeze(1).expand(-1, tc, -1).reshape(b * tc, e_count)
            v_chunk = v_idx.unsqueeze(1).expand(-1, tc, -1).reshape(b * tc, e_count)
            off = torch.arange(b * tc, device=x.device).view(-1, 1) * r
            lin_u = (off + u_chunk).reshape(-1)
            lin_v = (off + v_chunk).reshape(-1)

            h_flat = h.reshape(-1, self.d_model)
            deg = torch.zeros(
                b * tc * r, 1, device=x.device, dtype=h_flat.dtype
            ).index_add_(
                0,
                lin_u,
                torch.ones(lin_u.numel(), 1, device=x.device, dtype=h_flat.dtype),
            ).clamp_min(1.0)

            for layer in self.layers:
                h_flat = layer(h_flat, ef.reshape(-1, 2), lin_u, lin_v, deg)

            h = h_flat.view(b * tc, r, self.d_model)

            # 5) Readout por instante: media + máximo sobre nodos
            g = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
            g_steps.append(g.view(b, tc, -1))

        g_all = torch.cat(g_steps, dim=1)                        # (B,Te,2d)
        g_all = self.readout_norm(g_all)

        # 6) Atención temporal sobre los instantes
        logits_att = self.att_proj(g_all).squeeze(-1)            # (B,Te)
        alpha = torch.softmax(logits_att, dim=1)
        context = (alpha.unsqueeze(-1) * g_all).sum(dim=1)       # (B,2d)

        # 7) Clasificador
        out = self.classifier(context)
        if return_attn:
            return out, alpha
        return out
