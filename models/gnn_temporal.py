"""
GNN temporal para clasificación ASD/TD desde series BOLD (ABIDE I, CC200).

Sin LSTM, sin GRU y sin transformer. La dinámica temporal se resuelve
procesando el grafo timestep a timestep con pesos compartidos:

    1. z-score por ROI sobre el tiempo  ->  z (B, T, R)
    2. Para cada timestep t, grafo disperso k-NN dinámico:
           e_uv(t)  = z_u(t) · z_v(t)
           score_uv = |e_uv(t)|
           top-k por nodo + aristas inversas   ->  E = 2·R·k
    3. Cada arista lleva directamente e_uv(t). No hay ventana, media,
       desviación ni correlación temporal: ventana = 1.
    4. Para cada timestep t, una GNN de pesos compartidos actualiza:
           h_u(t) = W_in · z_u(t)
           m_u(t) = Σ_{v∈N_t(u)} phi([h_v(t) || e_uv(t)]) / deg_u
           h_u(t) ← LayerNorm(h_u(t) + Dropout(m_u(t)))
           h_u(t) ← LayerNorm(h_u(t) + Dropout(psi([h_u(t) || m_u(t)])))
    5. Selección temporal por nodo:
           H_u = max_t h_u(t)          ->  (B, R, d)
       Los timesteps se procesan en bloques independientes para aprovechar la
       GPU. Cada timestep conserva su propio grafo y no hay mensajes entre
       timesteps; el resultado es idéntico a recorrerlos uno a uno.
    6. Atención sobre nodos (ROIs):
           alpha_u = softmax_u(w^T tanh(W H_u))
           h_G = Σ_u alpha_u · H_u     ->  (B, d)
    7. Clasificador de sujeto:
           LayerNorm(d) → Linear(d, 32) → GELU → Dropout(0.5)
                        → Linear(32, 16) → GELU → Dropout(0.5)
                        → Linear(16, 2)

El grafo se construye por sujeto y por timestep, pero los pesos de la GNN y
del clasificador se comparten entre sujetos y timesteps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphMessageLayer(nn.Module):
    """Una capa de paso de mensajes sobre un grafo disperso ya construido."""

    def __init__(self, d_model: int, edge_dim: int, dropout: float):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_model + edge_dim, d_model),
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
        h:           (N, d)  con N = B·R
        edge_feats:  (M, edge_dim)  con M = B·E
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


class GraphTemporalEncoder(nn.Module):
    """Convierte una serie BOLD (B,T,R) en un vector por ROI (B,R,d).

    En cada timestep construye un grafo k-NN con los productos instantáneos
    |z_u(t)·z_v(t)|, aplica la misma GNN y al final se queda con el máximo
    temporal de cada nodo/ROI. No hay atención temporal ni recurrencia.
    """

    def __init__(
        self,
        n_rois: int = 200,
        d_model: int = 16,
        knn: int = 5,
        num_layers: int = 2,
        dropout: float = 0.3,
        stride: int = 1,
        edge_dim: int = 1,
        time_chunk: int = 16,
    ):
        super().__init__()
        self.n_rois = int(n_rois)
        self.d_model = int(d_model)
        self.knn = int(knn)
        self.stride = max(1, int(stride))
        self.edge_dim = int(edge_dim)
        self.time_chunk = max(1, int(time_chunk))

        self.node_in = nn.Linear(1, d_model)
        self.node_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [
                GraphMessageLayer(d_model, self.edge_dim, dropout)
                for _ in range(int(num_layers))
            ]
        )

    @torch.no_grad()
    def _build_graph(self, z_t: torch.Tensor):
        """z_t: (B,R) → (u_idx, v_idx) (B,E) para este timestep."""
        b, r = z_t.shape
        score = torch.einsum("br,bs->brs", z_t, z_t).abs()
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score.diagonal(dim1=-2, dim2=-1).zero_()

        k = max(1, min(self.knn, r - 1))
        nbr = score.topk(k, dim=-1, sorted=False).indices        # (B,R,k)
        node = torch.arange(r, device=z_t.device).view(1, r, 1).expand(b, r, k)

        u_idx = torch.cat([node, nbr], dim=-1).reshape(b, -1)    # receptor
        v_idx = torch.cat([nbr, node], dim=-1).reshape(b, -1)    # emisor
        return u_idx, v_idx

    @staticmethod
    def _edge_values(z_t, u_idx, v_idx):
        """e_uv(t) = z_u(t)·z_v(t), sin media ni ventana: (B,E,1)."""
        zu = z_t.gather(1, u_idx)                                # (B,E)
        zv = z_t.gather(1, v_idx)
        return (zu * zv).unsqueeze(-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B,T,R) estandarizado → H: (B,R,d) con máximo temporal."""
        b, t, r = z.shape
        if t < 1:
            raise ValueError("La serie debe tener al menos un timestep")
        if r != self.n_rois:
            raise ValueError(f"n_rois={self.n_rois}, pero la entrada tiene R={r}")

        times = torch.arange(0, t, self.stride, device=z.device)
        te = times.numel()

        # Embedding inicial de cada ROI en cada timestep.
        h0 = self.node_norm(self.node_in(z.unsqueeze(-1)))       # (B,T,R,d)
        h0 = h0.index_select(1, times)                           # (B,Te,R,d)

        best: torch.Tensor | None = None

        # Bloques de timesteps independientes: no hay mensajes entre tiempos.
        # Cada timestep conserva su propio grafo dinámico y sus pesos compartidos.
        for start in range(0, te, self.time_chunk):
            stop = min(start + self.time_chunk, te)
            chunk = stop - start
            z_c = z.index_select(1, times[start:stop])           # (B,C,R)
            z_f = z_c.reshape(b * chunk, r)                      # (B·C,R)

            u_idx, v_idx = self._build_graph(z_f)                # (B·C,E)
            e = self._edge_values(z_f, u_idx, v_idx)             # (B·C,E,1)

            n = b * chunk
            off = (torch.arange(n, device=z.device) * r).view(n, 1)
            lin_u = (off + u_idx).reshape(-1)
            lin_v = (off + v_idx).reshape(-1)
            deg = torch.zeros(n * r, 1, device=z.device, dtype=z.dtype)
            deg.index_add_(
                0, lin_u,
                torch.ones(lin_u.numel(), 1, device=z.device, dtype=z.dtype),
            )
            deg.clamp_min_(1.0)

            h = h0[:, start:stop].reshape(n * r, self.d_model)
            ef = e.reshape(n * e.size(1), self.edge_dim)
            for layer in self.layers:
                h = layer(h, ef, lin_u, lin_v, deg)
            h = h.view(b, chunk, r, self.d_model).amax(dim=1)    # (B,R,d)

            best = h if best is None else torch.maximum(best, h)

        return best


class NodeAttentionClassifier(nn.Module):
    """Clasificador de sujeto con atención aprendida sobre las 200 ROIs.

    Cada ROI u recibe un peso alpha_u (softmax sobre las 200 ROIs) y el
    vector del sujeto es h_G = sum_u alpha_u * H_u.
    """

    def __init__(
        self,
        d_model: int = 16,
        hidden_dims: tuple[int, ...] = (32, 16),
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        layers: list[nn.Module] = [nn.LayerNorm(d_model)]
        prev = d_model
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, int(hidden)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = int(hidden)
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, nodes: torch.Tensor, return_attn: bool = False):
        """
        nodes: (B, R, d)
        returns: logits (B, num_classes), y opcionalmente alpha (B, R).
        """
        scores = self.att(nodes).squeeze(-1)                     # (B,R)
        alpha = torch.softmax(scores, dim=1)                     # suma 1 por sujeto
        subject = (alpha.unsqueeze(-1) * nodes).sum(dim=1)       # (B,d)
        logits = self.classifier(subject)

        if return_attn:
            return logits, alpha
        return logits


class TemporalGNN(nn.Module):
    """Encoder de grafo temporal + clasificador con atención sobre ROIs."""

    def __init__(
        self,
        n_rois: int = 200,
        d_model: int = 16,
        knn: int = 5,
        num_layers: int = 2,
        dropout: float = 0.3,
        stride: int = 1,
        time_chunk: int = 16,
        clf_hidden: tuple[int, ...] = (32, 16),
        clf_dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        self.n_rois = int(n_rois)
        self.d_model = int(d_model)
        self.knn = int(knn)
        self.stride = max(1, int(stride))
        self.time_chunk = max(1, int(time_chunk))

        self.encoder = GraphTemporalEncoder(
            n_rois=n_rois,
            d_model=d_model,
            knn=knn,
            num_layers=num_layers,
            dropout=dropout,
            stride=stride,
            time_chunk=time_chunk,
        )
        self.classifier = NodeAttentionClassifier(
            d_model=d_model,
            hidden_dims=tuple(int(x) for x in clf_hidden),
            dropout=clf_dropout,
            num_classes=num_classes,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,R) crudo → H: (B,R,d) con máximo temporal por ROI."""
        if x.dim() != 3:
            raise ValueError(f"Se espera (B, T, R); recibido {tuple(x.shape)}")
        if x.size(2) != self.n_rois:
            raise ValueError(f"n_rois={self.n_rois}, pero la entrada tiene R={x.size(2)}")

        # Estandarización por ROI sobre el tiempo.
        z = x - x.mean(dim=1, keepdim=True)
        z = z / z.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return self.encoder(z)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """x: (B,T,R) → logits (B,2); opcionalmente alpha (B,R)."""
        nodes = self.encode(x)
        return self.classifier(nodes, return_attn=return_attn)
