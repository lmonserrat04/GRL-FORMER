"""
Ponderación de la pérdida por sitio (site-only weighting).

Diferencia con `site_class_weights.py`:
  - site_class_weights: w(s, c) = (1/count(s,c))^power  → pondera por
    (sitio, clase), forzando contribución igualitaria de cada celda
    (sitio × clase).
  - site_weights (este módulo): w(s) = (1/n_s)^power  → pondera solo
    por sitio, forzando contribución igualitaria de cada sitio al
    gradiente, independientemente de su tamaño y de su composición
    de clases.

Fórmula:
    w(s) = (1 / n_s) ^ power

Normalización:
    - "mean":   media de w sobre sitios presentes = 1
    - "sum":    suma de w sobre sitios presentes = num_sites_presentes
    - "none":   sin normalizar

Uso:
    w = compute_site_weight_vector(
        site_ids=train_site_ids,
        num_sites=19,
        power=1.0,
        normalize="mean",
    )
    # w.shape = (19,)  → w[site_id] = weight
    # En el trainer:
    #   w_tensor = torch.from_numpy(w).to(device)
    #   sample_w = w_tensor[site_batch]   # (B,)
"""
from __future__ import annotations

import numpy as np


def compute_site_weight_vector(
    site_ids: np.ndarray,
    num_sites: int,
    power: float = 1.0,
    normalize: str = "mean",
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Devuelve w: (num_sites,) float32.

    Args:
        site_ids:  (N,) int con índice de sitio de cada muestra del train split.
        num_sites: nº total de sitios (len(site_to_idx)).
        power:     exponente. 0 → uniforme, 1 → inversa frecuencia.
        normalize: "mean" | "sum" | "none".

    Returns:
        w: (num_sites,) float32. Sitios ausentes en train → w=0.
    """
    site_ids = np.asarray(site_ids, dtype=np.int64)
    counts = np.bincount(site_ids, minlength=num_sites).astype(np.float64)

    w = np.zeros(num_sites, dtype=np.float64)
    present = counts > 0
    w[present] = (1.0 / counts[present]) ** float(power)

    if normalize == "mean":
        if present.sum() > 0:
            m = w[present].mean()
            if m > epsilon:
                w[present] = w[present] / m
    elif normalize == "sum":
        if present.sum() > 0:
            s = w[present].sum()
            if s > epsilon:
                w[present] = w[present] * (present.sum() / s)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"normalize desconocido: {normalize!r}")

    return w.astype(np.float32)


def summarize_site_weights(w: np.ndarray, site_names: list[str] | None = None) -> str:
    """String legible para debugging."""
    lines = ["=== Site weights ==="]
    for s in range(len(w)):
        name = site_names[s] if site_names and s < len(site_names) else f"site_{s:02d}"
        lines.append(f"  {name:<12s}  w={w[s]:.4f}")
    present = w[w > 0]
    if len(present) > 0:
        lines.append(f"  (n_present={len(present)}, "
                     f"sum={present.sum():.4f}, mean={present.mean():.4f})")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TEST 1: sitios con tamaños distintos ===")
    # 3 sitios: 100, 50, 10 muestras
    site_ids = np.array([0]*100 + [1]*50 + [2]*10, dtype=np.int64)
    w = compute_site_weight_vector(site_ids, num_sites=3, power=1.0, normalize="none")
    print(f"  w (sin normalizar): {w}")
    assert w[0] < w[1] < w[2], "sitios grandes deben tener MENOS peso (w=1/n)"
    assert np.isclose(w[0], 1/100)
    assert np.isclose(w[1], 1/50)
    assert np.isclose(w[2], 1/10)
    print("  ✓ w = 1/n_s")

    print("\n=== TEST 2: normalize='mean' → media de pesos presentes = 1 ===")
    w2 = compute_site_weight_vector(site_ids, num_sites=3, power=1.0, normalize="mean")
    print(f"  w (mean): {w2}")
    print(f"  media de sitios presentes: {w2[w2>0].mean():.6f}")
    assert np.isclose(w2[w2 > 0].mean(), 1.0, atol=1e-6)
    print("  ✓ media = 1")

    print("\n=== TEST 3: normalize='sum' → suma = num_sites_presentes ===")
    w3 = compute_site_weight_vector(site_ids, num_sites=3, power=1.0, normalize="sum")
    print(f"  w (sum): {w3}")
    print(f"  suma de sitios presentes: {w3[w3>0].sum():.6f}")
    assert np.isclose(w3[w3 > 0].sum(), 3.0, atol=1e-6)
    print("  ✓ suma = 3.0")

    print("\n=== TEST 4: power=0 → todos los sitios igual peso ===")
    w4 = compute_site_weight_vector(site_ids, num_sites=3, power=0.0, normalize="mean")
    print(f"  w (power=0): {w4}")
    assert np.allclose(w4, [1.0, 1.0, 1.0])
    print("  ✓ todos iguales")

    print("\n=== TEST 5: sitio ausente → w=0 ===")
    site_ids_missing = np.array([0]*10 + [1]*10, dtype=np.int64)  # site 2 ausente
    w5 = compute_site_weight_vector(site_ids_missing, num_sites=3, power=1.0, normalize="mean")
    print(f"  w (site 2 ausente): {w5}")
    assert w5[2] == 0.0, "sitio ausente debe tener peso 0"
    print("  ✓ sitio ausente = 0")

    print("\n=== TEST 6: cada sitio contribuye con el mismo total (w_s × n_s const) ===")
    w6 = compute_site_weight_vector(site_ids, num_sites=3, power=1.0, normalize="sum")
    counts = np.bincount(site_ids, minlength=3)
    per_site_total = w6 * counts
    print(f"  w_s * n_s (por sitio): {per_site_total}")
    print(f"  (todos deben ser iguales → cada sitio aporta el mismo total al gradiente)")
    # Todos los sitios con w>0 deben tener el mismo w_s * n_s
    present_totals = per_site_total[per_site_total > 0]
    assert np.allclose(present_totals, present_totals[0], atol=1e-5), \
        f"no son iguales: {present_totals}"
    print(f"  ✓ todos contribuyen con {present_totals[0]:.6f}")

    print("\n=== TEST 6b: normalize='none' → suma total ponderada = n_present ===")
    w6b = compute_site_weight_vector(site_ids, num_sites=3, power=1.0, normalize="none")
    per_site_total_6b = w6b * counts
    print(f"  w_s * n_s (por sitio): {per_site_total_6b}")
    print(f"  suma total: {per_site_total_6b.sum():.6f}  (debe = 3 = n_present)")
    assert np.isclose(per_site_total_6b.sum(), 3.0, atol=1e-5)
    assert np.allclose(per_site_total_6b, 1.0)
    print("  ✓ suma = n_present, cada sitio contribuye con 1")

    print("\n✅ Todos los tests de site_weights.py pasaron.")
