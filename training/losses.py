"""
Funciones de pérdida.

- FocalLoss: focal loss (Lin et al. 2017) para clasificación multiclase.

Referencia:
    Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV.

Fórmula:
    L = -alpha_t * (1 - p_t)^gamma * log(p_t)

donde:
    p_t = softmax(logits)[target]
    gamma >= 0: parámetro de focusing.
        - gamma = 0  → CrossEntropy estándar
        - gamma > 0  → reduce el peso de ejemplos fáciles (p_t alta)
    alpha_t: peso por clase (opcional).
        - None       → sin peso por clase (alpha_t = 1)
        - float      → mismo peso para todas las clases
        - tensor(C,) → peso por clase, indexado por el target
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss multi-clase.

    Args:
        alpha:     None | float | tensor(C,)
        gamma:     float >= 0
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha=None, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma debe ser >= 0, recibido {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction debe ser 'mean'|'sum'|'none', "
                f"recibido {reduction!r}"
            )

        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (int, float)):
            self.register_buffer(
                "alpha", torch.tensor(float(alpha), dtype=torch.float32)
            )
        else:
            alpha_t = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha_t)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C)
            targets: (B,) int64

        Returns:
            loss escalar (mean/sum) o (B,) si reduction='none'.
        """
        # log_softmax numéricamente estable
        log_probs = F.log_softmax(logits, dim=1)               # (B, C)
        target_log_probs = log_probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)                                           # (B,)
        target_probs = torch.exp(target_log_probs)             # (B,) = p_t

        # focal factor: (1 - p_t)^gamma
        focal_factor = (1.0 - target_probs) ** self.gamma      # (B,)

        # alpha_t
        if self.alpha is None:
            alpha_t = 1.0
        elif self.alpha.ndim == 0:
            alpha_t = self.alpha.to(logits.device)
        else:
            alpha_t = self.alpha.to(logits.device)[targets]    # (B,)

        loss = -alpha_t * focal_factor * target_log_probs      # (B,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TEST 1: gamma=0, alpha=None → igual que CrossEntropy ===")
    torch.manual_seed(0)
    B, C = 16, 2
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))

    ce  = F.cross_entropy(logits, targets, reduction="mean")
    fl0 = FocalLoss(alpha=None, gamma=0.0, reduction="mean")(logits, targets)
    print(f"  CE:                {ce.item():.6f}")
    print(f"  FocalLoss(gamma=0): {fl0.item():.6f}")
    assert torch.allclose(ce, fl0, atol=1e-6), "gamma=0 debe igualar CE"
    print("  ✓ idénticos")

    print("\n=== TEST 2: gamma=2, alpha=None → focal loss estándar ===")
    fl2 = FocalLoss(alpha=None, gamma=2.0, reduction="mean")(logits, targets)
    # Calcular manualmente
    log_probs = F.log_softmax(logits, dim=1)
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = torch.exp(target_log_probs)
    manual = (-(1 - p_t) ** 2 * target_log_probs).mean()
    print(f"  FocalLoss(gamma=2): {fl2.item():.6f}")
    print(f"  manual:             {manual.item():.6f}")
    assert torch.allclose(fl2, manual, atol=1e-6)
    print("  ✓ coincide con fórmula manual")

    print("\n=== TEST 3: gamma=2, focal < CE para ejemplos fáciles ===")
    # Ejemplos "fáciles pero no triviales": p_t ≈ 0.9, no 0.99999997.
    # Con logits [2, 0] y target=0: p_0 = e^2 / (e^2 + 1) ≈ 0.881
    logits_easy = torch.tensor([[2.0, 0.0]] * 8)
    targets_easy = torch.zeros(8, dtype=torch.long)
    ce_easy = F.cross_entropy(logits_easy, targets_easy, reduction="mean")
    fl_easy = FocalLoss(alpha=None, gamma=2.0)(logits_easy, targets_easy)
    print(f"  Ejemplos fáciles — CE: {ce_easy.item():.6f}, "
          f"Focal: {fl_easy.item():.6f}")
    assert fl_easy < ce_easy, "focal debe reducir el peso de ejemplos fáciles"
    ratio = fl_easy / ce_easy
    print(f"  ratio focal/CE = {ratio.item():.4f} (< 1, focal menor)")
    print("  ✓ focal reduce la loss en ejemplos fáciles")

    print("\n=== TEST 4: alpha escalar escala la loss ===")
    fl_a1 = FocalLoss(alpha=1.0, gamma=2.0)(logits, targets)
    fl_a05 = FocalLoss(alpha=0.5, gamma=2.0)(logits, targets)
    print(f"  alpha=1.0: {fl_a1.item():.6f}")
    print(f"  alpha=0.5: {fl_a05.item():.6f}")
    assert torch.allclose(fl_a05, fl_a1 * 0.5, atol=1e-6)
    print("  ✓ alpha escala lineal")

    print("\n=== TEST 5: alpha tensor por clase ===")
    alpha_per_class = torch.tensor([0.25, 0.75])
    fl_pc = FocalLoss(alpha=alpha_per_class, gamma=2.0)(logits, targets)
    print(f"  alpha=[0.25, 0.75]: {fl_pc.item():.6f}")
    print("  ✓ sin error")

    print("\n=== TEST 6: reduction='none' devuelve vector (B,) ===")
    fl_none = FocalLoss(alpha=None, gamma=2.0, reduction="none")(logits, targets)
    print(f"  shape: {tuple(fl_none.shape)}")
    assert fl_none.shape == (B,)
    print("  ✓ per-sample loss")

    print("\n=== TEST 7: gradient flow ===")
    logits_g = logits.clone().requires_grad_(True)
    loss = FocalLoss(alpha=None, gamma=2.0)(logits_g, targets)
    loss.backward()
    assert logits_g.grad is not None and logits_g.grad.abs().sum() > 0
    print(f"  ✓ gradiente fluye, |grad|={logits_g.grad.abs().sum().item():.4f}")

    print("\n=== TEST 8: estabilidad numérica con logits extremos ===")
    logits_ext = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
    targets_ext = torch.tensor([0, 1])
    fl_ext = FocalLoss(alpha=None, gamma=2.0)(logits_ext, targets_ext)
    print(f"  loss con logits extremos: {fl_ext.item():.6e}")
    assert torch.isfinite(fl_ext), "loss debe ser finita (no inf/nan)"
    print("  ✓ numéricamente estable")

    print("\n✅ Todos los tests de FocalLoss pasaron.")
