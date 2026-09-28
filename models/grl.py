"""
Gradient Reversal Layer (Ganin & Lempitsky, 2015).

En forward es la identidad: y = x
En backward invierte el signo del gradiente y lo escala por lambda_:
    ∂y/∂x = -lambda_ · I

Se usa para domain adaptation adversarial: el domain_classifier
intenta minimizar la pérdida de dominio, mientras el extractor de
features (fusion + encoders) recibe gradiente invertido y por tanto
aprende representaciones invariantes al dominio.
"""
import torch


class _GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """
    Aplica Gradient Reversal.

    Args:
        x: tensor de entrada.
        lambda_: factor de escala del gradiente invertido.

    Returns:
        Tensor con el mismo valor que x en forward, pero cuyo backward
        propaga el gradiente multiplicado por -lambda_.
    """
    return _GradReverse.apply(x, lambda_)


import math


def ganin_lambda(progress: float, gamma: float = 10.0) -> float:
    """
    Schedule de Ganin & Lempitsky (2015).

        lambda(p) = 2 / (1 + exp(-gamma * p)) - 1

    Args:
        progress: p ∈ [0, 1], fracción de entrenamiento completada.
                  0 → lambda ≈ 0  (sin inversión al inicio)
                  1 → lambda ≈ 1  (inversión máxima)
        gamma: controla la pendiente. 10 es el valor del paper original.

    Returns:
        Escalar en [0, 1).
    """
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0