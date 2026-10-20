"""Optional DEQ diagnostics: Jacobian-regularization loss and spectral radius.

Adapted from this project's existing ``deq_gm/lib/jacobian.py`` (and
its ``deq_nlp`` copy). Not called by ``QDEQ`` — call these manually
on the ``(info["new_z"], info["z_star"])`` pair from a training-mode forward
pass if you want jacobian regularization or a spectral-radius estimate.

Call these *before* ``.backward()``, and fold their result into the loss you
then call ``.backward()`` on. ``new_z``'s autograd graph is freed once that
``.backward()`` runs (normal PyTorch behavior) — calling these functions
afterwards raises ``RuntimeError: Trying to backward through the graph a
second time``.
"""

from __future__ import annotations

import numpy as np
import torch


def jac_loss_estimate(f0: torch.Tensor, z0: torch.Tensor, vecs: int = 2, create_graph: bool = True) -> torch.Tensor:
    """Estimate tr(J^T J) = tr(J J^T) via the Hutchinson estimator.

    Args:
        f0: Output of the function f (whose Jacobian J is analyzed).
        z0: Input to f.
        vecs: Number of random Gaussian probe vectors.
        create_graph: Whether to build a backward graph (needed to train on this loss).

    Returns:
        A scalar tensor: the shape-normalized jacobian loss.
    """
    if vecs < 1:
        raise ValueError("vecs must be positive")
    result = 0
    for _ in range(vecs):
        v = torch.randn(*z0.shape).to(z0)
        vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
        result = result + vJ.norm() ** 2
    return result / vecs / np.prod(z0.shape)


def power_method(f0: torch.Tensor, z0: torch.Tensor, n_iters: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the spectral radius of J = ∂f0/∂z0 via power iteration on J^T.

    A diagnostic, not a convergence guarantee: a dominant real eigenvalue
    and a spectral gap are needed for ordinary power iteration to converge.
    The graph is retained so the caller can subsequently backpropagate loss.

    Args:
        f0: Output of the function f (whose Jacobian J is analyzed).
        z0: Input to f.
        n_iters: Number of power-method iterations.

    Returns:
        (dominant eigenvector of J^T, estimated absolute eigenvalue).
    """
    if n_iters < 1:
        raise ValueError("n_iters must be positive")
    evector = torch.randn_like(z0)
    bsz = evector.shape[0]
    evalue = None
    tiny = torch.finfo(z0.dtype).tiny
    for _ in range(n_iters):
        vTJ = torch.autograd.grad(f0, z0, evector, retain_graph=True, create_graph=False)[0]
        vector = evector.reshape(bsz, -1)
        product = vTJ.reshape(bsz, -1)
        evalue = (product * vector).sum(1, keepdim=True) / vector.square().sum(1, keepdim=True).clamp_min(tiny)
        evector = (product / product.norm(dim=1, keepdim=True).clamp_min(tiny)).reshape_as(z0)
    return evector, torch.abs(evalue)
