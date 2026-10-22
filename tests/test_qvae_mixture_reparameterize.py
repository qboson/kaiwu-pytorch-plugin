"""MixtureGeneric must keep the implicit gradient in training mode."""

import torch

from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric


def test_training_mode_keeps_the_implicit_gradient():
    torch.manual_seed(0)
    param = torch.zeros(3, requires_grad=True)
    dist = MixtureGeneric(param, smoothing_dist_beta=0.5)

    zeta = dist.reparameterize(is_training=True)

    assert zeta.shape == (3,)
    assert torch.isfinite(zeta).all()
    assert zeta.requires_grad is True
    zeta.sum().backward()
    assert param.grad is not None
    assert torch.isfinite(param.grad).all()
