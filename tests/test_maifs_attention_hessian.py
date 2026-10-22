"""Mask Hessians support real SDPA without leaking backend configuration."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, TensorDataset
from kaiwu.torch_plugin import FeatureSelectionWrapper


class Attention(torch.nn.Module):
    def forward(self, x):
        q = x.unsqueeze(1)
        return F.scaled_dot_product_attention(q, q, q).squeeze(1)


def backend_flags():
    return (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled())


@pytest.mark.parametrize('mode', ['full', 'diagonal'])
def test_attention_hessian_matches_finite_difference(mode):
    torch.manual_seed(17)
    x = torch.randn(2, 3, 3, dtype=torch.float64) * .3
    y = torch.randn_like(x) * .2
    wrapper = FeatureSelectionWrapper(Attention(), 3).double()
    wrapper.train()
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    flags = backend_flags()
    gradient, hessian = wrapper.compute_mask_derivatives(loader, torch.nn.MSELoss(), mode)
    assert backend_flags() == flags
    assert wrapper.training
    assert wrapper.model.training
    torch.testing.assert_close(wrapper.mask, torch.ones(3, dtype=torch.float64))

    # Independent central differences of loss values, without autograd.
    def value(mask):
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            return F.mse_loss(Attention()(x * mask), y).item()

    mask = torch.ones(3, dtype=torch.float64)
    eps = 1e-3
    basis = torch.eye(3, dtype=torch.float64) * eps
    expected_g = np.array([(value(mask + d) - value(mask - d)) / (2 * eps) for d in basis])
    expected_h = np.array([[(value(mask + a + b) - value(mask + a - b)
                            - value(mask - a + b) + value(mask - a - b)) / (4 * eps**2)
                           for b in basis] for a in basis])
    if mode == 'diagonal':
        expected_h = np.diag(np.diag(expected_h))
    np.testing.assert_allclose(gradient, expected_g, rtol=2e-4, atol=1e-7)
    np.testing.assert_allclose(hessian, expected_h, rtol=2e-4, atol=1e-7)


def test_backend_and_parameter_flags_restore_on_failure():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), Attention())
    wrapper = FeatureSelectionWrapper(model, 3)
    x = torch.randn(2, 3, 3)
    flags = backend_flags()
    original = [p.requires_grad for p in model.parameters()]
    def fail(output, target):
        raise RuntimeError('loss failed')
    with pytest.raises(RuntimeError, match='loss failed'):
        wrapper.compute_mask_derivatives([(x, x)], fail)
    assert backend_flags() == flags
    assert [p.requires_grad for p in model.parameters()] == original
    assert wrapper.training and model.training
