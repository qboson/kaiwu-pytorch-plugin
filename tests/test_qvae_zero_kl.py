"""Zero KL weight avoids the solver while preserving reconstruction gradients."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from kaiwu.torch_plugin import BoltzmannMachine, QVAE


@pytest.mark.parametrize('loss_type', ['mse', 'bernoulli'])
def test_zero_weight_has_no_solver_dependency(loss_type):
    config = SimpleNamespace(num_latent_units=2, loss_type=loss_type,
                             dist_beta=2., kl_beta=0., weight_decay=.1)
    bm = BoltzmannMachine(2, device='cpu')
    sampler = Mock()
    sampler.solve.side_effect = RuntimeError('solver unavailable')
    model = QVAE(3, None, config, encoder=torch.nn.Linear(3, 2),
                 decoder=torch.nn.Linear(2, 3), bm=bm, sampler=sampler)
    x = torch.rand(4, 3)
    recon, posterior, q, _ = model(x)
    loss = model.loss(x, recon, posterior)
    expected = (torch.nn.functional.mse_loss(recon, x, reduction='sum') / len(x)
                if loss_type == 'mse' else
                torch.nn.functional.binary_cross_entropy_with_logits(recon, x, reduction='sum') / len(x))
    expected = expected + .1 * bm.quadratic_coef.square().sum() + .05 * bm.linear_bias.square().sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in list(model.encoder.parameters()) + list(model.decoder.parameters()))
    sampler.solve.assert_not_called()
    # Diagnostics and separate BM training still explicitly invoke the sampler.
    with pytest.raises(RuntimeError, match='solver unavailable'):
        model.loss(x, recon, posterior, diagnose_zero_weight_kl=True)
    with pytest.raises(RuntimeError, match='solver unavailable'):
        model.bm_loss(q)
    assert sampler.solve.call_count == 2
    model.kl_beta = .5
    with pytest.raises(RuntimeError, match='solver unavailable'):
        model.loss(x, recon, posterior)


def test_positive_weight_and_opt_in_diagnostic_values():
    config = SimpleNamespace(num_latent_units=2, loss_type='mse',
                             dist_beta=2., kl_beta=.3, weight_decay=0.)
    bm = BoltzmannMachine(2, device='cpu')
    model = QVAE(3, None, config, encoder=torch.nn.Linear(3, 2),
                 decoder=torch.nn.Linear(2, 3), bm=bm, sampler=object())
    negative = torch.tensor([[0., 1.], [1., 0.]])
    bm.sample = Mock(return_value=negative)
    x = torch.rand(4, 3)
    recon, posterior, q, _ = model(x)
    reconstruction = torch.nn.functional.mse_loss(recon, x, reduction='sum') / len(x)
    probabilities = torch.sigmoid(q)
    entropy = torch.distributions.Bernoulli(logits=q).entropy().sum(1).mean()
    surrogate = bm(probabilities).mean() - bm(negative).mean() - entropy
    torch.testing.assert_close(model.loss(x, recon, posterior), reconstruction + .3 * surrogate)
    bm.sample.assert_called_once()
    model.kl_beta = 0.
    torch.testing.assert_close(model.loss(x, recon, posterior, diagnose_zero_weight_kl=True), reconstruction)
    assert bm.sample.call_count == 2
