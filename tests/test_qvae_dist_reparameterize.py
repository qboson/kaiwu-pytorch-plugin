"""FactorialBernoulliUtil.reparameterize must reject training mode."""

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import FactorialBernoulliUtil


def test_training_mode_is_rejected():
    with pytest.raises(NotImplementedError, match="not differentiable"):
        FactorialBernoulliUtil(torch.zeros(4)).reparameterize(is_training=True)


def test_test_mode_samples_binary_values():
    torch.manual_seed(0)

    samples = FactorialBernoulliUtil(torch.zeros(4)).reparameterize(is_training=False)

    assert samples.shape == (4,)
    assert torch.isin(samples, torch.tensor([0.0, 1.0])).all()
    assert samples.requires_grad is False
