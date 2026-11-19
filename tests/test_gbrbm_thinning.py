"""Thinning changes which samples are retained while keeping the chain intact."""

import pytest
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


@pytest.mark.parametrize("n_burnin", [0, 2])
def test_thinned_chain_is_the_expected_subsequence(n_burnin):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(17)
        model = GaussianBernoulliRestrictedBoltzmannMachine(2, 3, device="cpu")
        initial = torch.ones((2, 2))
        torch.default_generator.manual_seed(125)
        all_steps = model.gibbs_sample(8, n_burnin=n_burnin, s_gaussian=initial)
        final_rng = torch.random.get_rng_state()
        torch.default_generator.manual_seed(125)
        sparse_steps = model.gibbs_sample(8, n_burnin=n_burnin, s_gaussian=initial, thin=3)
        torch.testing.assert_close(torch.random.get_rng_state(), final_rng)

    full = all_steps.reshape(-1, initial.shape[0], model.num_nodes)
    torch.testing.assert_close(sparse_steps, full[::3].flatten(0, 1))


@pytest.mark.parametrize("thin", [0, -1, 1.5, True])
def test_invalid_thinning_is_rejected(thin):
    model = GaussianBernoulliRestrictedBoltzmannMachine(2, 3, device="cpu")
    with pytest.raises(ValueError, match="thin must be a positive integer"):
        model.gibbs_sample(5, n_sample=2, thin=thin)
