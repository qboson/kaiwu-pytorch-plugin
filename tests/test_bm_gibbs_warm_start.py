"""Full Boltzmann machine Gibbs chains can resume from caller-owned samples."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


def test_split_sweeps_match_one_uninterrupted_chain():
    initial = torch.tensor([[1, 0, 1], [0, 1, 0]])
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(12)
        model = BoltzmannMachine(3, device="cpu").double()
        torch.default_generator.manual_seed(27)
        uninterrupted = model.gibbs_sample(7, initial_state=initial)
        torch.default_generator.manual_seed(27)
        partial = model.gibbs_sample(3, initial_state=initial)
        resumed = model.gibbs_sample(4, initial_state=partial)

    torch.testing.assert_close(resumed, uninterrupted)
    assert resumed.dtype == torch.float64
    torch.testing.assert_close(initial, torch.tensor([[1, 0, 1], [0, 1, 0]]))


def test_zero_sweeps_returns_a_copy_without_consuming_random_values():
    model = BoltzmannMachine(3, device="cpu")
    initial = torch.tensor([[0.0, 0.0, 1.0]])
    saved_rng = torch.random.get_rng_state()

    result = model.gibbs_sample(0, initial_state=initial)

    torch.testing.assert_close(torch.random.get_rng_state(), saved_rng)
    torch.testing.assert_close(result, initial)
    result.fill_(1)
    torch.testing.assert_close(initial, torch.tensor([[0.0, 0.0, 1.0]]))


@pytest.mark.parametrize("initial, kwargs, message", [
    (torch.ones(1, 2), {}, "shape"),
    (torch.tensor([[0.0, 0.5, 1.0]]), {}, "binary"),
    (torch.ones(1, 3), {"num_sample": 1}, "cannot be combined"),
])
def test_invalid_initial_states_are_rejected(initial, kwargs, message):
    with pytest.raises(ValueError, match=message):
        BoltzmannMachine(3, device="cpu").gibbs_sample(0, initial_state=initial, **kwargs)
