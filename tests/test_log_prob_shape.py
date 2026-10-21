"""log_prob_per_var must validate the sample matrix shape."""

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import FactorialBernoulliUtil, MixtureGeneric


def _factorial() -> FactorialBernoulliUtil:
    return FactorialBernoulliUtil(torch.zeros(2, 3))


def _mixture() -> MixtureGeneric:
    return MixtureGeneric(torch.zeros(2, 3), 1.0)


@pytest.mark.parametrize("factory", [_factorial, _mixture])
def test_one_dimensional_samples_are_rejected(factory) -> None:
    with pytest.raises(ValueError, match=r"must have shape \(num_samples, 3\)"):
        factory().log_prob_per_var(torch.zeros(3))


@pytest.mark.parametrize("factory", [_factorial, _mixture])
def test_wrong_width_is_rejected(factory) -> None:
    with pytest.raises(ValueError, match=r"must have shape \(num_samples, 3\)"):
        factory().log_prob_per_var(torch.zeros(2, 4))


@pytest.mark.parametrize("factory", [_factorial, _mixture])
def test_valid_shapes_still_work(factory) -> None:
    assert factory().log_prob_per_var(torch.zeros(2, 3)).shape == (2, 3)


def test_empty_batch_with_matching_parameters_works() -> None:
    empty = torch.zeros(0, 3)
    assert FactorialBernoulliUtil(empty).log_prob_per_var(empty).shape == (0, 3)
    assert MixtureGeneric(empty, 1.0).log_prob_per_var(empty).shape == (0, 3)


def test_one_dimensional_parameters_still_broadcast_over_the_batch() -> None:
    log_prob = FactorialBernoulliUtil(torch.zeros(3)).log_prob_per_var(
        torch.zeros(2, 3)
    )
    assert log_prob.shape == (2, 3)
