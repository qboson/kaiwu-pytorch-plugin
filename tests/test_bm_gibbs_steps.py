"""BoltzmannMachine.gibbs_sample must validate the step count."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


def test_negative_step_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="num_steps must be a non-negative integer"):
        BoltzmannMachine(num_nodes=4).gibbs_sample(-5, num_sample=2)


@pytest.mark.parametrize("num_steps", [1.5, True])
def test_non_integer_step_count_is_rejected(num_steps) -> None:
    with pytest.raises(TypeError, match="num_steps must be a non-negative integer"):
        BoltzmannMachine(num_nodes=4).gibbs_sample(num_steps, num_sample=2)


def test_zero_steps_returns_the_initial_state() -> None:
    samples = BoltzmannMachine(num_nodes=4).gibbs_sample(0, num_sample=3)
    assert samples.shape == (3, 4)


def test_empty_visible_batch_still_returns_empty_samples() -> None:
    samples = BoltzmannMachine(num_nodes=4).gibbs_sample(
        2, s_visible=torch.zeros(0, 3)
    )
    assert samples.shape == (0, 4)
