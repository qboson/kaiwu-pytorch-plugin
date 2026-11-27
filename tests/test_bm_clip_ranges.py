"""clip_parameters must reject ranges that cannot describe a clamp window."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


def test_inverted_range_is_rejected() -> None:
    bm = BoltzmannMachine(num_nodes=3)
    with pytest.raises(ValueError, match="must be ordered"):
        bm.clip_parameters((1.0, -1.0), (-1.0, 1.0))


def test_non_finite_range_is_rejected() -> None:
    rbm = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    with pytest.raises(ValueError, match="finite"):
        rbm.clip_parameters((-1.0, float("nan")), (-1.0, 1.0))


@pytest.mark.parametrize("bounds", [(), (0.0, 1.0, 2.0)])
def test_range_must_have_one_or_two_bounds(bounds) -> None:
    with pytest.raises(TypeError, match="must be a"):
        BoltzmannMachine(num_nodes=3).clip_parameters(bounds, (-1.0, 1.0))


def test_valid_range_still_clips_parameters() -> None:
    bm = BoltzmannMachine(num_nodes=3)
    bm.linear_bias.data.fill_(5.0)
    bm.clip_parameters((-1.0, 1.0), (-2.0, 2.0))
    assert torch.allclose(bm.linear_bias, torch.ones(3))

    single = BoltzmannMachine(num_nodes=3)
    single.linear_bias.data.fill_(-5.0)
    single.clip_parameters((0.0,), (-1.0, 1.0))
    assert torch.allclose(single.linear_bias, torch.zeros(3))
