"""BoltzmannMachine bias getters must validate the requested slice width."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


def test_hidden_bias_rejects_counts_above_num_nodes() -> None:
    bm = BoltzmannMachine(num_nodes=5)
    with pytest.raises(ValueError, match="num_hidden must be between 0 and 5"):
        bm.hidden_bias(6)


def test_hidden_bias_rejects_negative_counts() -> None:
    bm = BoltzmannMachine(num_nodes=5)
    with pytest.raises(ValueError, match="num_hidden must be between 0 and 5"):
        bm.hidden_bias(-1)


def test_visible_bias_rejects_counts_above_num_nodes() -> None:
    bm = BoltzmannMachine(num_nodes=5)
    with pytest.raises(ValueError, match="num_visible must be between 0 and 5"):
        bm.visible_bias(7)


@pytest.mark.parametrize("call", ["hidden", "visible"])
def test_non_integer_counts_are_rejected(call) -> None:
    bm = BoltzmannMachine(num_nodes=5)
    with pytest.raises(TypeError, match="must be an integer"):
        if call == "hidden":
            bm.hidden_bias(2.5)
        else:
            bm.visible_bias(True)


def test_valid_counts_still_slice_the_bias() -> None:
    bm = BoltzmannMachine(num_nodes=5)
    bm.linear_bias.data.copy_(torch.arange(5, dtype=torch.float32))
    assert torch.equal(bm.visible_bias(3), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(bm.hidden_bias(2), torch.tensor([3.0, 4.0]))
    assert bm.visible_bias(0).numel() == 0
    assert bm.hidden_bias(5).numel() == 5
