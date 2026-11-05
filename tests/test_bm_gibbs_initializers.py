"""gibbs_sample must reject passing both s_visible and num_sample."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


def test_both_initializers_are_rejected() -> None:
    bm = BoltzmannMachine(num_nodes=4)
    with pytest.raises(ValueError, match="mutually exclusive"):
        bm.gibbs_sample(1, s_visible=torch.zeros(1, 3), num_sample=50)


def test_visible_state_alone_works() -> None:
    samples = BoltzmannMachine(num_nodes=4).gibbs_sample(1, s_visible=torch.zeros(2, 3))
    assert samples.shape == (2, 4)


def test_sample_count_alone_works() -> None:
    samples = BoltzmannMachine(num_nodes=4).gibbs_sample(1, num_sample=3)
    assert samples.shape == (3, 4)
