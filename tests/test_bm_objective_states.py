"""BoltzmannMachine.objective must validate its two state batches."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


def test_empty_batches_are_rejected() -> None:
    bm = BoltzmannMachine(num_nodes=5)
    with pytest.raises(ValueError, match="must contain at least one sample"):
        bm.objective(torch.zeros(0, 5), torch.zeros(0, 5))


def test_one_dimensional_states_are_rejected() -> None:
    bm = BoltzmannMachine(num_nodes=5)
    with pytest.raises(ValueError, match=r"must have shape \(batch, 5\)"):
        bm.objective(torch.zeros(5), torch.zeros(5))


def test_wrong_width_is_rejected() -> None:
    bm = BoltzmannMachine(num_nodes=5)
    with pytest.raises(ValueError, match=r"must have shape \(batch, 5\)"):
        bm.objective(torch.zeros(2, 4), torch.zeros(2, 5))


def test_different_batch_sizes_are_allowed() -> None:
    bm = BoltzmannMachine(num_nodes=5)
    objective = bm.objective(torch.zeros(2, 5), torch.zeros(3, 5))
    assert objective.ndim == 0
    assert torch.isfinite(objective)
