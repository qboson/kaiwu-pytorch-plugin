"""The Ising step must return exactly num_basis directions."""

import torch

from kaiwu.torch_plugin.qdeq import _search_directions


def _unit(vector):
    return vector / (vector.norm() + 1e-9)


def test_extra_history_is_truncated_to_num_basis():
    residual = torch.tensor([1.0, 0.0])
    residuals = [
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 2.0]),
    ]
    updates = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([2.0, 2.0]),
    ]
    changes = [
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 2.0]),
    ]

    directions = _search_directions(residual, residuals, updates, changes, 2)

    assert directions.shape == (2, 2)
    torch.testing.assert_close(directions[0], _unit(-residual))
    torch.testing.assert_close(directions[1], _unit(residuals[-1]))


def test_a_basis_limited_by_history_is_not_padded_with_random_vectors():
    residual = torch.tensor([1.0, 0.0])
    directions = _search_directions(residual, [torch.tensor([0.0, 1.0])], [], [], 2)

    assert directions.shape == (2, 2)
    torch.testing.assert_close(directions[1], _unit(torch.tensor([0.0, 1.0])))
