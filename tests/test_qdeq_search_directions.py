"""The Ising step directions must follow the documented order and norm."""

import torch

from kaiwu.torch_plugin.qdeq import _search_directions


def _unit(vector):
    return vector / (vector.norm() + 1e-9)


def test_first_direction_is_the_normalized_negative_residual():
    residual = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

    directions = _search_directions(residual, [], [], [], 1)

    assert directions.shape == (1, 2, 2)
    torch.testing.assert_close(directions[0], _unit(-residual))


def test_residual_history_is_used_newest_first():
    residual = torch.tensor([1.0, 0.0])
    old = torch.tensor([2.0, 0.0])
    new = torch.tensor([0.0, 3.0])

    directions = _search_directions(residual, [old, new], [], [], 3)

    torch.testing.assert_close(directions[1], _unit(new))
    torch.testing.assert_close(directions[2], _unit(old))


def test_update_pairs_follow_the_newest_residual():
    residual = torch.tensor([1.0, 0.0])
    dz_old, dr_old = torch.tensor([1.0, 1.0]), torch.tensor([2.0, 0.0])
    dz_new, dr_new = torch.tensor([0.0, 1.0]), torch.tensor([0.0, 2.0])

    directions = _search_directions(residual, [], [dz_old, dz_new], [dr_old, dr_new], 5)

    torch.testing.assert_close(directions[1], _unit(dz_new))
    torch.testing.assert_close(directions[2], _unit(dr_new))
    torch.testing.assert_close(directions[3], _unit(dz_old))
    torch.testing.assert_close(directions[4], _unit(dr_old))


def test_missing_directions_are_filled_with_unit_sign_vectors():
    residual = torch.tensor([1.0, 0.0])

    directions = _search_directions(residual, [], [], [], 4)

    assert directions.shape == (4, 2)
    torch.testing.assert_close(directions[0], _unit(-residual))
    magnitude = 1.0 / (residual.numel() ** 0.5 + 1e-9)
    expected = torch.full_like(directions[1:], magnitude)
    torch.testing.assert_close(torch.abs(directions[1:]), expected)
    torch.testing.assert_close(directions.norm(dim=-1), torch.ones(4))
