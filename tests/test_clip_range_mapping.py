"""h_range clamps the linear bias and j_range the quadratic couplings."""

import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


def test_bm_clip_ranges_map_h_to_linear_and_j_to_quadratic():
    model = BoltzmannMachine(
        3,
        quadratic_coef=torch.full((3, 3), 0.5),
        linear_bias=torch.tensor([0.5, -0.5, 0.2]),
    )

    model.clip_parameters(h_range=(-0.1, 0.1), j_range=(-0.2, 0.2))

    torch.testing.assert_close(model.linear_bias.detach(), torch.tensor([0.1, -0.1, 0.1]))
    torch.testing.assert_close(model.quadratic_coef.detach(), torch.full((3, 3), 0.2))


def test_rbm_clip_ranges_map_h_to_linear_and_j_to_quadratic():
    model = RestrictedBoltzmannMachine(
        2,
        2,
        quadratic_coef=torch.full((2, 2), 0.5),
        linear_bias=torch.tensor([0.5, -0.5, 0.2, -0.2]),
    )

    model.clip_parameters(h_range=(-0.1, 0.1), j_range=(-0.2, 0.2))

    torch.testing.assert_close(
        model.linear_bias.detach(), torch.tensor([0.1, -0.1, 0.1, -0.1])
    )
    torch.testing.assert_close(model.quadratic_coef.detach(), torch.full((2, 2), 0.2))
