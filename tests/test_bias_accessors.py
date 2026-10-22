"""Bias accessors must slice the documented sides of linear_bias."""

import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


def test_boltzmann_machine_bias_accessors_slice_the_ends():
    model = BoltzmannMachine(5, linear_bias=torch.arange(5, dtype=torch.float32))

    torch.testing.assert_close(model.visible_bias(2), torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(model.hidden_bias(3), torch.tensor([2.0, 3.0, 4.0]))


def test_restricted_boltzmann_machine_bias_accessors_slice_the_ends():
    model = RestrictedBoltzmannMachine(
        2, 3, linear_bias=torch.arange(5, dtype=torch.float32)
    )

    torch.testing.assert_close(model.visible_bias, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(model.hidden_bias, torch.tensor([2.0, 3.0, 4.0]))
