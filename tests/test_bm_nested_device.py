"""Boltzmann machines nested in a parent module must follow parent moves."""

import torch
from torch import nn

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


class Parent(nn.Module):
    """Parent module holding two Boltzmann machines."""

    def __init__(self):
        super().__init__()
        self.bm = BoltzmannMachine(3)
        self.rbm = RestrictedBoltzmannMachine(2, 2)


def test_a_parent_move_updates_the_nested_device_attributes():
    parent = Parent()

    parent.to(torch.device("meta"))

    assert parent.bm.device.type == "meta"
    assert parent.rbm.device.type == "meta"
    assert parent.bm.linear_bias.device.type == "meta"
