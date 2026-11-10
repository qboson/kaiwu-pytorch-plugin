from __future__ import annotations

import os
import sys

import numpy as np
import torch

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

from kaiwu.torch_plugin.restricted_boltzmann_machine import RestrictedBoltzmannMachine


def test_double_model_ising_matrix_keeps_dtype() -> None:
    """A .double() RBM must not silently downcast its Ising matrix."""
    rbm = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2).double()

    ising = rbm.get_ising_matrix()

    assert ising.dtype == np.float64
    assert ising.shape == (6, 6)


def test_float32_ising_matrix_unchanged() -> None:
    """The default float32 pipeline keeps its dtype."""
    rbm = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)

    ising = rbm.get_ising_matrix()

    assert ising.dtype == np.float32
