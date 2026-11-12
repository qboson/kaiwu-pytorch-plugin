"""Export sampler matrices from bfloat16 Boltzmann machines."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def make_model(kind, dtype):
    if kind == "full":
        model = BoltzmannMachine(4, device="cpu").to(dtype=dtype)
    else:
        model = GaussianBernoulliRestrictedBoltzmannMachine(2, 2, dtype=dtype, device="cpu")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.fill_(0.0 if name == "log_var" else 0.25)
    return model


@pytest.mark.parametrize("kind", ["full", "gaussian-bernoulli"])
def test_bfloat16_ising_matrix_exports_as_float32(kind):
    model = make_model(kind, torch.bfloat16)
    reference = make_model(kind, torch.float32).get_ising_matrix()

    actual = model.get_ising_matrix()

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, reference)
    assert next(model.parameters()).dtype == torch.bfloat16


def test_full_bm_conditioned_sampler_receives_float32_ising_matrix():
    class AllUpSampler:
        def solve(self, matrix):
            assert matrix.dtype == np.float32
            assert matrix.shape == (3, 3)
            return np.ones((1, 3), dtype=np.int8)

    model = make_model("full", torch.bfloat16)
    visible = torch.tensor([[0., 1.]], dtype=torch.bfloat16)
    states = model.condition_sample(AllUpSampler(), visible, dtype=visible.dtype)
    torch.testing.assert_close(states, torch.tensor([[0., 1., 1., 1.]], dtype=visible.dtype))


def test_float64_ising_output_preserves_precision():
    assert make_model("full", torch.float64).get_ising_matrix().dtype == np.float64
