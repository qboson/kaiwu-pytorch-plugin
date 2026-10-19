"""Cell energy scoring must use the precision of its actual BM parameters."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


@pytest.fixture(name="cell_class")
def fixture_cell_class():
    """Load the real optional model without importing Scanpy or a solver."""
    path = Path(__file__).resolve().parents[1] / "example/qvae_cell/models.py"
    spec = importlib.util.spec_from_file_location("cell_precision_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CellQVAE


def make_model(cell_class, loss_type):
    """Build a small real encoder and BM with exactly representable weights."""
    encoder = torch.nn.Linear(4, 4)
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(4))
        encoder.bias.zero_()
    model = cell_class(
        input_dimension=4,
        activation_fct=None,
        config=SimpleNamespace(
            num_latent_units=4, loss_type=loss_type, dist_beta=1.0,
            kl_beta=0.1, weight_decay=0.0,
        ),
        encoder=encoder,
        decoder=torch.nn.Linear(4, 4),
        bm=BoltzmannMachine(
            4, device="cpu",
            linear_bias=torch.tensor([0.5, -1.0, 0.25, -0.5]),
            quadratic_coef=torch.tensor([
                [0.0, 0.25, -0.5, 0.75],
                [0.0, 0.0, 1.0, -0.25],
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.0],
            ]),
        ),
        sampler=object(),
    ).eval()
    model.set_dataset_mean(torch.full((4,), 0.5))
    return model


@pytest.mark.parametrize("loss_type", ["mse", "bernoulli"])
@pytest.mark.parametrize("dtype", [
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
])
def test_converted_cell_energy_matches_binary_hamiltonian(cell_class, loss_type, dtype):
    """Exercise actual encoding and posterior code, BM scoring and gradients."""
    model = make_model(cell_class, loss_type).to(dtype=dtype)
    inputs = torch.tensor([
        [-1.0, 0.0, 0.25, 1.0],
        [1.0, 0.75, 0.0, -1.0],
        [0.5, 1.0, -0.5, 0.75],
    ], dtype=dtype)
    centered = inputs - 0.5 if loss_type == "bernoulli" else inputs
    states = (centered > 0).to(dtype)
    biases = [0.5, -1.0, 0.25, -0.5]
    pairs = {(0, 1): 0.25, (0, 2): -0.5, (0, 3): 0.75,
             (1, 2): 1.0, (1, 3): -0.25, (2, 3): 0.5}
    expected = torch.tensor([
        -sum(biases[i] * row[i] for i in range(4))
        -sum(weight * row[i] * row[j] for (i, j), weight in pairs.items())
        for row in states.tolist()
    ], dtype=dtype)

    actual = model.energy(inputs, loss_type)

    assert actual.dtype == dtype
    assert actual.device == inputs.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(model.bm.linear_bias.grad, -states.sum(0))
    expected_coupling_grad = -(states.T @ states).triu(1)
    torch.testing.assert_close(model.bm.quadratic_coef.grad, expected_coupling_grad)


def test_cell_energy_uses_bm_precision_in_a_mixed_model(cell_class):
    """The BM, rather than the float32 encoder, owns binary-state precision."""
    model = make_model(cell_class, "mse")
    model.bm.double()
    inputs = torch.tensor([[1.0, -1.0, 0.0, 1.0]])

    energy = model.energy(inputs, "mse")

    assert next(model.encoder.parameters()).dtype == torch.float32
    assert energy.dtype == torch.float64
    torch.testing.assert_close(energy, torch.tensor([-0.75], dtype=torch.float64))


def test_cell_energy_keeps_loss_type_validation(cell_class):
    """Precision alignment must not silently accept a mismatched loss type."""
    model = make_model(cell_class, "mse").double()
    with pytest.raises(ValueError, match="loss_type must match"):
        model.energy(torch.ones(1, 4, dtype=torch.float64), "bernoulli")


class ParameterlessEnergy(torch.nn.Module):
    """An allowed user-supplied energy module with no parameter metadata."""

    def forward(self, states):
        """Score binary states using their own floating-point precision."""
        assert states.is_floating_point()
        return -states.sum(dim=-1)


def test_parameterless_cell_energy_uses_encoder_precision(cell_class):
    """Missing BM parameters retain a usable floating-point fallback."""
    model = make_model(cell_class, "mse").double()
    model.bm = ParameterlessEnergy()
    energy = model.energy(torch.tensor([[1.0, 0.0, -1.0, 2.0]]).double(), "mse")
    torch.testing.assert_close(energy, torch.tensor([-2.0], dtype=torch.float64))
