"""Validation reconstructions keep the likelihood's output semantics (Issue #527)."""

import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
from kaiwu.torch_plugin.qvae import QVAE  # noqa: E402

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
MODEL_TUNER_PATH = EXAMPLE_DIR / "trainer" / "model_tuner.py"


def load_model_tuner_class():
    """Load ModelTuner without importing the example's plotting dependencies."""
    tree = ast.parse(MODEL_TUNER_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelTuner"
    )
    namespace = {"torch": torch, "logger": logging.getLogger(__name__)}
    module = ast.Module(body=[class_node], type_ignores=[])
    exec(compile(module, str(MODEL_TUNER_PATH), "exec"), namespace)  # pylint: disable=exec-used
    return namespace["ModelTuner"]


class FixedValueDecoder(torch.nn.Module):
    """Map any latent sample to the continuous reconstruction [-2, 0, 2]."""

    def __init__(self):
        super().__init__()
        self.register_buffer("values", torch.tensor([[-2.0, 0.0, 2.0]]))

    def forward(self, zeta):
        return self.values.expand(zeta.size(0), -1)


class ZeroSampler:
    """Deterministic solver used by the Boltzmann machine's negative phase."""

    def solve(self, ising_matrix):
        return np.zeros((1, ising_matrix.shape[0]), dtype=np.float32)


def build_tuner(loss_type):
    """Build a real QVAE + ModelTuner for the requested likelihood."""
    config = SimpleNamespace(
        type="QVAE",
        num_latent_units=2,
        loss_type=loss_type,
        dist_beta=2.0,
        kl_beta=1e-6,
        weight_decay=0.0,
    )
    model = QVAE(
        3,
        None,
        config,
        encoder=torch.nn.Linear(3, 2),
        decoder=FixedValueDecoder(),
        bm=RestrictedBoltzmannMachine(1, 1),
        sampler=ZeroSampler(),
    )
    model.eval()

    inputs = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    loader = DataLoader(
        TensorDataset(inputs, torch.zeros(4, dtype=torch.long)),
        batch_size=2,
        shuffle=False,
    )
    tuner = load_model_tuner_class()(config)
    tuner.register_model(model)
    tuner.register_dataLoaders(loader, loader)
    return tuner, inputs


@pytest.mark.parametrize("loss_type", ["mse", "bernoulli"])
def test_validation_reconstruction_matches_loss_semantics(loss_type):
    """MSE returns the continuous decoder output; Bernoulli returns probabilities."""
    tuner, inputs = build_tuner(loss_type)

    loss, last_inputs, reconstruction, label_list = tuner.test()

    expected = torch.tensor([[-2.0, 0.0, 2.0]]).expand(2, -1)
    if loss_type == "bernoulli":
        expected = torch.sigmoid(expected)
    torch.testing.assert_close(reconstruction, expected)
    torch.testing.assert_close(last_inputs, inputs[-2:])
    assert label_list is None
    assert torch.isfinite(loss)
    assert not reconstruction.requires_grad
