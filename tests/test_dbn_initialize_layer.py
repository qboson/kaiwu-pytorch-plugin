"""Fine-tune initialisation contract for the supervised DBN example."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


def build_rbm():
    """Builds a 2-visible/3-hidden RBM with a known coefficient matrix."""
    rbm = RestrictedBoltzmannMachine(num_visible=2, num_hidden=3)
    with torch.no_grad():
        rbm.quadratic_coef.copy_(
            torch.arange(6, dtype=rbm.quadratic_coef.dtype).view(2, 3)
        )
        rbm.linear_bias.copy_(torch.arange(5, dtype=rbm.linear_bias.dtype))
    return rbm


def test_layer_is_initialised_from_the_transposed_rbm_coefficients():
    """Weights come from the transposed interaction matrix, bias from the hidden part."""
    model = SupervisedDBNClassification.__new__(SupervisedDBNClassification)
    layer = torch.nn.Linear(2, 3)
    rbm = build_rbm()
    model._initialize_layer_with_pretrained(layer, rbm)
    assert layer.weight.data.tolist() == rbm.quadratic_coef.T.tolist()
    assert layer.bias.data.tolist() == [2.0, 3.0, 4.0]
    assert layer.weight.dtype == torch.float32
