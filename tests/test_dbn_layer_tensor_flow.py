"""DBN inference exports NumPy only after the final RBM layer."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import UnsupervisedDBN


def build_dbn(structure):
    model = UnsupervisedDBN(structure)
    model.create_rbm_layer(3)
    for layer in model.rbm_layers:
        with torch.no_grad():
            layer.quadratic_coef.fill_(0.2)
            layer.linear_bias.fill_(0.1)
    return model.mark_as_trained()


@pytest.mark.parametrize("structure", [[4, 2], [4, 3, 2]])
def test_forward_keeps_layer_outputs_as_tensors(monkeypatch, structure):
    model = build_dbn(structure)
    data = np.array([[0.25, 0.5, 0.75], [0.5, 0.75, 1.]], dtype=np.float64)
    # Independent dense-layer reference; each RBM has constant weights and bias.
    expected = torch.tensor(data, dtype=torch.float32)
    for width in structure:
        expected = torch.sigmoid(expected.sum(-1, keepdim=True) * 0.2 + 0.1)
        expected = expected.expand(-1, width)
    expected = expected.numpy()
    exports = []
    tensor_numpy = torch.Tensor.numpy

    def record_numpy(tensor, *args, **kwargs):
        exports.append(tuple(tensor.shape))
        return tensor_numpy(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "numpy", record_numpy)
    output = model.transform(data)

    np.testing.assert_allclose(output, expected, rtol=1e-6)
    assert output.dtype == np.float32
    assert exports == [(2, structure[-1])]
    assert model.rbm_layers[-1].quadratic_coef.grad is None


def test_empty_trained_stack_keeps_float32_numpy_contract():
    model = build_dbn([])
    data = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)

    result = model(data)

    np.testing.assert_array_equal(result, data.astype(np.float32))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_forward_still_requires_a_built_and_trained_model():
    model = UnsupervisedDBN([2])
    data = np.ones((1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="not built"):
        model(data)
    model.create_rbm_layer(3)
    with pytest.raises(ValueError, match="not trained"):
        model(data)
