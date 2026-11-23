"""A single DBN pass exposes reproducible features from each trained layer."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin.dbn import UnsupervisedDBN


def test_transform_layers_matches_explicit_rbm_probabilities(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = UnsupervisedDBN([3, 2]).create_rbm_layer(2).mark_as_trained()
    weights = [
        torch.tensor([[0.1, 0.2, -0.4], [-0.5, 0.1, 0.7]]),
        torch.tensor([[0.4, 0.3], [-0.1, -0.8], [0.6, 0.7]]),
    ]
    biases = [torch.tensor([-0.3, 0.1, 0.3]), torch.tensor([0.25, -0.4])]
    with torch.no_grad():
        for rbm, weight, bias in zip(model.rbm_layers, weights, biases):
            rbm.quadratic_coef.copy_(weight)
            rbm.hidden_bias.copy_(bias)
    samples = np.array([[0.0, 0.5], [0.25, 1.0], [1.0, 0.0]], dtype=np.float64)
    expected_first = torch.sigmoid(torch.tensor(samples, dtype=torch.float32) @ weights[0] + biases[0])
    expected_second = torch.sigmoid(expected_first @ weights[1] + biases[1])

    first, second = model.transform_layers(samples)

    np.testing.assert_allclose(first, expected_first.numpy(), rtol=1e-6)
    np.testing.assert_allclose(second, expected_second.numpy(), rtol=1e-6)
    np.testing.assert_array_equal(model.transform(samples), second)


def test_transform_layers_requires_a_trained_stack():
    model = UnsupervisedDBN([2])
    samples = np.zeros((1, 2))
    with pytest.raises(ValueError, match="not built"):
        model.transform_layers(samples)
    model.create_rbm_layer(2)
    with pytest.raises(ValueError, match="not trained"):
        model.transform_layers(samples)


def test_empty_trained_stack_has_no_intermediate_features():
    model = UnsupervisedDBN([]).create_rbm_layer(2).mark_as_trained()
    samples = np.ones((1, 2))
    assert model.transform_layers(samples) == ()
    np.testing.assert_array_equal(model.transform(samples), samples.astype(np.float32))
