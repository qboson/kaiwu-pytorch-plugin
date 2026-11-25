"""Check full-stack reconstruction independently of the individual layer helper."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin.dbn import UnsupervisedDBN


def test_two_layer_reconstruction_matches_manual_probability_chain(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = UnsupervisedDBN([3, 1]).create_rbm_layer(2).mark_as_trained()
    weights = [torch.tensor([[0.5, 0.6, -0.3], [-0.4, 0.3, 0.7]]),
               torch.tensor([[0.2], [-0.8], [0.6]])]
    visible_bias = [torch.tensor([0.1, 0.7]), torch.tensor([-0.4, 0.3, -0.2])]
    hidden_bias = [torch.tensor([-0.1, 0.6, 0.1]), torch.tensor([0.7])]
    with torch.no_grad():
        for rbm, weight, visible, hidden in zip(
            model.rbm_layers, weights, visible_bias, hidden_bias
        ):
            rbm.quadratic_coef.copy_(weight)
            rbm.visible_bias.copy_(visible)
            rbm.hidden_bias.copy_(hidden)

    samples = torch.tensor([[1., 0.], [0.2, 0.8], [0., 1.]])
    encoded = torch.sigmoid(samples @ weights[0] + hidden_bias[0])
    encoded = torch.sigmoid(encoded @ weights[1] + hidden_bias[1])
    decoded = torch.sigmoid(encoded @ weights[1].T + visible_bias[1])
    expected = torch.sigmoid(decoded @ weights[0].T + visible_bias[0])

    actual, error = model.reconstruct_stack(samples.numpy())
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-6)
    np.testing.assert_allclose(error, (samples - expected).square().mean(dim=1).numpy())


def test_one_layer_matches_existing_reconstruction(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = UnsupervisedDBN([3]).create_rbm_layer(2).mark_as_trained()
    samples = np.array([[0.7, 0.2], [1.0, 0.0]], dtype=np.float32)
    for actual, expected in zip(model.reconstruct_stack(samples), model.reconstruct(samples)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_untrained_and_empty_stack_contracts():
    samples = np.zeros((2, 3))
    model = UnsupervisedDBN([]).create_rbm_layer(3)
    with pytest.raises(ValueError, match="not trained"):
        model.reconstruct_stack(samples)
    recovered, error = model.mark_as_trained().reconstruct_stack(samples)
    np.testing.assert_array_equal(recovered, samples)
    np.testing.assert_array_equal(error, np.zeros(2))
    with pytest.raises(ValueError, match="2D"):
        model.reconstruct_stack(np.zeros(3))
