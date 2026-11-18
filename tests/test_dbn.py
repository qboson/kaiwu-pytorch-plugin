"""DBN inspection and transformation tests across model lifecycle states."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin.dbn import UnsupervisedDBN


@pytest.fixture
def cpu_dbn(monkeypatch):
    """Create real RBM layers on CPU even on hosts with an available GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return UnsupervisedDBN(hidden_layers_structure=[4, 2])


@pytest.mark.parametrize("structure", [None, [], [4, 2]])
def test_new_dbn_has_no_created_layers_or_known_output_dimension(cpu_dbn, structure):
    """Inspection is valid before input dimensions and RBM layers are known."""
    dbn = UnsupervisedDBN(hidden_layers_structure=structure)
    assert dbn.num_layers == 0
    assert dbn.output_dim is None


@pytest.mark.parametrize("index", [-1, 0, 1])
def test_new_dbn_returns_none_for_missing_layers(cpu_dbn, index):
    """A deferred layer is absent rather than an error during inspection."""
    assert cpu_dbn.get_rbm_layer(index) is None


def test_unbuilt_metadata_allows_the_trainer_to_create_layers(cpu_dbn):
    """The example trainer's num_layers check can initialize a fresh DBN."""
    data = np.zeros((2, 3), dtype=np.float32)

    if cpu_dbn.num_layers == 0:
        cpu_dbn.create_rbm_layer(data.shape[1])

    assert cpu_dbn.num_layers == 2
    assert cpu_dbn.output_dim == 2
    assert cpu_dbn.get_rbm_layer(0).num_visible == 3
    assert cpu_dbn.get_rbm_layer(0).num_hidden == 4
    assert cpu_dbn.get_rbm_layer(1).num_visible == 4


@pytest.mark.parametrize("index,expected_index", [(0, 0), (1, 1), (-2, 0), (-1, 1)])
def test_built_dbn_retains_positive_and_negative_layer_lookup(cpu_dbn, index, expected_index):
    """Valid negative indices preserve the existing Python sequence semantics."""
    cpu_dbn.create_rbm_layer(input_dim=3)

    assert cpu_dbn.get_rbm_layer(index) is cpu_dbn.rbm_layers[expected_index]


@pytest.mark.parametrize("index", [-100, -3, 2, 100])
def test_built_dbn_returns_none_outside_both_index_boundaries(cpu_dbn, index):
    """Indices past either end refer to a missing layer."""
    cpu_dbn.create_rbm_layer(input_dim=3)

    assert cpu_dbn.get_rbm_layer(index) is None


def test_inspection_does_not_bypass_forward_lifecycle_guards(cpu_dbn):
    """Safe metadata does not make an unbuilt or untrained model transformable."""
    data = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Model not built yet"):
        cpu_dbn.transform(data)

    cpu_dbn.create_rbm_layer(input_dim=3)
    assert cpu_dbn.num_layers == 2
    assert cpu_dbn.output_dim == 2
    with pytest.raises(ValueError, match="Model not trained yet"):
        cpu_dbn.transform(data)


def test_real_cpu_training_preserves_metadata_and_transform_dimensions(cpu_dbn):
    """After real parameter updates, feature dimensions still match the stack."""
    cpu_dbn.create_rbm_layer(input_dim=3)
    for rbm in cpu_dbn.rbm_layers:
        assert rbm.quadratic_coef.device.type == "cpu"
        original_weights = rbm.quadratic_coef.detach().clone()
        optimizer = torch.optim.SGD(rbm.parameters(), lr=0.1)
        positive = torch.ones(2, rbm.num_nodes)
        negative = torch.zeros_like(positive)
        optimizer.zero_grad()
        rbm.objective(positive, negative).backward()
        optimizer.step()
        assert not torch.equal(rbm.quadratic_coef, original_weights)

    assert cpu_dbn.mark_as_trained() is cpu_dbn
    assert cpu_dbn.num_layers == 2
    assert cpu_dbn.output_dim == 2
    data = np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]], dtype=np.float64)
    expected = torch.tensor(data, dtype=torch.float32)
    for rbm in cpu_dbn.rbm_layers:
        expected = torch.sigmoid(expected @ rbm.quadratic_coef + rbm.hidden_bias)

    actual = cpu_dbn.transform(data)

    assert actual.shape == (2, cpu_dbn.output_dim)
    np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=1e-6, atol=1e-7)


def test_rebuilding_updates_metadata_and_requires_training_again(cpu_dbn):
    """Rebuilding replaces the stack and resets the training requirement."""
    cpu_dbn.create_rbm_layer(input_dim=3).mark_as_trained()
    old_layer = cpu_dbn.get_rbm_layer(0)
    cpu_dbn.hidden_layers_structure = [5]

    cpu_dbn.create_rbm_layer(input_dim=6)

    assert cpu_dbn.num_layers == 1
    assert cpu_dbn.output_dim == 5
    assert cpu_dbn.get_rbm_layer(0) is not old_layer
    assert cpu_dbn.get_rbm_layer(0).num_visible == 6
    assert cpu_dbn.get_rbm_layer(-2) is None
    with pytest.raises(ValueError, match="Model not trained yet"):
        cpu_dbn.transform(np.zeros((2, 6), dtype=np.float32))


def test_empty_stack_has_input_dimension_and_identity_transform(cpu_dbn):
    """A built zero-layer stack preserves input features and has no layers."""
    empty_dbn = UnsupervisedDBN(hidden_layers_structure=[])
    empty_dbn.create_rbm_layer(input_dim=3)

    assert empty_dbn.num_layers == 0
    assert empty_dbn.output_dim == 3
    for index in [-1, 0, 1]:
        assert empty_dbn.get_rbm_layer(index) is None

    data = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Model not trained yet"):
        empty_dbn.transform(data)
    empty_dbn.mark_as_trained()
    assert empty_dbn.output_dim == 3
    np.testing.assert_array_equal(empty_dbn.transform(data), data.astype(np.float32))
