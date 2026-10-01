"""DBN numpy inference must align with each actual RBM's parameter precision."""
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin.dbn import UnsupervisedDBN
from kaiwu.torch_plugin import dbn as dbn_module
from kaiwu.torch_plugin import restricted_boltzmann_machine as rbm_module

for module in (dbn_module, rbm_module):
    assert Path(module.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / 'src')

DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
DATA = np.array([[.1, .2, .3], [.7, .4, .9], [.25, .8, .6]], dtype=np.float64)


def numpy_values(tensor):
    return tensor.detach().cpu().to(torch.float64).numpy()


def numpy_dtype(dtype):
    return {torch.float64: np.float64, torch.float16: np.float16,
            torch.float32: np.float32, torch.bfloat16: np.float32}[dtype]


def trained_dbn(dtypes):
    dbn = UnsupervisedDBN([2, 1]).create_rbm_layer(3).mark_as_trained()
    for index, (rbm, dtype) in enumerate(zip(dbn.rbm_layers, dtypes)):
        # Use the actual inherited PyTorch conversions on already-built layers.
        {torch.float32: rbm.float, torch.float64: rbm.double,
         torch.float16: rbm.half, torch.bfloat16: rbm.bfloat16}[dtype]()
        with torch.no_grad():
            weights = torch.linspace(-.7 + .1 * index, .8, rbm.num_visible * rbm.num_hidden)
            rbm.quadratic_coef.copy_(weights.reshape(rbm.num_visible, rbm.num_hidden))
            rbm.linear_bias.copy_(torch.linspace(-.2, .3, rbm.num_visible + rbm.num_hidden))
    return dbn


def energy_marginals(rbm, values, hidden=True):
    """Enumerate the conditional Hamiltonian distribution independently in numpy."""
    weights = numpy_values(rbm.quadratic_coef)
    biases = numpy_values(rbm.linear_bias)
    visible_bias, hidden_bias = biases[:rbm.num_visible], biases[rbm.num_visible:]
    if hidden:
        states = np.array(list(product((0., 1.), repeat=rbm.num_hidden)))
        energies = -(values @ visible_bias)[:, None] - states @ hidden_bias
        energies -= values @ weights @ states.T
    else:
        states = np.array(list(product((0., 1.), repeat=rbm.num_visible)))
        energies = -(values @ hidden_bias)[:, None] - states @ visible_bias
        energies -= values @ weights.T @ states.T
    distribution = np.exp(-energies - np.max(-energies, axis=1, keepdims=True))
    distribution /= distribution.sum(axis=1, keepdims=True)
    return distribution @ states


def observe_hidden(rbm, monkeypatch):
    """Keep the real public RBM hook and record only its dtype/device contract."""
    calls = []
    get_hidden = rbm.get_hidden
    def hidden(values):
        output = get_hidden(values)
        calls.append((values.detach().clone(), output.dtype))
        return output
    monkeypatch.setattr(rbm, 'get_hidden', hidden)
    return calls


def tolerance(dtype):
    return max(2e-7, 4 * torch.finfo(dtype).eps)


@pytest.mark.parametrize('dtype', DTYPES)
def test_forward_and_transform_keep_real_layer_hooks_and_native_outputs(dtype, monkeypatch):
    dbn = trained_dbn([dtype, dtype])
    before = {name: tensor.detach().clone() for name, tensor in dbn.state_dict().items()}
    observations = [observe_hidden(rbm, monkeypatch) for rbm in dbn.rbm_layers]
    output = dbn.forward(DATA)
    expected = DATA
    for rbm, calls in zip(dbn.rbm_layers, observations):
        values, container_dtype = calls[0]
        assert values.dtype == rbm.quadratic_coef.dtype
        assert values.device == rbm.quadratic_coef.device
        assert not values.requires_grad
        expected_values = numpy_values(torch.as_tensor(expected, dtype=values.dtype))
        np.testing.assert_allclose(numpy_values(values), expected_values,
                                   atol=tolerance(dtype), rtol=tolerance(dtype))
        expected = energy_marginals(rbm, expected_values).astype(numpy_dtype(container_dtype))
    assert isinstance(output, np.ndarray)
    assert output.dtype == numpy_dtype(observations[-1][0][1])
    np.testing.assert_allclose(output, expected, atol=tolerance(dtype), rtol=tolerance(dtype))
    np.testing.assert_array_equal(dbn.transform(DATA), output)
    for name, tensor in dbn.state_dict().items():
        assert torch.equal(tensor, before[name])
        assert tensor.grad is None


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('route', ['instance', 'static', 'explicit_device'])
def test_reconstruction_preserves_parameter_precision_and_numpy_error(dtype, route, monkeypatch):
    dbn = trained_dbn([dtype, dtype])
    rbm = dbn.rbm_layers[0]
    calls = observe_hidden(rbm, monkeypatch)
    if route == 'instance':
        visible, errors = dbn.reconstruct(DATA)
    else:
        device = torch.device('cpu') if route == 'explicit_device' else None
        visible, errors = dbn.reconstruct_with_rbm(rbm, DATA, device)
    values, container_dtype = calls[0]
    assert values.dtype == dtype and values.device == rbm.quadratic_coef.device
    inputs = numpy_values(values)
    # f047's get_hidden has a known float32 state container (owned by PR186).
    # The oracle retains that native hook contract; it does not pretend DBN
    # can recover precision already discarded by the supplied RBM implementation.
    hidden = energy_marginals(rbm, inputs).astype(numpy_dtype(container_dtype))
    hidden = numpy_values(torch.as_tensor(hidden, dtype=dtype))
    expected_visible = energy_marginals(rbm, hidden, hidden=False)
    expected_errors = np.mean((inputs - expected_visible) ** 2, axis=1)
    assert visible.dtype == errors.dtype == numpy_dtype(dtype)
    np.testing.assert_allclose(visible, expected_visible, atol=tolerance(dtype), rtol=tolerance(dtype))
    np.testing.assert_allclose(errors, expected_errors, atol=tolerance(dtype), rtol=tolerance(dtype))
    assert visible.shape == DATA.shape and errors.shape == (len(DATA),)


@pytest.mark.parametrize('dtypes', [
    [torch.float64, torch.float16], [torch.bfloat16, torch.float64],
    [torch.float16, torch.bfloat16],
])
def test_mixed_precision_layers_convert_at_each_real_hook(dtypes, monkeypatch):
    dbn = trained_dbn(dtypes)
    observations = [observe_hidden(rbm, monkeypatch) for rbm in dbn.rbm_layers]
    output = dbn.forward(DATA)
    assert output.shape == (len(DATA), 1)
    for rbm, calls, dtype in zip(dbn.rbm_layers, observations, dtypes):
        assert calls[0][0].dtype == rbm.quadratic_coef.dtype == dtype
    second_input = DATA[:, :2]
    visible, errors = dbn.reconstruct(second_input, layer_index=1)
    assert observations[1][-1][0].dtype == dtypes[1]
    assert visible.dtype == errors.dtype == numpy_dtype(dtypes[1])
    assert visible.shape == second_input.shape


def test_double_model_receives_input_details_below_float32_resolution():
    dbn = UnsupervisedDBN([1]).create_rbm_layer(1).double().mark_as_trained()
    rbm = dbn.rbm_layers[0]
    with torch.no_grad():
        rbm.quadratic_coef.fill_(1e7)
        rbm.linear_bias.copy_(torch.tensor([0., -1e7], dtype=torch.float64))
    data = np.array([[1. + 1e-8]], dtype=np.float64)
    expected = energy_marginals(rbm, data)
    actual = dbn.forward(data)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
    assert abs(actual.item() - .5) > .02


def test_default_float32_forward_and_reconstruction_match_existing_calculation():
    dbn = trained_dbn([torch.float32, torch.float32])
    values = DATA.astype(np.float32)
    for rbm in dbn.rbm_layers:
        values = rbm.get_hidden(torch.FloatTensor(values))[:, rbm.num_visible:].cpu().numpy()
    np.testing.assert_array_equal(dbn.forward(DATA), values)
    rbm = dbn.rbm_layers[0]
    with torch.no_grad():
        inputs = torch.FloatTensor(DATA)
        hidden = rbm.get_hidden(inputs)[:, rbm.num_visible:]
        expected = torch.sigmoid(hidden @ rbm.quadratic_coef.T + rbm.visible_bias)
        expected_error = ((inputs - expected) ** 2).mean(dim=1)
    visible, errors = dbn.reconstruct(DATA)
    np.testing.assert_array_equal(visible, expected.numpy())
    np.testing.assert_array_equal(errors, expected_error.numpy())


@pytest.mark.parametrize('dtype', [np.float64, np.int64])
def test_empty_trained_stack_keeps_original_float32_numpy_default(dtype):
    dbn = UnsupervisedDBN([]).create_rbm_layer(3).double().mark_as_trained()
    data = np.asarray([[1, 2, 3]], dtype=dtype)
    np.testing.assert_array_equal(dbn.forward(data), data.astype(np.float32))
    assert dbn.transform(data).dtype == np.float32
