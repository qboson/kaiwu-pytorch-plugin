"""Native CUDA recurrence supports mask curvature without changing weight training."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kaiwu.torch_plugin import FeatureSelectionWrapper

CUDA_CUDNN = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.backends.cudnn.is_available(),
    reason="requires CUDA and cuDNN",
)


class RecurrentClassifier(nn.Module):
    """Ordinary one-layer sequence classifier with a shared feature mask."""

    def __init__(self, kind):
        super().__init__()
        self.recurrent = getattr(nn, kind)(4, 5, batch_first=True)
        self.head = nn.Linear(5, 2)

    def forward(self, values):
        """Classify the last recurrent hidden state."""
        _sequence, hidden = self.recurrent(values)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.head(hidden[-1])


def _flags():
    backend = torch.backends.cudnn
    return (backend.enabled, backend.benchmark, backend.benchmark_limit,
            backend.deterministic, backend.allow_tf32)


def _rng_state():
    return (random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
            torch.cuda.get_rng_state_all())


def _assert_rng_state(expected):
    actual = _rng_state()
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    assert np.array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert torch.equal(actual[2], expected[2])
    assert len(actual[3]) == len(expected[3])
    assert all(torch.equal(left, right) for left, right in zip(actual[3], expected[3]))


def _fixture(kind, dtype, device):
    with torch.random.fork_rng():
        torch.manual_seed(637)
        model = RecurrentClassifier(kind).to(device=device, dtype=dtype)
    generator = torch.Generator().manual_seed(741)
    values = torch.randn(7, 4, 3, generator=generator, dtype=dtype).transpose(1, 2)
    labels = torch.tensor([0, 1, 1, 0, 1, 0, 1])
    wrapper = FeatureSelectionWrapper(model, 4).to(device=device, dtype=dtype)
    with torch.no_grad():
        wrapper.mask.copy_(wrapper.mask.new_tensor([1, 0, 1, 1]))
    return wrapper, values, labels


def _explicit_logits(model, values):
    """Evaluate RNN/LSTM/GRU equations without recurrent kernels or autograd."""
    recurrent = model.recurrent
    hidden = values.new_zeros(values.size(0), recurrent.hidden_size)
    cell = torch.zeros_like(hidden)
    for value in values.unbind(1):
        inputs = F.linear(value, recurrent.weight_ih_l0, recurrent.bias_ih_l0)
        previous = F.linear(hidden, recurrent.weight_hh_l0, recurrent.bias_hh_l0)
        if isinstance(recurrent, nn.LSTM):
            input_gate, forget_gate, candidate, output_gate = (inputs + previous).chunk(4, 1)
            cell = forget_gate.sigmoid() * cell + input_gate.sigmoid() * candidate.tanh()
            hidden = output_gate.sigmoid() * cell.tanh()
        elif isinstance(recurrent, nn.GRU):
            input_reset, input_update, input_new = inputs.chunk(3, 1)
            hidden_reset, hidden_update, hidden_new = previous.chunk(3, 1)
            reset = (input_reset + hidden_reset).sigmoid()
            update = (input_update + hidden_update).sigmoid()
            candidate = (input_new + reset * hidden_new).tanh()
            hidden = (1 - update) * candidate + update * hidden
        else:
            hidden = (inputs + previous).tanh()
    return model.head(hidden)


@CUDA_CUDNN
@pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["full", "diagonal"])
def test_cuda_recurrent_curvature_matches_explicit_equations(kind, dtype, mode):
    """Both derivative modes agree with independent scalar loss differences."""
    wrapper, values, labels = _fixture(kind, dtype, "cuda")
    original_values = values.clone()
    original_pointer = values.data_ptr()
    gradient, hessian = wrapper.compute_mask_derivatives([(values, labels)], F.cross_entropy, mode)
    reference = deepcopy(wrapper.model).cpu().double()
    mask = wrapper.mask.cpu().double()
    values64 = values.double()

    def loss(candidate):
        with torch.no_grad():
            return F.cross_entropy(_explicit_logits(reference, values64 * candidate), labels).item()

    step = 1e-3
    directions = torch.eye(4, dtype=torch.float64) * step
    expected_gradient = np.array([
        (loss(mask + delta) - loss(mask - delta)) / (2 * step) for delta in directions
    ])
    expected_hessian = np.array([
        [(loss(mask + left + right) - loss(mask + left - right)
          - loss(mask - left + right) + loss(mask - left - right)) / (4 * step ** 2)
         for right in directions] for left in directions
    ])
    if mode == "diagonal":
        expected_hessian = np.diag(np.diag(expected_hessian))
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-3, atol=2e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=3e-3, atol=3e-6)
    assert values.data_ptr() == original_pointer
    torch.testing.assert_close(values, original_values, rtol=0, atol=0)


@CUDA_CUDNN
@pytest.mark.parametrize("failure", [None, "forward", "loss", "hessian"])
def test_cuda_recurrent_backend_and_state_restore(monkeypatch, failure):
    """Nondefault caller flags, existing gradients and RNG survive all exits."""
    wrapper, values, labels = _fixture("LSTM", torch.float64, "cuda")
    optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=0.01)
    wrapper.fit_weights([(values, labels)], F.cross_entropy, optimizer)
    gradients = [(parameter.grad, parameter.grad.clone()) for parameter in wrapper.parameters()]
    state = {name: value.clone() for name, value in wrapper.state_dict().items()}
    optimizer_state = deepcopy(optimizer.state_dict())
    original_requires_grad = [parameter.requires_grad for parameter in wrapper.parameters()]
    observed = []
    sentinel = RuntimeError("curvature sentinel")

    def observe(_model, _inputs):
        observed.append(_flags())
        if failure == "forward":
            raise sentinel

    def loss(prediction, target):
        if failure == "loss":
            raise sentinel
        return F.cross_entropy(prediction, target)

    def fail_hessian(*_args):
        raise sentinel

    if failure == "hessian":
        monkeypatch.setattr(wrapper, "_hessian_row", fail_hessian)
    hook = wrapper.model.register_forward_pre_hook(observe)
    try:
        with torch.backends.cudnn.flags(
            enabled=True, benchmark=True, benchmark_limit=7,
            deterministic=True, allow_tf32=False,
        ):
            flags = _flags()
            rng = _rng_state()
            if failure is None:
                wrapper.compute_mask_derivatives([(values, labels)], loss)
            else:
                with pytest.raises(RuntimeError) as error:
                    wrapper.compute_mask_derivatives([(values, labels)], loss)
                assert error.value is sentinel
            assert _flags() == flags
            assert observed == [(False, *flags[1:])]
            _assert_rng_state(rng)
    finally:
        hook.remove()
    assert wrapper.training and wrapper.model.training
    assert [parameter.requires_grad for parameter in wrapper.parameters()] == original_requires_grad
    for parameter, (original, value) in zip(wrapper.parameters(), gradients):
        assert parameter.grad is original
        torch.testing.assert_close(parameter.grad, value, rtol=0, atol=0)
    for name, value in wrapper.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    actual_optimizer = optimizer.state_dict()
    assert actual_optimizer["param_groups"] == optimizer_state["param_groups"]
    for key, fields in optimizer_state["state"].items():
        for field, expected in fields.items():
            torch.testing.assert_close(
                actual_optimizer["state"][key][field], expected, rtol=0, atol=0
            )


@CUDA_CUDNN
def test_automatic_recurrent_refresh_preserves_training_backend():
    """Weight training continues with caller flags across real local mask updates."""
    wrapper, values, labels = _fixture("RNN", torch.float32, "cuda")
    wrapper.mask_update_epochs = 1
    optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=0.01)
    flags = _flags()
    observed = []

    def observe(model, _inputs):
        trainable = any(parameter.requires_grad for parameter in model.parameters())
        observed.append((trainable, _flags()))

    hook = wrapper.model.register_forward_pre_hook(observe)
    try:
        loss = wrapper.fit_weights([(values, labels)], F.cross_entropy, optimizer, train_epochs=2)
    finally:
        hook.remove()
    assert np.isfinite(loss)
    assert all(state["step"].item() == 2 for state in optimizer.state.values())
    assert _flags() == flags
    training = [choice for trainable, choice in observed if trainable]
    curvature = [choice for trainable, choice in observed if not trainable]
    assert len(training) == len(curvature) == 2
    assert all(choice == flags for choice in training)
    assert all(choice == (False, *flags[1:]) for choice in curvature)


def test_cpu_recurrence_keeps_cudnn_flags():
    """CPU recurrence does not enter the CUDA backend override."""
    wrapper, values, labels = _fixture("RNN", torch.float64, "cpu")
    flags = _flags()
    observed = []
    hook = wrapper.model.register_forward_pre_hook(lambda *_args: observed.append(_flags()))
    try:
        wrapper.compute_mask_derivatives([(values, labels)], F.cross_entropy)
    finally:
        hook.remove()
    assert observed == [flags]
    assert _flags() == flags


@CUDA_CUDNN
def test_cuda_nonrecurrent_model_keeps_cudnn_flags():
    """Other CUDA models retain their cuDNN selection during curvature."""
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2)).cuda()
    wrapper = FeatureSelectionWrapper(model, 4).cuda()
    values = torch.randn(7, 4)
    labels = torch.tensor([0, 1, 1, 0, 1, 0, 1])
    flags = _flags()
    observed = []
    hook = model.register_forward_pre_hook(lambda *_args: observed.append(_flags()))
    try:
        wrapper.compute_mask_derivatives([(values, labels)], F.cross_entropy)
    finally:
        hook.remove()
    assert observed == [flags]
    assert _flags() == flags
