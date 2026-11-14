"""Gibbs scan filtering preserves samples and RNG without scalar clamp checks."""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from kaiwu.torch_plugin import BoltzmannMachine


def model():
    """Use exactly representable nonzero couplings for an independent scan oracle."""
    weights = (torch.arange(49, dtype=torch.float32).reshape(7, 7) % 9 - 4) / 32
    bias = (torch.arange(7, dtype=torch.float32) - 3) / 16
    return BoltzmannMachine(7, quadratic_coef=weights, linear_bias=bias, device="cpu")


def reference(machine, sweeps, visible, batch=4):
    """Apply a full random scan with scalar Python indices and explicit local fields."""
    state = torch.bernoulli(torch.full((batch, machine.num_nodes), .5))
    fixed = 0 if visible is None else visible.shape[1]
    if visible is not None:
        state[:, :fixed] = visible
    upper = machine.quadratic_coef.detach().triu(1)
    coupling = upper + upper.T
    with torch.no_grad():
        for _ in range(sweeps):
            for index in torch.randperm(machine.num_nodes).tolist():
                if index >= fixed:
                    # Sum dyadic terms directly, independent of the production matmul.
                    field = (state * coupling[:, index]).sum(-1) + machine.linear_bias[index]
                    state[:, index] = (torch.sigmoid(field) > torch.rand_like(field)).float()
    return state


@pytest.mark.parametrize("sweeps", [0, 3])
@pytest.mark.parametrize("clamped", [None, 3, 7])
def test_scan_filter_preserves_reference_samples_and_rng(sweeps, clamped):
    machine = model()
    torch.manual_seed(44)
    visible = None if clamped is None else torch.randint(0, 5, (4, clamped)).float() / 4
    original_input = None if visible is None else visible.clone()
    parameters = {name: parameter.detach().clone() for name, parameter in machine.named_parameters()}
    torch.manual_seed(2026)
    expected = reference(machine, sweeps, visible)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(2026)
    actual = machine.gibbs_sample(num_steps=sweeps, s_visible=visible,
                                 num_sample=4 if visible is None else None)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert not actual.requires_grad
    if visible is not None:
        assert torch.equal(visible, original_input)
        assert torch.equal(actual[:, :clamped], visible)
    for name, parameter in machine.named_parameters():
        assert torch.equal(parameter, parameters[name])
        assert parameter.grad is None


def test_clamped_scan_has_bounded_tensor_boolean_decisions():
    machine = model()
    visible = torch.tensor([[0., 1., .5], [1., 0., .25]])
    with profile(activities=[ProfilerActivity.CPU]) as captured:
        machine.gibbs_sample(num_steps=3, s_visible=visible)
    counts = {event.key: event.count for event in captured.key_averages()}
    # The old loop made 7 * 3 Tensor-to-bool decisions. Bound scalar control
    # flow per scan, allowing efficient alternatives with any vector predicate.
    assert counts.get("aten::is_nonzero", 0) <= 3
