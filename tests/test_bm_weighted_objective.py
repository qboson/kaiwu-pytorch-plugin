"""Phase histograms give the same energy estimates and gradients as repeated rows."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


@pytest.mark.parametrize("constructor", [lambda: BoltzmannMachine(3, device="cpu"),
                                       lambda: RestrictedBoltzmannMachine(2, 1, device="cpu")])
def test_histogram_matches_repeated_states_in_value_and_gradient(constructor):
    model = constructor().double()
    data = torch.tensor([[0., 0., 0.], [1., 1., 1.]], dtype=torch.float64)
    sampled = torch.tensor([[0., 1., 1.], [1., 0., 1.], [0., 0., 1.]], dtype=torch.float64)
    data_counts = torch.tensor([5, 3])
    sample_counts = torch.tensor([2, 4, 0])
    baseline = model.objective(
        data.repeat_interleave(data_counts, dim=0), sampled.repeat_interleave(sample_counts, dim=0)
    )
    weights = sample_counts.double().requires_grad_()
    weighted = model.objective(data, sampled, positive_weights=data_counts, negative_weights=weights)
    torch.testing.assert_close(weighted, baseline)
    expected = torch.autograd.grad(baseline, tuple(model.parameters()))
    weighted.backward()
    for parameter, reference in zip(model.parameters(), expected):
        torch.testing.assert_close(parameter.grad, reference)
    assert weights.grad is None


def test_one_phase_can_use_large_weights():
    model = BoltzmannMachine(3, device="cpu")
    data = torch.eye(3)
    reference = model.objective(data, data[1:])
    actual = model.objective(data, data, negative_weights=[0.0, 2e38, 2e38])
    torch.testing.assert_close(actual, reference)
    very_large = torch.tensor([0.0, 1e300, 1e300], dtype=torch.float64)
    precise = model.objective(data, data, negative_weights=very_large)
    torch.testing.assert_close(precise.float(), reference)


@pytest.mark.parametrize(
    "weights, message",
    [([1.0], "one weight"), ([[1.0], [1.0]], "one weight"), ([0.0, 0.0], "positive total"),
     ([-1.0, 3.0], "nonnegative"), ([float("inf"), 1.0], "finite"), ([1j, 1.0], "real")],
)
def test_invalid_weights_raise_before_reducing(weights, message):
    model = BoltzmannMachine(3, device="cpu")
    states = torch.zeros(2, 3)
    with pytest.raises(ValueError, match=message):
        model.objective(states, states, positive_weights=weights)
