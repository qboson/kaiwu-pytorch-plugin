"""Preserve custom coupling columns when sampling through a transpose view."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


class ColumnCouplings(BoltzmannMachine):
    """Expose a nonsymmetric public symmetry override in several storage layouts."""

    def __init__(self, coupling):
        super().__init__(
            4, linear_bias=torch.tensor([0., .25, -.25, .125]), device="cpu"
        )
        self.register_buffer("coupling", coupling)
        self.symmetry_calls = 0

    def symmetrized_quadratic_coef(self):
        self.symmetry_calls += 1
        return self.coupling


@pytest.mark.parametrize("layout", ["dense", "transposed", "offset"])
def test_single_free_node_uses_custom_column_without_changing_rng_or_gradients(layout):
    """Only node3 is free; its explicit column defines a loop-independent oracle."""
    coupling = torch.tensor([
        [0., .125, .25, 1.],
        [-.25, 0., -.125, -.5],
        [.125, .25, 0., .25],
        [-1., .75, -.5, 0.],
    ])
    if layout == "transposed":
        coupling = coupling.T.contiguous().T
    elif layout == "offset":
        storage = torch.zeros(9, 9)
        storage[1::2, 1::2] = coupling
        coupling = storage[1::2, 1::2]
        assert not coupling.is_contiguous() and coupling.storage_offset() > 0
    machine = ColumnCouplings(coupling)
    storage = torch.tensor([
        [0., 99., 1., 99., .5, 99.],
        [1., 99., .25, 99., 0., 99.],
        [.5, 99., 0., 99., 1., 99.],
        [.25, 99., .5, 99., .25, 99.],
    ])
    visible = storage[:, ::2].detach().requires_grad_()
    assert not visible.is_contiguous()
    visible.grad = torch.full_like(visible, .75)
    before_input = visible.detach().clone()
    before_input_grad = visible.grad.clone()
    parameters = {name: value.detach().clone() for name, value in machine.named_parameters()}
    for parameter in machine.parameters():
        parameter.grad = torch.full_like(parameter, .25)
    gradients = {name: value.grad.clone() for name, value in machine.named_parameters()}

    # The last column is explicitly [1, -.5, .25, 0]; using the last row is wrong.
    # No production matrix indexing or generic Gibbs loop is used by this oracle.
    with torch.no_grad():
        field = visible[:, 0] - .5 * visible[:, 1] + .25 * visible[:, 2] + .125
        torch.manual_seed(2027)
        expected = torch.bernoulli(torch.full((4, 4), .5))
        expected[:, :3] = visible
        for _ in range(5):
            torch.randperm(4)  # A full scan always updates the sole free node once.
            expected[:, 3] = (torch.sigmoid(field) > torch.rand_like(field)).float()
        expected_rng = torch.get_rng_state()

    torch.manual_seed(2027)
    actual = machine.gibbs_sample(num_steps=5, s_visible=visible)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert machine.symmetry_calls == 1
    assert not actual.requires_grad
    assert torch.equal(visible, before_input)
    assert torch.equal(visible.grad, before_input_grad)
    for name, value in machine.named_parameters():
        assert torch.equal(value, parameters[name])
        assert torch.equal(value.grad, gradients[name])
