"""Mathematical and boundary contract tests for ``RestrictedBoltzmannMachine``.

Every case is pure CPU: models are built with an explicit ``torch.device("cpu")``
argument, parameter tensors are passed explicitly, and seeded random work is
isolated with ``torch.random.fork_rng``. The suite pins the closed-form
Hamiltonian, exact analytic gradients, conditional layer probabilities,
bipartite Ising construction and the inherited sampling integration.
"""

from contextlib import AbstractContextManager, contextmanager
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin import abstract_boltzmann_machine as abm_module


class _RecordingSampler:
    """Deterministic mock sampler that records every Ising matrix passed to solve."""

    def __init__(
        self,
        solutions: Optional[Sequence[np.ndarray]] = None,
        error: Optional[Exception] = None,
        on_solve: Optional[Callable[[], None]] = None,
    ) -> None:
        self._solutions: List[np.ndarray] = list(solutions) if solutions is not None else []
        self._error: Optional[Exception] = error
        self._on_solve: Optional[Callable[[], None]] = on_solve
        self.received_matrices: List[np.ndarray] = []

    def solve(self, ising_mat: np.ndarray) -> np.ndarray:
        """Record the Ising matrix and return the next configured spin array."""
        call_index = len(self.received_matrices)
        self.received_matrices.append(np.array(ising_mat, copy=True))
        if self._on_solve is not None:
            self._on_solve()
        error = self._error
        if error is not None:
            raise error
        if self._solutions:
            return self._solutions[min(call_index, len(self._solutions) - 1)]
        return np.ones((1, ising_mat.shape[0]), dtype=np.float32)

    def call_count(self) -> int:
        """Return how many times solve() has been invoked."""
        return len(self.received_matrices)


def _make_context_spy(events: List[str]) -> Callable[[], AbstractContextManager]:
    """Build a context-manager factory that records enter and exit lifecycle events."""

    @contextmanager
    def _spy():
        """Record one enter event, delegate the body and always record the exit."""
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    return _spy


def _make_rbm(
    quadratic_coef: torch.Tensor,
    linear_bias: torch.Tensor,
) -> RestrictedBoltzmannMachine:
    """Build a pure-CPU machine from cloned explicit parameters."""
    return RestrictedBoltzmannMachine(
        num_visible=quadratic_coef.shape[0],
        num_hidden=quadratic_coef.shape[1],
        quadratic_coef=quadratic_coef.clone(),
        linear_bias=linear_bias.clone(),
        device=torch.device("cpu"),
    )


def _make_asymmetric_model(dtype: torch.dtype) -> RestrictedBoltzmannMachine:
    """Build the canonical 3x2 machine used by the multi-layer contract cases."""
    quadratic_coef = torch.tensor(
        [[0.5, -1.25], [2.0, 0.75], [-0.5, 1.5]],
        dtype=dtype,
    )
    linear_bias = torch.tensor([1.0, -2.0, 0.5, -1.5, 0.25], dtype=dtype)
    return _make_rbm(quadratic_coef, linear_bias)


def _all_binary_states(num_nodes: int, dtype: torch.dtype) -> Tuple[torch.Tensor, np.ndarray]:
    """Enumerate all 2**num_nodes binary states and their augmented Ising spins."""
    rows = [
        [(mask >> bit) & 1 for bit in range(num_nodes)]
        for mask in range(1 << num_nodes)
    ]
    binary = torch.tensor(rows, dtype=dtype)
    ising = np.hstack([2.0 * binary.numpy() - 1.0, np.ones((len(rows), 1))])
    return binary, ising


def test_init_default_parameters_and_seeded_quadratic_scale() -> None:
    """Default construction seeds a 0.01-scaled quadratic term and an all-zero bias."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        expected_quadratic = torch.randn((3, 5)) * 0.01
        torch.manual_seed(2026)
        model = RestrictedBoltzmannMachine(3, 5, device=torch.device("cpu"))

    assert model.num_visible == 3
    assert model.num_hidden == 5
    assert model.num_nodes == 8
    assert set(dict(model.named_parameters())) == {"quadratic_coef", "linear_bias"}
    assert isinstance(model.quadratic_coef, torch.nn.Parameter)
    assert isinstance(model.linear_bias, torch.nn.Parameter)
    assert model.quadratic_coef.requires_grad is True
    assert model.linear_bias.requires_grad is True
    assert torch.equal(model.quadratic_coef, expected_quadratic)
    assert torch.equal(model.linear_bias, torch.zeros(8))


def test_init_custom_and_partial_parameters_wrap_as_nn_parameters() -> None:
    """Explicit and partially explicit parameters are wrapped as trainable parameters."""
    quadratic_coef = torch.tensor([[0.5, -1.0], [2.5, 3.0], [-4.0, 0.25]])
    linear_bias = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        full_model = _make_rbm(quadratic_coef, linear_bias)
        torch.manual_seed(7)
        quadratic_only = RestrictedBoltzmannMachine(
            3, 2, quadratic_coef=quadratic_coef.clone(), device=torch.device("cpu")
        )
        torch.manual_seed(7)
        bias_only = RestrictedBoltzmannMachine(
            3, 2, linear_bias=linear_bias.clone(), device=torch.device("cpu")
        )

    assert isinstance(full_model.quadratic_coef, torch.nn.Parameter)
    assert isinstance(full_model.linear_bias, torch.nn.Parameter)
    assert full_model.quadratic_coef.requires_grad is True
    assert full_model.linear_bias.requires_grad is True
    assert torch.equal(full_model.quadratic_coef, quadratic_coef)
    assert torch.equal(full_model.linear_bias, linear_bias)
    assert torch.equal(quadratic_only.linear_bias, torch.zeros(5))
    assert bias_only.quadratic_coef.shape == (3, 2)
    assert torch.isfinite(bias_only.quadratic_coef).all()
    assert torch.count_nonzero(bias_only.quadratic_coef) > 0
    assert torch.equal(bias_only.linear_bias, linear_bias)


def test_init_minimum_1x1_model_supports_all_core_operations() -> None:
    """The smallest 1x1 machine still honours the full forward, propagation and Ising API."""
    model = _make_rbm(torch.tensor([[2.0]]), torch.tensor([1.0, -3.0]))
    states = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    visible = torch.tensor([[1.0]])
    hidden = torch.tensor([[1.0]])

    assert model.num_nodes == 2
    assert torch.equal(model.visible_bias, torch.tensor([1.0]))
    assert torch.equal(model.hidden_bias, torch.tensor([-3.0]))
    torch.testing.assert_close(model(states), torch.tensor([0.0, -1.0, 3.0, 0.0]))
    torch.testing.assert_close(
        model.get_hidden(visible),
        torch.tensor([[1.0, torch.sigmoid(torch.tensor(-1.0)).item()]]),
    )
    torch.testing.assert_close(
        model.get_visible(hidden),
        torch.tensor([[torch.sigmoid(torch.tensor(3.0)).item(), 1.0]]),
    )
    np.testing.assert_array_equal(
        model.get_ising_matrix(),
        np.array([[0.0, 0.25, 0.5], [0.25, 0.0, -0.5], [0.5, -0.5, 0.0]], dtype=np.float32),
    )


def test_visible_and_hidden_bias_properties_partition_linear_bias_and_backpropagate() -> None:
    """The two bias views partition the packed bias and carry gradients back to it."""
    model = _make_rbm(
        torch.tensor([[0.5, -1.0], [2.0, 0.5], [-0.25, 1.5]]),
        torch.tensor([0.5, -1.0, 2.0, -3.5, 4.0]),
    )

    assert model.visible_bias.shape == (3,)
    assert model.hidden_bias.shape == (2,)
    torch.testing.assert_close(model.visible_bias, torch.tensor([0.5, -1.0, 2.0]))
    torch.testing.assert_close(model.hidden_bias, torch.tensor([-3.5, 4.0]))
    assert torch.equal(
        torch.cat([model.visible_bias, model.hidden_bias]), model.linear_bias.detach()
    )

    loss = (model.visible_bias * torch.tensor([2.0, 3.0, 5.0])).sum()
    loss = loss + (model.hidden_bias * torch.tensor([7.0, 11.0])).sum()
    loss.backward()

    assert torch.equal(model.linear_bias.grad, torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0]))


def test_clip_parameters_bounds_linear_by_h_range_and_quadratic_by_j_range_inplace() -> None:
    """Clipping maps h_range onto the linear bias and j_range onto the quadratic term."""
    model = _make_rbm(
        torch.tensor([[-5.0, -2.0, 0.0], [1.0, 3.0, 5.0]]),
        torch.tensor([-4.0, -1.0, 0.5, 2.5, 4.0]),
    )
    linear_parameter = model.linear_bias
    quadratic_parameter = model.quadratic_coef

    assert model.clip_parameters((-1.5, 2.0), (-0.5, 1.0)) is None
    assert model.linear_bias is linear_parameter
    assert model.quadratic_coef is quadratic_parameter
    torch.testing.assert_close(
        model.linear_bias, torch.tensor([-1.5, -1.0, 0.5, 2.0, 2.0])
    )
    torch.testing.assert_close(
        model.quadratic_coef, torch.tensor([[-0.5, -0.5, 0.0], [1.0, 1.0, 1.0]])
    )


def test_clip_parameters_one_sided_none_bounds_and_invalid_none_errors() -> None:
    """One-sided bounds clamp only the addressed side while invalid ranges raise."""
    model = _make_rbm(
        torch.tensor([[-2.0, 0.0, 2.0], [3.0, -3.0, 0.5]]),
        torch.tensor([-3.0, -0.5, 0.25, 2.5, 3.0]),
    )

    model.clip_parameters((None, 1.0), (-1.5, None))
    torch.testing.assert_close(model.linear_bias, torch.tensor([-3.0, -0.5, 0.25, 1.0, 1.0]))
    torch.testing.assert_close(
        model.quadratic_coef, torch.tensor([[-1.5, 0.0, 2.0], [3.0, -1.5, 0.5]])
    )

    clipped_linear = model.linear_bias.detach().clone()
    with pytest.raises(RuntimeError):
        model.clip_parameters((None, None), (-1.0, 1.0))
    assert torch.equal(model.linear_bias.detach(), clipped_linear)

    with pytest.raises(TypeError):
        model.clip_parameters(None, (-1.0, 1.0))
    with pytest.raises(TypeError):
        model.clip_parameters((-1.0, 1.0), None)


def test_forward_matches_closed_form_hamiltonian_for_asymmetric_rbm() -> None:
    """Forward returns the exact bipartite closed-form energy for any real-valued batch."""
    model = _make_asymmetric_model(torch.float64)
    states = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.25, -0.75, 1.5, 0.5, -1.25],
            [-1.0, 2.0, -0.5, 1.75, 0.125],
        ],
        dtype=torch.float64,
    )
    visible = states[:, :3]
    hidden = states[:, 3:]
    expected = (
        -(visible @ model.visible_bias)
        - (hidden @ model.hidden_bias)
        - torch.sum((visible @ model.quadratic_coef) * hidden, dim=-1)
    )

    energy = model(states)

    assert energy.shape == (4,)
    assert energy.dtype == torch.float64
    torch.testing.assert_close(energy, expected, rtol=1e-12, atol=1e-12)


def test_forward_zero_input_zero_weights_and_bipartite_layer_decoupling() -> None:
    """The energy vanishes for empty states and depends only on cross-layer couplings."""
    model = _make_asymmetric_model(torch.float32)
    visible = torch.tensor([[1.0, 0.0, 1.0], [0.5, -1.0, 2.0], [0.0, 0.0, 1.0]])
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    zero_state = torch.zeros((3, 5))
    assert torch.equal(model(zero_state), torch.zeros(3))

    zero_model = _make_rbm(torch.zeros((3, 2)), torch.zeros(5))
    assert torch.equal(
        zero_model(torch.tensor([[1.0, -2.0, 3.0, 0.5, -1.0]])), torch.zeros(1)
    )

    only_visible = torch.cat([visible, torch.zeros((3, 2))], dim=1)
    only_hidden = torch.cat([torch.zeros((3, 3)), hidden], dim=1)
    torch.testing.assert_close(
        model(only_visible), -(visible @ model.visible_bias), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        model(only_hidden), -(hidden @ model.hidden_bias), rtol=1e-6, atol=1e-6
    )

    biased = _make_rbm(torch.zeros((3, 2)), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(biased(only_visible), -(visible @ biased.visible_bias))


def test_forward_backpropagates_exact_closed_form_gradients() -> None:
    """Autograd reproduces the analytic energy gradients on every parameter and input."""
    model = _make_asymmetric_model(torch.float64)
    states = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.5, -1.5, 2.0, 0.25, -0.75],
            [-1.0, 0.5, 0.25, 1.5, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    visible = states.detach()[:, :3]
    hidden = states.detach()[:, 3:]

    model(states).sum().backward()

    torch.testing.assert_close(
        model.linear_bias.grad, -states.detach().sum(dim=0), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        model.quadratic_coef.grad, -(visible.t() @ hidden), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        states.grad[:, :3],
        -(model.visible_bias.detach() + hidden @ model.quadratic_coef.detach().t()),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        states.grad[:, 3:],
        -(model.hidden_bias.detach() + visible @ model.quadratic_coef.detach()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_objective_matches_closed_form_value_and_exact_cd_gradients() -> None:
    """The contrastive objective is the mean energy gap and its gradient is data minus model."""
    model = _make_asymmetric_model(torch.float64)
    positive = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float64
    )
    negative = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )

    assert float(model.objective(positive, positive)) == 0.0

    loss = model.objective(positive, negative)
    expected_loss = model(positive).mean() - model(negative).mean()
    assert loss.ndim == 0
    torch.testing.assert_close(loss, expected_loss, rtol=1e-12, atol=1e-12)

    loss.backward()

    torch.testing.assert_close(
        model.linear_bias.grad,
        negative.mean(dim=0) - positive.mean(dim=0),
        rtol=1e-12,
        atol=1e-12,
    )
    expected_quadratic = (negative[:, :3].t() @ negative[:, 3:]) / 5.0
    expected_quadratic = expected_quadratic - (positive[:, :3].t() @ positive[:, 3:]) / 2.0
    torch.testing.assert_close(
        model.quadratic_coef.grad, expected_quadratic, rtol=1e-12, atol=1e-12
    )


def test_get_hidden_probabilities_match_closed_form_sigmoid_and_preserve_input() -> None:
    """Hidden probabilities follow the exact sigmoid law and never mutate the input."""
    model = _make_asymmetric_model(torch.float32)
    visible = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.5, -1.0, 2.0], [1.0, 1.0, 1.0]]
    )
    visible_before = visible.clone()

    activated = model.get_hidden(visible)
    expected_hidden = torch.sigmoid(visible @ model.quadratic_coef + model.hidden_bias)

    assert activated.shape == (4, 5)
    assert activated.dtype == torch.float32
    assert activated.requires_grad is False
    assert torch.equal(activated[:, :3], visible)
    torch.testing.assert_close(activated[:, 3:], expected_hidden, rtol=1e-6, atol=1e-6)
    assert torch.equal(visible, visible_before)

    zero_model = _make_rbm(torch.zeros((3, 2)), torch.zeros(5))
    flat = zero_model.get_hidden(visible)
    assert torch.equal(flat[:, 3:], torch.full((4, 2), 0.5))
    assert torch.equal(flat[:, :3], visible)


def test_get_visible_probabilities_match_closed_form_sigmoid_and_preserve_input() -> None:
    """Visible probabilities follow the exact transposed sigmoid law without mutating input."""
    model = _make_asymmetric_model(torch.float32)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -1.0], [1.0, 1.0]])
    hidden_before = hidden.clone()

    activated = model.get_visible(hidden)
    expected_visible = torch.sigmoid(hidden @ model.quadratic_coef.t() + model.visible_bias)

    assert activated.shape == (4, 5)
    assert activated.dtype == torch.float32
    assert activated.requires_grad is False
    assert torch.equal(activated[:, 3:], hidden)
    torch.testing.assert_close(activated[:, :3], expected_visible, rtol=1e-6, atol=1e-6)
    assert torch.equal(hidden, hidden_before)

    zero_model = _make_rbm(torch.zeros((3, 2)), torch.zeros(5))
    flat = zero_model.get_visible(hidden)
    assert torch.equal(flat[:, :3], torch.full((4, 3), 0.5))
    assert torch.equal(flat[:, 3:], hidden)


def test_conditional_probabilities_match_hamiltonian_single_unit_flip_differences() -> None:
    """Conditional probabilities equal the sigmoid of the single-unit energy drop."""
    model = _make_asymmetric_model(torch.float32)
    visible = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    hidden_prob = model.get_hidden(visible)
    visible_prob = model.get_visible(hidden)

    for unit in range(model.num_hidden):
        hidden_off = hidden.clone()
        hidden_on = hidden.clone()
        hidden_off[:, unit] = 0.0
        hidden_on[:, unit] = 1.0
        delta = model(torch.cat([visible, hidden_off], dim=1))
        delta = delta - model(torch.cat([visible, hidden_on], dim=1))
        torch.testing.assert_close(
            hidden_prob[:, model.num_visible + unit],
            torch.sigmoid(delta),
            rtol=1e-6,
            atol=1e-6,
        )

    for unit in range(model.num_visible):
        visible_off = visible.clone()
        visible_on = visible.clone()
        visible_off[:, unit] = 0.0
        visible_on[:, unit] = 1.0
        delta = model(torch.cat([visible_off, hidden], dim=1))
        delta = delta - model(torch.cat([visible_on, hidden], dim=1))
        torch.testing.assert_close(
            visible_prob[:, unit], torch.sigmoid(delta), rtol=1e-6, atol=1e-6
        )


def test_get_hidden_requires_grad_exact_gradients_and_nested_no_grad_override() -> None:
    """An explicit grad request overrides an outer no-grad scope with exact Jacobians."""
    model = _make_asymmetric_model(torch.float32)
    visible = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.5, -1.0, 2.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    visible_values = visible.detach()

    with torch.no_grad():
        activated = model.get_hidden(visible, requires_grad=True, bernoulli=False)
    with torch.no_grad():
        frozen = model.get_hidden(visible, requires_grad=False, bernoulli=False)

    assert activated.requires_grad is True
    assert frozen.requires_grad is False

    activated[:, 3:].sum().backward()

    logits = visible_values @ model.quadratic_coef.detach() + model.hidden_bias.detach()
    prob = torch.sigmoid(logits)
    derivative = prob * (1.0 - prob)
    torch.testing.assert_close(
        model.quadratic_coef.grad, visible_values.t() @ derivative, rtol=1e-4, atol=1e-6
    )
    assert torch.equal(model.linear_bias.grad[:3], torch.zeros(3))
    torch.testing.assert_close(
        model.linear_bias.grad[3:], derivative.sum(dim=0), rtol=1e-4, atol=1e-6
    )
    torch.testing.assert_close(
        visible.grad, derivative @ model.quadratic_coef.detach().t(), rtol=1e-4, atol=1e-6
    )


def test_get_hidden_bernoulli_determinism_extremes_and_non_differentiable_sampling() -> None:
    """Bernoulli hidden sampling is seed reproducible, saturated and gradient-free."""
    model = _make_asymmetric_model(torch.float32)
    visible = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        first = model.get_hidden(visible, bernoulli=True)
        torch.manual_seed(31)
        second = model.get_hidden(visible, bernoulli=True)

        positive_coupling = _make_rbm(
            torch.tensor([[50.0, -50.0], [50.0, -50.0], [50.0, -50.0]]), torch.zeros(5)
        )
        negative_coupling = _make_rbm(
            torch.tensor([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]]), torch.zeros(5)
        )
        saturated = positive_coupling.get_hidden(torch.ones((2, 3)), bernoulli=True)
        saturated_flipped = negative_coupling.get_hidden(torch.ones((2, 3)), bernoulli=True)

    assert torch.equal(first, second)
    assert torch.equal(first[:, :3], visible)
    assert set(torch.unique(first[:, 3:]).tolist()) <= {0.0, 1.0}
    assert torch.equal(saturated[:, 3:], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert torch.equal(saturated_flipped[:, 3:], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

    detached_output = model.get_hidden(visible, requires_grad=True, bernoulli=True)
    assert detached_output.requires_grad is False

    trainable = visible.clone().requires_grad_(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        sampled = model.get_hidden(trainable, requires_grad=True, bernoulli=True)
    assert sampled.requires_grad is True
    sampled.sum().backward()
    assert torch.equal(trainable.grad, torch.ones_like(trainable))
    assert model.quadratic_coef.grad is None
    assert model.linear_bias.grad is None


def test_get_visible_bernoulli_determinism_extremes_and_no_grad() -> None:
    """Visible sampling is seed reproducible, saturated and always detached."""
    model = _make_asymmetric_model(torch.float32)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        first = model.get_visible(hidden, bernoulli=True)
        torch.manual_seed(47)
        second = model.get_visible(hidden, bernoulli=True)

        saturated_model = _make_rbm(
            torch.tensor([[50.0, -50.0], [-50.0, 50.0], [50.0, -50.0]]), torch.zeros(5)
        )
        saturated = saturated_model.get_visible(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), bernoulli=True
        )

    assert torch.equal(first, second)
    assert torch.equal(first[:, 3:], hidden)
    assert set(torch.unique(first[:, :3]).tolist()) <= {0.0, 1.0}
    assert torch.equal(saturated[:, :3], torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))

    trainable = torch.tensor([[1.0, 0.0]], requires_grad=True)
    assert model.get_visible(trainable, bernoulli=False).requires_grad is False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(59)
        assert model.get_visible(trainable, bernoulli=True).requires_grad is False


def test_alternating_block_gibbs_chain_determinism_and_strong_coupling_convergence() -> None:
    """Alternating block updates are bitwise reproducible and converge to the ground state."""
    model = _make_asymmetric_model(torch.float32)
    start = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(97)
        zero_step = model.get_hidden(start, bernoulli=True)
        chain = zero_step
        for _ in range(4):
            chain = model.get_visible(chain[:, 3:], bernoulli=True)
            chain = model.get_hidden(chain[:, :3], bernoulli=True)
        torch.manual_seed(97)
        replay = model.get_hidden(start, bernoulli=True)
        for _ in range(4):
            replay = model.get_visible(replay[:, 3:], bernoulli=True)
            replay = model.get_hidden(replay[:, :3], bernoulli=True)

    assert torch.equal(zero_step[:, :3], start)
    assert torch.equal(chain, replay)
    assert set(torch.unique(chain[:, :3]).tolist()) <= {0.0, 1.0}

    strong_model = _make_rbm(
        torch.tensor([[50.0, 50.0], [-50.0, -50.0], [50.0, 50.0]]), torch.zeros(5)
    )
    first_state = torch.tensor([[1.0, 1.0, 1.0]])
    second_state = torch.tensor([[1.0, 0.0, 0.0]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        first_step = strong_model.get_hidden(first_state, bernoulli=True)
        first_step = strong_model.get_visible(first_step[:, 3:], bernoulli=True)
        second_step = strong_model.get_hidden(second_state, bernoulli=True)
        second_step = strong_model.get_visible(second_step[:, 3:], bernoulli=True)

    ground_state = torch.tensor([[1.0, 0.0, 1.0]])
    assert not torch.equal(first_state, second_state)
    assert torch.equal(first_step[:, :3], ground_state)
    assert torch.equal(second_step[:, :3], ground_state)


def test_get_ising_matrix_exact_bipartite_block_structure_symmetry_and_zero_diag() -> None:
    """The Ising matrix exposes only cross-layer couplings plus the augmented bias border."""
    model = _make_asymmetric_model(torch.float32)
    quadratic = model.quadratic_coef.detach().numpy()
    linear = model.linear_bias.detach().numpy()

    ising_mat = model.get_ising_matrix()

    assert isinstance(ising_mat, np.ndarray)
    assert ising_mat.shape == (6, 6)
    assert ising_mat.dtype == np.float32
    np.testing.assert_array_equal(ising_mat, ising_mat.T)
    np.testing.assert_array_equal(np.diag(ising_mat), np.zeros(6))
    np.testing.assert_array_equal(ising_mat[:3, :3], np.zeros((3, 3)))
    np.testing.assert_array_equal(ising_mat[3:5, 3:5], np.zeros((2, 2)))
    np.testing.assert_allclose(ising_mat[:3, 3:5], quadratic / 8.0, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(ising_mat[3:5, :3], quadratic.T / 8.0, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        ising_mat[:3, 5], linear[:3] / 4.0 + quadratic.sum(axis=1) / 8.0, rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        ising_mat[3:5, 5], linear[3:] / 4.0 + quadratic.sum(axis=0) / 8.0, rtol=0.0, atol=1e-7
    )
    np.testing.assert_array_equal(ising_mat[5, :3], ising_mat[:3, 5])


def test_get_ising_matrix_preserves_hamiltonian_across_all_binary_states() -> None:
    """Every one of the 2**5 binary states satisfies the exact quadratic-form offset identity."""
    model = _make_asymmetric_model(torch.float32)
    ising_mat = model.get_ising_matrix().astype(np.float64)
    binary, ising = _all_binary_states(5, torch.float32)
    energies = model(binary).detach().numpy().astype(np.float64)

    quadratic_form = np.einsum("ij,jk,ik->i", ising, ising_mat, ising)
    offset = -quadratic_form - energies
    expected_offset = 0.5 * float(model.linear_bias.detach().sum())
    expected_offset = expected_offset + 0.25 * float(model.quadratic_coef.detach().sum())

    assert binary.shape == (32, 5)
    assert ising.shape == (32, 6)
    np.testing.assert_allclose(offset, np.full(32, expected_offset), rtol=0.0, atol=1e-5)


def test_get_ising_matrix_zero_parameters_returns_zero_and_preserves_grad_state() -> None:
    """Zero parameters yield an all-zero matrix and the conversion mutates nothing."""
    zero_model = _make_rbm(torch.zeros((3, 2)), torch.zeros(5))

    np.testing.assert_array_equal(
        zero_model.get_ising_matrix(), np.zeros((6, 6), dtype=np.float32)
    )

    model = _make_asymmetric_model(torch.float32)
    model(torch.ones((2, 5))).sum().backward()
    quadratic_before = model.quadratic_coef.detach().clone()
    linear_before = model.linear_bias.detach().clone()
    quadratic_grad_before = model.quadratic_coef.grad.detach().clone()
    linear_grad_before = model.linear_bias.grad.detach().clone()

    model.get_ising_matrix()

    assert torch.equal(model.quadratic_coef.detach(), quadratic_before)
    assert torch.equal(model.linear_bias.detach(), linear_before)
    assert torch.equal(model.quadratic_coef.grad, quadratic_grad_before)
    assert torch.equal(model.linear_bias.grad, linear_grad_before)


def test_get_ising_matrix_returns_float32_even_for_float64_parameters() -> None:
    """The conversion always narrows to single precision regardless of parameter dtype."""
    model = _make_asymmetric_model(torch.float64)
    quadratic = model.quadratic_coef.detach().numpy()
    linear = model.linear_bias.detach().numpy()
    expected = np.zeros((6, 6), dtype=np.float64)
    expected[:3, 3:5] = quadratic / 8.0
    expected[3:5, :3] = quadratic.T / 8.0
    border = linear / 4.0 + expected.sum(axis=0)[:5]
    expected[:5, 5] = border
    expected[5, :5] = border

    ising_mat = model.get_ising_matrix()

    assert isinstance(ising_mat, np.ndarray)
    assert ising_mat.shape == (6, 6)
    assert ising_mat.dtype == np.float32
    assert model.quadratic_coef.dtype == torch.float64
    np.testing.assert_allclose(ising_mat, expected.astype(np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(ising_mat, ising_mat.T)
    np.testing.assert_array_equal(np.diag(ising_mat), np.zeros(6))
    np.testing.assert_array_equal(ising_mat[:3, :3], np.zeros((3, 3)))
    np.testing.assert_array_equal(ising_mat[3:5, 3:5], np.zeros((2, 2)))


def test_sample_passes_exact_ising_matrix_maps_gauge_spins_and_sign_invariance() -> None:
    """Sampling forwards the exact Ising matrix and normalises gauge spins to the 0/1 basis."""
    model = _make_asymmetric_model(torch.float32)
    raw = np.array(
        [
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    sampler = _RecordingSampler([raw])

    samples = model.sample(sampler)

    assert sampler.call_count() == 1
    np.testing.assert_array_equal(sampler.received_matrices[0], model.get_ising_matrix())
    assert samples.shape == (2, 5)
    assert samples.dtype == torch.float32
    assert samples.device == torch.device("cpu")
    expected = torch.tensor((raw[:, :-1] * raw[:, [-1]] + 1.0) / 2.0, dtype=torch.float32)
    assert torch.equal(samples, expected)

    flipped = model.sample(_RecordingSampler([-raw]))
    assert torch.equal(samples, flipped)


def test_sample_wraps_solve_in_kpp_caller_context_and_cleans_up_on_error(monkeypatch) -> None:
    """The caller context brackets the solve call and still exits when solve raises."""
    model = _make_asymmetric_model(torch.float32)
    events: List[str] = []
    monkeypatch.setattr(abm_module, "kpp_caller_context", _make_context_spy(events))

    sampler = _RecordingSampler(on_solve=lambda: events.append("solve"))
    model.sample(sampler)
    assert events == ["enter", "solve", "exit"]

    events.clear()
    failing = _RecordingSampler(
        error=RuntimeError("rbm solve failed"), on_solve=lambda: events.append("solve")
    )
    with pytest.raises(RuntimeError, match="rbm solve failed"):
        model.sample(failing)
    assert events == ["enter", "solve", "exit"]
