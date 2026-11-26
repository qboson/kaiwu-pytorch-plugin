"""Exact mathematical and boundary contract tests for ``BoltzmannMachine``."""

from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine
from kaiwu.torch_plugin import abstract_boltzmann_machine as abm_module
from kaiwu.torch_plugin import full_boltzmann_machine as fbm_module


class _RecordingSampler:
    """Deterministic sampler stub that records every Ising matrix it receives."""

    def __init__(
        self,
        solutions: Optional[Sequence[np.ndarray]] = None,
        error_message: Optional[str] = None,
        on_solve: Optional[Callable[[], None]] = None,
    ) -> None:
        self._solutions: List[np.ndarray] = list(solutions) if solutions is not None else []
        self._error_message = error_message
        self._on_solve = on_solve
        self.received_matrices: List[np.ndarray] = []

    def solve(self, ising_mat: np.ndarray) -> np.ndarray:
        """Record the Ising matrix and return the configured spin rows."""
        call_index = len(self.received_matrices)
        self.received_matrices.append(np.array(ising_mat, copy=True))
        if self._on_solve is not None:
            self._on_solve()
        if self._error_message is not None:
            raise RuntimeError(self._error_message)
        if self._solutions:
            return self._solutions[min(call_index, len(self._solutions) - 1)]
        return np.ones((1, ising_mat.shape[0]), dtype=np.float32)

    def call_count(self) -> int:
        """Return how many times solve() has been called."""
        return len(self.received_matrices)


def _make_context_spy(events: List[str]) -> Callable[[], AbstractContextManager]:
    """Create a context-manager factory that records entry and exit events."""

    @contextmanager
    def _spy() -> Iterator[None]:
        """Append lifecycle markers around the wrapped block."""
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    return _spy


def _make_model(
    quadratic_coef: torch.Tensor, linear_bias: torch.Tensor
) -> BoltzmannMachine:
    """Create a pure-CPU BoltzmannMachine from cloned explicit parameters."""
    return BoltzmannMachine(
        num_nodes=linear_bias.shape[0],
        quadratic_coef=quadratic_coef.clone(),
        linear_bias=linear_bias.clone(),
        device=torch.device("cpu"),
    )


def _all_binary_states(
    num_nodes: int, dtype: torch.dtype
) -> Tuple[torch.Tensor, np.ndarray]:
    """Enumerate every 2**num_nodes binary state and its augmented Ising spins."""
    rows = [
        [(mask >> bit) & 1 for bit in range(num_nodes)]
        for mask in range(1 << num_nodes)
    ]
    binary = torch.tensor(rows, dtype=dtype)
    ising_spins = np.hstack([2.0 * binary.numpy() - 1.0, np.ones((len(rows), 1))])
    return binary, ising_spins


def _dense_quadratic() -> torch.Tensor:
    """Return a fixed asymmetric 4x4 float64 quadratic coefficient."""
    return torch.tensor(
        [
            [1.0, 2.0, -3.0, 4.0],
            [-5.0, 6.0, 7.0, -8.0],
            [9.0, -10.0, 11.0, 12.0],
            [-13.0, 14.0, -15.0, 16.0],
        ],
        dtype=torch.float64,
    )


def _dense_bias() -> torch.Tensor:
    """Return a fixed non-zero float64 linear bias for four nodes."""
    return torch.tensor([0.5, -1.5, 2.5, -3.5], dtype=torch.float64)


def _make_dense_model() -> BoltzmannMachine:
    """Create a 4-node float64 model with asymmetric weights and non-zero bias."""
    return _make_model(_dense_quadratic(), _dense_bias())


def _make_conditional_model() -> BoltzmannMachine:
    """Create a 5-node float64 model used for conditional submatrix checks."""
    quadratic_coef = torch.tensor(
        [
            [0.0, 1.25, -2.0, 0.75, 3.0],
            [-1.0, 0.0, 2.5, -3.25, 1.5],
            [2.0, -2.5, 0.0, 4.0, -1.75],
            [-0.75, 3.25, -4.0, 0.0, 2.25],
            [-3.0, -1.5, 1.75, -2.25, 0.0],
        ],
        dtype=torch.float64,
    )
    linear_bias = torch.tensor([0.75, -1.25, 2.0, -2.75, 3.5], dtype=torch.float64)
    return _make_model(quadratic_coef, linear_bias)


def _contrastive_batches() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return fixed float64 positive (2, 4) and negative (5, 4) spin batches."""
    positive = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]], dtype=torch.float64
    )
    negative = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    return positive, negative


def _closed_form_energy(
    states: torch.Tensor, quadratic_coef: torch.Tensor, linear_bias: torch.Tensor
) -> torch.Tensor:
    """Evaluate the Hamiltonian in its strict-upper-triangle closed form."""
    energies = -(states * linear_bias).sum(dim=-1)
    num_nodes = linear_bias.shape[0]
    for row in range(num_nodes):
        for col in range(row + 1, num_nodes):
            energies = (
                energies
                - quadratic_coef[row, col] * states[..., row] * states[..., col]
            )
    return energies


def _ising_components(
    quadratic_coef: torch.Tensor, linear_bias: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the exact quadratic block and bias vector of the Ising form."""
    with torch.no_grad():
        quadratic_block = (quadratic_coef / 8.0).numpy()
        bias_vector = (linear_bias / 4.0 + quadratic_coef.sum(dim=0) / 8.0).numpy()
    return quadratic_block, bias_vector


def _assemble_ising_matrix(
    quadratic_block: np.ndarray, bias_vector: np.ndarray
) -> np.ndarray:
    """Assemble a symmetric Ising matrix from a quadratic block and bias vector."""
    size = quadratic_block.shape[0]
    ising_mat = np.zeros((size + 1, size + 1), dtype=quadratic_block.dtype)
    ising_mat[:-1, :-1] = quadratic_block
    ising_mat[:-1, -1] = bias_vector
    ising_mat[-1, :-1] = bias_vector
    return ising_mat


def _assert_strict_upper_gradient(gradient: torch.Tensor) -> None:
    """Assert that a 4x4 gradient vanishes outside the strict upper triangle."""
    torch.testing.assert_close(
        torch.diagonal(gradient), torch.zeros(4, dtype=gradient.dtype)
    )
    torch.testing.assert_close(
        torch.tril(gradient, -1), torch.zeros((4, 4), dtype=gradient.dtype)
    )


# Group A: initialization and parameter registration.


def test_init_default_parameters_and_seeded_quadratic_scale() -> None:
    """Defaults give a seeded 0.01-scaled quadratic term and a zero bias."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        expected_quadratic = torch.randn((5, 5)) * 0.01
        torch.manual_seed(1234)
        model = BoltzmannMachine(5, device=torch.device("cpu"))

    assert model.num_nodes == 5
    parameters = dict(model.named_parameters())
    assert set(parameters) == {"quadratic_coef", "linear_bias"}
    for _, parameter in parameters.items():
        assert isinstance(parameter, torch.nn.Parameter)
        assert parameter.requires_grad is True
    torch.testing.assert_close(model.quadratic_coef, expected_quadratic)
    torch.testing.assert_close(model.linear_bias, torch.zeros(5))


def test_init_custom_and_partial_parameters_wrap_as_nn_parameters() -> None:
    """Explicit tensors are wrapped as parameters while missing ones default."""
    quadratic_coef = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    linear_bias = torch.tensor([0.5, -1.5, 2.5])
    cpu = torch.device("cpu")

    full = BoltzmannMachine(
        num_nodes=3, quadratic_coef=quadratic_coef, linear_bias=linear_bias, device=cpu
    )
    assert isinstance(full.quadratic_coef, torch.nn.Parameter)
    assert isinstance(full.linear_bias, torch.nn.Parameter)
    assert full.quadratic_coef.requires_grad is True
    assert full.linear_bias.requires_grad is True
    torch.testing.assert_close(full.quadratic_coef, quadratic_coef)
    torch.testing.assert_close(full.linear_bias, linear_bias)

    quadratic_only = BoltzmannMachine(3, quadratic_coef=quadratic_coef, device=cpu)
    assert quadratic_only.linear_bias.shape == (3,)
    torch.testing.assert_close(quadratic_only.linear_bias, torch.zeros(3))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2024)
        bias_only = BoltzmannMachine(3, linear_bias=linear_bias, device=cpu)
        torch.manual_seed(2024)
        expected_quadratic = torch.randn((3, 3)) * 0.01

    assert bias_only.quadratic_coef.shape == (3, 3)
    assert int(torch.count_nonzero(bias_only.quadratic_coef)) > 0
    torch.testing.assert_close(bias_only.quadratic_coef, expected_quadratic)
    torch.testing.assert_close(bias_only.linear_bias, linear_bias)


def test_init_minimum_single_node_model_supports_core_operations() -> None:
    """A one-node model keeps every core operation well defined."""
    model = _make_model(torch.tensor([[5.0]]), torch.tensor([2.0]))

    torch.testing.assert_close(
        model.symmetrized_quadratic_coef(), torch.zeros((1, 1))
    )
    energies = model(torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(energies, torch.tensor([0.0, -2.0]))

    ising_mat = model.get_ising_matrix()
    expected = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
    assert ising_mat.dtype == np.float32
    np.testing.assert_allclose(ising_mat, expected, rtol=0.0, atol=1e-6)


# Group B: symmetrization, clipping and bias slicing.


def test_symmetrized_quadratic_coef_mirrors_strict_upper_triangle_and_zeros_diagonal() -> None:
    """Only the strict upper triangle is mirrored; diagonal and lower part vanish."""
    quadratic_coef = torch.tensor(
        [
            [1.5, 2.0, -3.0, 4.0],
            [-0.5, 6.0, 7.5, -8.0],
            [9.0, -10.5, 11.0, 12.0],
            [-13.0, 14.5, -15.0, 16.0],
        ],
        dtype=torch.float64,
    )
    model = _make_model(quadratic_coef, torch.zeros(4, dtype=torch.float64))
    symmetrized = model.symmetrized_quadratic_coef()
    upper = quadratic_coef.triu(1)

    torch.testing.assert_close(symmetrized, upper + upper.t())
    torch.testing.assert_close(symmetrized, symmetrized.t())
    torch.testing.assert_close(
        torch.diagonal(symmetrized), torch.zeros(4, dtype=torch.float64)
    )

    weight_mask = torch.arange(1.0, 17.0, dtype=torch.float64).reshape(4, 4)
    (symmetrized * weight_mask).sum().backward()
    expected_gradient = torch.zeros((4, 4), dtype=torch.float64)
    for row in range(4):
        for col in range(row + 1, 4):
            expected_gradient[row, col] = weight_mask[row, col] + weight_mask[col, row]

    torch.testing.assert_close(model.quadratic_coef.grad, expected_gradient)


def test_clip_parameters_limits_linear_by_h_range_and_quadratic_by_j_range_inplace() -> None:
    """Clipping maps h_range to the bias and j_range to the quadratic term."""
    quadratic_coef = torch.linspace(-5.0, 5.0, 9).reshape(3, 3)
    linear_bias = torch.linspace(-4.0, 4.0, 3)
    model = _make_model(quadratic_coef, linear_bias)
    quadratic_reference = model.quadratic_coef
    bias_reference = model.linear_bias

    assert model.clip_parameters((-1.5, 2.0), (-0.5, 1.0)) is None
    assert model.quadratic_coef is quadratic_reference
    assert model.linear_bias is bias_reference
    torch.testing.assert_close(
        model.linear_bias, torch.clamp(linear_bias, -1.5, 2.0)
    )
    torch.testing.assert_close(
        model.quadratic_coef, torch.clamp(quadratic_coef, -0.5, 1.0)
    )
    assert model.linear_bias.max().item() == 2.0
    assert model.quadratic_coef.max().item() == 1.0


def test_clip_parameters_one_sided_none_bounds_and_invalid_none_errors() -> None:
    """One-sided ranges clamp a single endpoint while invalid ranges raise."""
    quadratic_coef = torch.tensor([[-5.0, 5.0], [2.0, -2.0]])
    linear_bias = torch.tensor([-3.0, 4.0])
    model = _make_model(quadratic_coef, linear_bias)

    model.clip_parameters((None, 1.0), (None, 1.0))
    torch.testing.assert_close(model.linear_bias, torch.tensor([-3.0, 1.0]))
    torch.testing.assert_close(
        model.quadratic_coef, torch.tensor([[-5.0, 1.0], [1.0, -2.0]])
    )

    model.clip_parameters((-1.5, None), (-1.5, None))
    torch.testing.assert_close(model.linear_bias, torch.tensor([-1.5, 1.0]))
    torch.testing.assert_close(
        model.quadratic_coef, torch.tensor([[-1.5, 1.0], [1.0, -1.5]])
    )

    with pytest.raises(RuntimeError):
        model.clip_parameters((None, None), (None, None))
    with pytest.raises(TypeError):
        model.clip_parameters(None, (-1.0, 1.0))


def test_visible_and_hidden_bias_partition_linear_bias_and_preserve_gradients() -> None:
    """Bias slices partition the parameter and keep the autograd path intact."""
    linear_bias = torch.tensor([0.5, -1.0, 2.0, -3.5, 4.0])
    model = _make_model(torch.zeros((5, 5)), linear_bias)
    visible = model.visible_bias(2)
    hidden = model.hidden_bias(3)

    torch.testing.assert_close(visible, torch.tensor([0.5, -1.0]))
    torch.testing.assert_close(hidden, torch.tensor([2.0, -3.5, 4.0]))
    torch.testing.assert_close(torch.cat([visible, hidden]), model.linear_bias)

    visible_weights = torch.tensor([2.0, 3.0])
    hidden_weights = torch.tensor([5.0, 7.0, 11.0])
    loss = (visible * visible_weights).sum() + (hidden * hidden_weights).sum()
    loss.backward()

    torch.testing.assert_close(
        model.linear_bias.grad, torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0])
    )


def test_visible_and_hidden_bias_zero_and_full_node_boundaries() -> None:
    """Zero and full node counts produce empty or complete bias views."""
    linear_bias = torch.tensor([1.0, -2.0, 3.0, -4.0])
    model = _make_model(torch.zeros((4, 4)), linear_bias)

    assert model.visible_bias(0).shape == (0,)
    assert model.hidden_bias(0).shape == (0,)
    torch.testing.assert_close(model.visible_bias(4), model.linear_bias)
    torch.testing.assert_close(model.hidden_bias(4), model.linear_bias)


# Group C: Hamiltonian energy and exact autograd gradients.


def test_forward_matches_closed_form_hamiltonian_for_batched_and_1d_inputs() -> None:
    """Energies match the closed form for a batch and for a single 1-D state."""
    quadratic_coef = torch.tensor(
        [
            [0.0, 1.25, -2.5, 0.75],
            [-3.0, 0.0, 0.5, -1.75],
            [4.25, -0.25, 0.0, 2.0],
            [-0.5, 3.5, -4.0, 0.0],
        ],
        dtype=torch.float64,
    )
    linear_bias = torch.tensor([0.25, -1.5, 2.0, -0.75], dtype=torch.float64)
    model = _make_model(quadratic_coef, linear_bias)
    states = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    energies = model(states)
    expected = _closed_form_energy(states, quadratic_coef, linear_bias)

    assert energies.shape == (3,)
    torch.testing.assert_close(energies, expected, rtol=0.0, atol=1e-12)

    single = model(states[0])
    assert single.shape == ()
    torch.testing.assert_close(single, energies[0], rtol=0.0, atol=1e-12)


def test_forward_zero_input_zero_weights_and_lower_triangle_invariance() -> None:
    """Zero states and triangular weights give zero energy; lower part is inert."""
    quadratic_coef = torch.tensor(
        [
            [1.0, 2.0, -3.0, 4.0],
            [-5.0, 6.0, 7.0, -8.0],
            [9.0, -10.0, 11.0, 12.0],
            [-13.0, 14.0, -15.0, 16.0],
        ]
    )
    linear_bias = torch.tensor([1.0, -2.0, 0.5, 3.0])
    model = _make_model(quadratic_coef, linear_bias)
    states = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]
    )

    torch.testing.assert_close(model(torch.zeros((3, 4))), torch.zeros(3))

    lower_only = torch.tensor(
        [
            [1.5, 0.0, 0.0, 0.0],
            [-2.0, 3.5, 0.0, 0.0],
            [4.5, -5.5, 6.5, 0.0],
            [-7.5, 8.5, -9.5, 10.5],
        ]
    )
    lower_model = _make_model(lower_only, torch.zeros(4))
    torch.testing.assert_close(lower_model(states), torch.zeros(3))

    upper = torch.tensor(
        [
            [0.0, 2.0, -3.0, 4.0],
            [0.0, 0.0, 5.0, -6.0],
            [0.0, 0.0, 0.0, 7.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    shared_bias = torch.tensor([0.5, -1.0, 1.5, -2.0])
    first_quadratic = (
        upper + torch.tril(torch.full((4, 4), 1.25), -1) + torch.eye(4) * 3.0
    )
    second_quadratic = (
        upper + torch.tril(torch.full((4, 4), -2.5), -1) - torch.eye(4) * 7.0
    )
    first = _make_model(first_quadratic, shared_bias)
    second = _make_model(second_quadratic, shared_bias)

    torch.testing.assert_close(first(states), second(states), rtol=0.0, atol=0.0)


def test_forward_backpropagates_exact_gradients_only_to_bias_and_upper_triangle() -> None:
    """Gradients hit the bias and the strict upper triangle only."""
    model = _make_dense_model()
    states = torch.tensor(
        [[1.0, 0.0, 1.0, 0.5], [0.0, 1.0, -0.5, 1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )

    model(states).sum().backward()

    torch.testing.assert_close(model.linear_bias.grad, -states.sum(dim=0))
    torch.testing.assert_close(
        model.quadratic_coef.grad, -(states.t() @ states).triu(1)
    )
    _assert_strict_upper_gradient(model.quadratic_coef.grad)


def test_objective_matches_closed_form_value_for_asymmetric_batches_and_identical_zero() -> None:
    """The objective equals the mean energy gap and vanishes on equal batches."""
    model = _make_dense_model()
    positive, negative = _contrastive_batches()

    loss = model.objective(positive, negative)

    assert loss.ndim == 0
    assert loss.dtype == torch.float64
    torch.testing.assert_close(loss, model(positive).mean() - model(negative).mean())
    torch.testing.assert_close(
        model.objective(positive, positive), torch.zeros((), dtype=torch.float64)
    )


def test_objective_backpropagates_exact_contrastive_divergence_gradients() -> None:
    """Contrastive divergence gradients equal the exact batch moment difference."""
    model = _make_dense_model()
    positive, negative = _contrastive_batches()

    model.objective(positive, negative).backward()

    torch.testing.assert_close(
        model.linear_bias.grad, negative.mean(dim=0) - positive.mean(dim=0)
    )
    expected_gradient = (
        (negative.t() @ negative) / negative.shape[0]
        - (positive.t() @ positive) / positive.shape[0]
    ).triu(1)
    torch.testing.assert_close(model.quadratic_coef.grad, expected_gradient)
    _assert_strict_upper_gradient(model.quadratic_coef.grad)


# Group D: full Ising matrix construction and state equivalence.


@pytest.mark.parametrize(
    ("torch_dtype", "numpy_dtype", "tolerance"),
    [(torch.float32, np.float32, 1e-6), (torch.float64, np.float64, 1e-12)],
)
def test_get_ising_matrix_matches_exact_entries_symmetry_zero_diag_and_dtype(
    torch_dtype: torch.dtype, numpy_dtype: np.dtype, tolerance: float
) -> None:
    """Entries, symmetry, zero diagonal and dtype all follow the exact formula."""
    model = _make_model(
        torch.tensor(
            [
                [0.0, 1.5, -2.0, 0.5],
                [-1.0, 0.0, 3.0, -3.5],
                [2.5, -0.5, 0.0, 4.0],
                [-4.5, 5.0, -5.5, 0.0],
            ],
            dtype=torch_dtype,
        ),
        torch.tensor([0.5, -1.5, 2.5, -3.5], dtype=torch_dtype),
    )
    ising_mat = model.get_ising_matrix()

    with torch.no_grad():
        symmetric = model.symmetrized_quadratic_coef()
    quadratic_block, bias_vector = _ising_components(symmetric, model.linear_bias)
    expected = _assemble_ising_matrix(quadratic_block, bias_vector)

    assert isinstance(ising_mat, np.ndarray)
    assert ising_mat.shape == (5, 5)
    assert ising_mat.dtype == numpy_dtype
    np.testing.assert_allclose(ising_mat, expected, rtol=0.0, atol=tolerance)
    np.testing.assert_array_equal(ising_mat, ising_mat.T)
    np.testing.assert_array_equal(
        np.diagonal(ising_mat), np.zeros(5, dtype=numpy_dtype)
    )


def test_get_ising_matrix_preserves_hamiltonian_across_all_binary_states() -> None:
    """Every binary state keeps the exact Hamiltonian offset under the map."""
    model = _make_dense_model()
    ising_mat = model.get_ising_matrix()
    binary_states, ising_spins = _all_binary_states(4, torch.float64)

    energies = model(binary_states)
    with torch.no_grad():
        symmetric = model.symmetrized_quadratic_coef()
        expected_offset = (
            0.5 * model.linear_bias.sum() + 0.125 * symmetric.sum()
        ).item()

    for index in range(binary_states.shape[0]):
        spin = ising_spins[index]
        residual = -(spin @ ising_mat @ spin) - energies[index].item()
        assert abs(residual - expected_offset) <= 1e-12


def test_get_ising_matrix_ignores_lower_triangle_diagonal_and_preserves_grad_state() -> None:
    """Lower triangle and diagonal are ignored and existing grads stay untouched."""
    upper = torch.tensor(
        [
            [0.0, 1.5, -2.5, 3.5],
            [0.0, 0.0, 4.5, -5.5],
            [0.0, 0.0, 0.0, 6.5],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    linear_bias = torch.tensor([0.75, -1.25, 2.5, -3.75])
    first_quadratic = (
        upper + torch.tril(torch.full((4, 4), 1.25), -1) + torch.eye(4) * 2.0
    )
    second_quadratic = (
        upper + torch.tril(torch.full((4, 4), -2.5), -1) - torch.eye(4) * 3.0
    )
    first = _make_model(first_quadratic, linear_bias)
    second = _make_model(second_quadratic, linear_bias)
    first.quadratic_coef.grad = torch.full((4, 4), 0.5)
    first.linear_bias.grad = torch.full((4,), -1.5)
    quadratic_before = first.quadratic_coef.detach().clone()
    bias_before = first.linear_bias.detach().clone()

    first_mat = first.get_ising_matrix()
    second_mat = second.get_ising_matrix()

    np.testing.assert_allclose(first_mat, second_mat, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(first.quadratic_coef, quadratic_before)
    torch.testing.assert_close(first.linear_bias, bias_before)
    torch.testing.assert_close(first.quadratic_coef.grad, torch.full((4, 4), 0.5))
    torch.testing.assert_close(first.linear_bias.grad, torch.full((4,), -1.5))


# Group E: conditional submatrix and conditional sampling.


def test_condition_sample_builds_exact_submatrices_for_batch_and_preserves_visible_prefix(
) -> None:
    """Each batch row gets its exact submatrix and the visible prefix is kept."""
    model = _make_dense_model()
    s_visible = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    solution = np.array([[1, -1, 1], [-1, 1, -1]], dtype=np.int64)
    sampler = _RecordingSampler(solutions=[solution])

    output = model.condition_sample(sampler, s_visible)

    assert sampler.call_count() == 3
    assert output.shape == (6, 4)

    with torch.no_grad():
        symmetric = model.symmetrized_quadratic_coef()
        sub_quadratic = symmetric[2:, 2:]
        sub_vh = symmetric[2:, :2]

    for index in range(s_visible.shape[0]):
        sub_linear = sub_vh @ s_visible[index] + model.linear_bias[2:]
        quadratic_block, bias_vector = _ising_components(sub_quadratic, sub_linear)
        expected = _assemble_ising_matrix(quadratic_block, bias_vector)
        captured = sampler.received_matrices[index]
        np.testing.assert_allclose(captured, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(np.diagonal(captured), np.zeros(3))
        np.testing.assert_allclose(captured, captured.T, rtol=0.0, atol=0.0)
        visible_block = output[index * 2 : (index + 1) * 2, :2]
        torch.testing.assert_close(
            visible_block, s_visible[index].unsqueeze(0).expand(2, 2)
        )


def test_condition_sample_submatrix_matches_conditional_hamiltonian_for_all_hidden_states(
) -> None:
    """The conditional submatrix reproduces the exact conditional energy offset."""
    model = _make_conditional_model()
    s_visible = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    sampler = _RecordingSampler()

    model.condition_sample(sampler, s_visible)

    captured = sampler.received_matrices[0]
    assert captured.shape == (4, 4)
    with torch.no_grad():
        symmetric = model.symmetrized_quadratic_coef()
        sub_quadratic = symmetric[2:, 2:]
        sub_vh = symmetric[2:, :2]
        sub_linear = sub_vh @ s_visible[0] + model.linear_bias[2:]
        expected_offset = (
            0.5 * sub_linear.sum() + 0.125 * sub_quadratic.sum()
        ).item()
        visible_energy = _closed_form_energy(
            s_visible, symmetric[:2, :2], model.linear_bias[:2]
        ).item()

    hidden_states, hidden_spins = _all_binary_states(3, torch.float64)
    for index in range(hidden_states.shape[0]):
        full_state = torch.cat([s_visible[0], hidden_states[index]]).unsqueeze(0)
        total_energy = model(full_state)[0].item()
        spin = hidden_spins[index]
        residual = -(spin @ captured @ spin) - total_energy
        assert abs(residual - (expected_offset - visible_energy)) <= 1e-12


def test_condition_sample_gauge_spin_sign_invariance_dtype_and_zero_hidden_boundary() -> None:
    """Gauge sign flips cancel, dtype is honoured and zero hidden nodes stay empty."""
    model = _make_model(
        torch.tensor([[0.0, 1.5, -2.0], [-0.5, 0.0, 3.0], [2.5, -3.5, 0.0]]),
        torch.tensor([0.5, -1.0, 2.0]),
    )
    s_visible = torch.tensor([[1.0]])
    positive = np.array([[1, -1, 1], [-1, 1, 1]], dtype=np.int64)
    negative = np.array([[-1, 1, -1], [1, -1, -1]], dtype=np.int64)

    positive_output = model.condition_sample(_RecordingSampler([positive]), s_visible)
    negative_output = model.condition_sample(_RecordingSampler([negative]), s_visible)

    torch.testing.assert_close(positive_output, negative_output)
    torch.testing.assert_close(
        positive_output[:, 1:], torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    )
    torch.testing.assert_close(positive_output[:, :1], s_visible.expand(2, 1))

    double_output = model.condition_sample(
        _RecordingSampler([positive]), s_visible, dtype=torch.float64
    )
    assert double_output.dtype == torch.float64
    torch.testing.assert_close(double_output, positive_output.double())

    full_visible = torch.tensor([[1.0, 0.0, 1.0]])
    boundary_sampler = _RecordingSampler([np.array([[1], [-1]], dtype=np.int64)])
    boundary_output = model.condition_sample(boundary_sampler, full_visible)

    np.testing.assert_array_equal(
        boundary_sampler.received_matrices[0], np.zeros((1, 1), dtype=np.float32)
    )
    assert boundary_output.shape == (2, 3)
    torch.testing.assert_close(boundary_output, full_visible.expand(2, 3))


def test_condition_sample_wraps_each_solve_in_kpp_caller_context_and_cleans_up_on_error(
    monkeypatch,
) -> None:
    """Every solve runs inside the caller context, which unwinds on failure."""
    model = _make_dense_model()
    s_visible = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    events: List[str] = []
    monkeypatch.setattr(fbm_module, "kpp_caller_context", _make_context_spy(events))
    sampler = _RecordingSampler(on_solve=partial(events.append, "solve"))

    model.condition_sample(sampler, s_visible)

    assert sampler.call_count() == 2
    assert events == ["enter", "solve", "exit", "enter", "solve", "exit"]

    failure_events: List[str] = []
    monkeypatch.setattr(
        fbm_module, "kpp_caller_context", _make_context_spy(failure_events)
    )
    failing = _RecordingSampler(
        error_message="cond solve failed",
        on_solve=partial(failure_events.append, "solve"),
    )

    with pytest.raises(RuntimeError, match="cond solve failed"):
        model.condition_sample(failing, s_visible)

    assert failure_events == ["enter", "solve", "exit"]


# Group F: Gibbs sampling determinism, math and boundaries.


def test_gibbs_sample_zero_steps_and_all_visible_boundary_preserve_visible_without_mutation(
) -> None:
    """Zero steps and all-visible input return the visible block unchanged."""
    model = _make_model(torch.zeros((4, 4)), torch.tensor([-50.0, 50.0, 0.0, 0.0]))
    s_visible = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    snapshot = s_visible.clone()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4321)
        frozen = model.gibbs_sample(num_steps=0, s_visible=s_visible)

    torch.testing.assert_close(frozen[:, :2], s_visible)
    torch.testing.assert_close(s_visible, snapshot)
    hidden = frozen[:, 2:]
    assert bool(torch.all((hidden == 0.0) | (hidden == 1.0)))

    all_visible = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        frozen_all = model.gibbs_sample(num_steps=5, s_visible=all_visible)

    torch.testing.assert_close(frozen_all, all_visible)


def test_gibbs_sample_seeded_reproducibility_and_lower_triangle_invariance() -> None:
    """Identical seeds reproduce bitwise results regardless of the lower triangle."""
    upper = torch.tensor(
        [
            [0.0, 0.5, -1.0, 1.5],
            [0.0, 0.0, 2.0, -2.5],
            [0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    linear_bias = torch.tensor([0.5, -1.0, 1.5, -2.0])
    first_quadratic = upper + torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    second_quadratic = (
        upper
        + torch.diag(torch.tensor([-4.0, -3.0, -2.0, -1.0]))
        + torch.tril(torch.full((4, 4), 0.75), -1)
    )
    first = _make_model(first_quadratic, linear_bias)
    second = _make_model(second_quadratic, linear_bias)
    s_visible = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(777)
        baseline = first.gibbs_sample(num_steps=5, num_sample=4)
        torch.manual_seed(777)
        repeated = first.gibbs_sample(num_steps=5, num_sample=4)
        torch.manual_seed(777)
        alternate = second.gibbs_sample(num_steps=5, num_sample=4)
        torch.manual_seed(555)
        baseline_conditional = first.gibbs_sample(num_steps=5, s_visible=s_visible)
        torch.manual_seed(555)
        alternate_conditional = second.gibbs_sample(num_steps=5, s_visible=s_visible)

    assert torch.equal(baseline, repeated)
    assert torch.equal(baseline, alternate)
    assert torch.equal(baseline_conditional, alternate_conditional)


def test_gibbs_sample_converges_deterministically_under_strong_bias_and_visible_coupling() -> None:
    """Strong bias and strong visible coupling drive exact one-step outcomes."""
    independent = _make_model(
        torch.zeros((4, 4)), torch.tensor([100.0, -100.0, 100.0, -100.0])
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31415)
        samples = independent.gibbs_sample(num_steps=1, num_sample=6)

    assert torch.equal(samples, torch.tensor([1.0, 0.0, 1.0, 0.0]).repeat(6, 1))

    coupled = _make_model(
        torch.tensor(
            [
                [0.0, 0.0, 100.0, 0.0],
                [0.0, 0.0, 0.0, -100.0],
                [100.0, 0.0, 0.0, 0.0],
                [0.0, -100.0, 0.0, 0.0],
            ]
        ),
        torch.tensor([0.0, 0.0, -50.0, 50.0]),
    )
    s_visible = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2718)
        coupled_samples = coupled.gibbs_sample(num_steps=1, s_visible=s_visible)

    expected = torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert torch.equal(coupled_samples, expected)


def test_gibbs_sample_visible_precedence_no_grad_and_missing_args_error() -> None:
    """Visible input wins over num_sample, stays detached and rejects empty calls."""
    model = _make_model(torch.zeros((4, 4)), torch.tensor([0.5, -1.0, 1.5, -2.0]))
    s_visible = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(606)
        samples = model.gibbs_sample(num_steps=2, s_visible=s_visible, num_sample=7)

    assert samples.shape == (2, 4)
    assert samples.requires_grad is False
    torch.testing.assert_close(samples[:, :2], s_visible)

    with pytest.raises(
        ValueError, match="Either s_visible or num_sample must be provided."
    ):
        model.gibbs_sample(num_steps=1)


# Group G: inherited unconditional sampling integration.


def test_sample_passes_exact_ising_matrix_maps_gauge_spins_and_uses_caller_context(
    monkeypatch,
) -> None:
    """Sampling forwards the Ising matrix, maps gauge spins and wraps the solver."""
    model = _make_model(
        torch.tensor([[0.0, 1.5, -2.0], [-0.5, 0.0, 3.0], [2.5, -3.5, 0.0]]),
        torch.tensor([0.5, -1.0, 2.0]),
    )
    events: List[str] = []
    monkeypatch.setattr(abm_module, "kpp_caller_context", _make_context_spy(events))
    sampler = _RecordingSampler(
        solutions=[np.array([[1, -1, 1, -1], [-1, 1, -1, 1]], dtype=np.int64)],
        on_solve=partial(events.append, "solve"),
    )

    samples = model.sample(sampler)

    assert sampler.call_count() == 1
    np.testing.assert_allclose(sampler.received_matrices[0], model.get_ising_matrix())
    assert samples.dtype == torch.float32
    assert samples.device == torch.device("cpu")
    torch.testing.assert_close(
        samples, torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    )
    assert events == ["enter", "solve", "exit"]
