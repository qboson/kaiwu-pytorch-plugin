"""Unit tests for the ``AbstractBoltzmannMachine`` base class contract.

The suite covers only the stable, pure-CPU contract of the base class:
construction and device bookkeeping, the docstring-only ``forward``
placeholder, ``get_ising_matrix`` -> ``_to_ising_matrix`` delegation,
``objective`` and the gauge-spin decoding performed by ``sample``.

Checkpoint serialization, parameter-level device re-synchronisation and any
accelerator transfer are deliberately out of scope so that this file stays
conflict-free with the concurrent checkpoint and cross-device work.
"""

from contextlib import AbstractContextManager, contextmanager
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import abstract_boltzmann_machine as abm_module
from kaiwu.torch_plugin.abstract_boltzmann_machine import AbstractBoltzmannMachine

DEFAULT_ISING_SIZE = 4


class _LinearEnergyBM(AbstractBoltzmannMachine):
    """Minimal pure-CPU concrete subclass with a deterministic linear energy."""

    def __init__(
        self,
        weights: torch.Tensor,
        ising_matrix: np.ndarray,
        device=None,
    ) -> None:
        """Store the trainable weights and the Ising matrix exposed to samplers."""
        super().__init__(device=device)
        self.weights = torch.nn.Parameter(weights.clone())
        self.ising_matrix = ising_matrix

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Compute the deterministic linear energy ``s_all @ weights``."""
        return s_all @ self.weights

    def _to_ising_matrix(self):
        """Return the Ising matrix configured at construction time."""
        return self.ising_matrix


class _RecordingSampler:
    """Deterministic mock sampler that records every Ising matrix it receives."""

    def __init__(
        self,
        solution: np.ndarray,
        error: Optional[Exception] = None,
        events: Optional[List[str]] = None,
    ) -> None:
        """Store the canned solution, the optional failure and the event sink."""
        self.solution = solution
        self.error = error
        self.events: List[str] = [] if events is None else events
        self.received_matrices: List[np.ndarray] = []

    def solve(self, ising_mat):
        """Record the request, then return the canned solution or raise."""
        self.events.append("solve")
        self.received_matrices.append(ising_mat)
        if self.error is not None:
            raise self.error
        return self.solution

    def call_count(self) -> int:
        """Return how many times ``solve`` has been invoked."""
        return len(self.received_matrices)


def _make_linear_bm(
    weights: List[float],
    ising_matrix: Optional[np.ndarray] = None,
) -> _LinearEnergyBM:
    """Build a CPU ``_LinearEnergyBM`` from plain floats and an optional matrix."""
    if ising_matrix is None:
        ising_matrix = np.zeros((DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE), dtype=np.float32)
    return _LinearEnergyBM(
        weights=torch.tensor(weights, dtype=torch.float32),
        ising_matrix=ising_matrix,
        device=torch.device("cpu"),
    )


def _make_context_spy(events: List[str]) -> Callable[[], AbstractContextManager[None]]:
    """Build a ``kpp_caller_context`` stand-in that records enter/exit markers."""

    @contextmanager
    def _spy() -> Iterator[None]:
        """Append ``enter`` and ``exit`` markers around the wrapped block."""
        events.append("enter")
        try:
            yield None
        finally:
            events.append("exit")

    return _spy


def test_init_is_nn_module_and_stores_explicit_cpu_device() -> None:
    """``__init__`` yields an ``nn.Module`` that stores the device verbatim."""
    model = AbstractBoltzmannMachine(device=torch.device("cpu"))
    string_model = AbstractBoltzmannMachine(device="cpu")

    assert isinstance(model, torch.nn.Module)
    assert model.device == torch.device("cpu")
    assert string_model.device == "cpu"


def test_init_default_device_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing device resolves to CPU when CUDA reports itself unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model = AbstractBoltzmannMachine()

    assert model.device == torch.device("cpu")
    assert model.device.type == "cpu"


def test_base_forward_returns_none_without_raising() -> None:
    """The base ``forward`` is a placeholder that returns ``None`` silently."""
    model = AbstractBoltzmannMachine(device=torch.device("cpu"))
    spins = torch.zeros((2, 3), dtype=torch.float32)

    assert model(spins) is None
    assert model.forward(spins) is None


def test_get_ising_matrix_raises_not_implemented_on_base_class() -> None:
    """The base class rejects Ising conversion with the documented message."""
    model = AbstractBoltzmannMachine(device=torch.device("cpu"))

    with pytest.raises(NotImplementedError, match="Subclasses must implement _ising method"):
        model.get_ising_matrix()


def test_get_ising_matrix_delegates_to_subclass_hook() -> None:
    """``get_ising_matrix`` forwards to the subclass ``_to_ising_matrix`` hook."""
    ising_matrix = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
    model = _make_linear_bm([1.0, -1.0], ising_matrix=ising_matrix)

    result = model.get_ising_matrix()

    assert result is ising_matrix
    np.testing.assert_array_equal(result, ising_matrix)


def test_objective_computes_mean_energy_difference_for_asymmetric_batches() -> None:
    """``objective`` returns the mean energy gap for ``b1 != b2`` batches."""
    weights = [1.5, -2.0, 0.5]
    model = _make_linear_bm(weights)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        s_positive = torch.randn((3, 3), dtype=torch.float32)
        s_negative = torch.randn((5, 3), dtype=torch.float32)
        loss = model.objective(s_positive, s_negative)

    expected = (s_positive @ weight_tensor).mean() - (s_negative @ weight_tensor).mean()

    assert loss.ndim == 0
    torch.testing.assert_close(loss, expected)


def test_objective_is_zero_when_positive_and_negative_batches_match() -> None:
    """Identical positive and negative batches collapse to an exact zero loss."""
    model = _make_linear_bm([1.0, -3.0])
    batch = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [1.0, -1.0]], dtype=torch.float32
    )

    loss = model.objective(batch, batch)

    torch.testing.assert_close(loss, torch.tensor(0.0, dtype=torch.float32))


def test_objective_dispatches_via_module_call_and_triggers_forward_hooks() -> None:
    """``objective`` goes through ``nn.Module.__call__``, so hooks fire per batch."""
    model = _make_linear_bm([0.5, -1.5])
    pre_shapes: List[Tuple[int, ...]] = []
    post_shapes: List[Tuple[int, ...]] = []

    def _record_pre_hook(module, inputs):
        """Record the dispatching module and the shape of its input batch."""
        assert module is model
        pre_shapes.append(tuple(inputs[0].shape))

    def _record_post_hook(module, inputs, output):
        """Record the dispatching module, the input batch and the output shape."""
        assert module is model
        assert inputs[0].shape[0] == output.shape[0]
        post_shapes.append(tuple(output.shape))

    pre_handle = model.register_forward_pre_hook(_record_pre_hook)
    post_handle = model.register_forward_hook(_record_post_hook)
    try:
        s_positive = torch.zeros((2, 2), dtype=torch.float32)
        s_negative = torch.ones((3, 2), dtype=torch.float32)
        model.objective(s_positive, s_negative)
    finally:
        pre_handle.remove()
        post_handle.remove()

    assert pre_shapes == [(2, 2), (3, 2)]
    assert post_shapes == [(2,), (3,)]


def test_objective_backpropagates_exact_parameter_gradients() -> None:
    """Backpropagation through ``objective`` yields exact mean-spin gradients."""
    model = _make_linear_bm([0.5, -1.0, 2.0])
    s_positive = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    s_negative = torch.tensor(
        [[-1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    loss = model.objective(s_positive, s_negative)
    loss.backward()

    assert model.weights.grad is not None
    torch.testing.assert_close(
        model.weights.grad,
        s_positive.mean(dim=0) - s_negative.mean(dim=0),
    )


def test_objective_raises_attribute_error_when_forward_is_unimplemented() -> None:
    """The placeholder ``forward`` makes ``objective`` fail with ``AttributeError``."""
    model = AbstractBoltzmannMachine(device=torch.device("cpu"))
    s_positive = torch.zeros((2, 2), dtype=torch.float32)
    s_negative = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(AttributeError):
        model.objective(s_positive, s_negative)


def test_sample_passes_ising_matrix_and_maps_positive_gauge_spins() -> None:
    """``sample`` forwards the Ising matrix and decodes a positive gauge spin."""
    ising_matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    model = _make_linear_bm([1.0, -1.0, 0.5, 0.5], ising_matrix=ising_matrix)
    sampler = _RecordingSampler(
        np.array([[1, -1, 1, 1], [-1, -1, 1, 1]], dtype=np.float32)
    )

    samples = model.sample(sampler)

    assert sampler.call_count() == 1
    assert sampler.received_matrices[0] is ising_matrix
    assert samples.shape == (2, 3)
    assert samples.dtype == torch.float32
    assert samples.device == torch.device("cpu")
    torch.testing.assert_close(samples, torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))


def test_sample_flips_spins_for_negative_gauge_and_is_sign_flip_invariant() -> None:
    """A negative gauge spin flips the decoded spins; a global sign flip cancels."""
    model = _make_linear_bm([1.0, 0.0, -1.0, 1.0])
    raw = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]], dtype=np.float32)
    expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

    samples_positive = model.sample(_RecordingSampler(raw))
    samples_negative = model.sample(_RecordingSampler(-raw))

    torch.testing.assert_close(samples_positive, expected)
    torch.testing.assert_close(samples_negative, samples_positive)


@pytest.mark.parametrize("np_dtype", [np.int64, np.float64])
def test_sample_converts_int64_and_float64_single_row_solutions_to_float32_cpu(
    np_dtype,
) -> None:
    """A single-row solution of any numeric dtype becomes ``float32`` on CPU."""
    model = _make_linear_bm([1.0, -1.0, 0.5])
    sampler = _RecordingSampler(np.array([[-1, 1, -1]], dtype=np_dtype))

    samples = model.sample(sampler)

    assert samples.shape == (1, 2)
    assert samples.dtype == torch.float32
    assert samples.device == torch.device("cpu")
    torch.testing.assert_close(samples, torch.tensor([[1.0, 0.0]], dtype=torch.float32))


def test_sample_wraps_solver_execution_in_kpp_caller_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sample`` enters the KPP caller context around the solver call."""
    events: List[str] = []
    model = _make_linear_bm([1.0, -1.0, 0.5, 0.5])
    sampler = _RecordingSampler(np.ones((1, 4), dtype=np.float32), events=events)
    monkeypatch.setattr(abm_module, "kpp_caller_context", _make_context_spy(events))

    model.sample(sampler)

    assert events == ["enter", "solve", "exit"]


def test_sample_exits_kpp_caller_context_when_solver_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The KPP caller context is exited even when the solver raises."""
    events: List[str] = []
    model = _make_linear_bm([1.0, -1.0, 0.5, 0.5])
    sampler = _RecordingSampler(
        np.ones((1, 4), dtype=np.float32),
        error=RuntimeError("solver failed"),
        events=events,
    )
    monkeypatch.setattr(abm_module, "kpp_caller_context", _make_context_spy(events))

    with pytest.raises(RuntimeError, match="solver failed"):
        model.sample(sampler)

    assert events == ["enter", "solve", "exit"]


def test_sample_raises_not_implemented_before_calling_sampler_on_base_class() -> None:
    """The base class fails on the Ising hook before touching the sampler."""
    model = AbstractBoltzmannMachine(device=torch.device("cpu"))
    sampler = _RecordingSampler(np.ones((1, 3), dtype=np.float32))

    with pytest.raises(NotImplementedError):
        model.sample(sampler)

    assert sampler.call_count() == 0


def test_sample_rejects_one_dimensional_solver_output() -> None:
    """A one-dimensional solver output is rejected by the gauge decoding."""
    model = _make_linear_bm([1.0, -1.0, 0.5])
    sampler = _RecordingSampler(np.array([1, -1, 1], dtype=np.float32))

    with pytest.raises(IndexError):
        model.sample(sampler)
