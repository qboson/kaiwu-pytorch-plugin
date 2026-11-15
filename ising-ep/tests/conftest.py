"""Shared pytest helpers.

Import paths come from the ``pythonpath`` setting in ``pyproject.toml``, so this
module does not manipulate ``sys.path``.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterator

import numpy as np
import torch

from ising_ep.kaiwu_solver import AnnealSettings

#: Annealing schedule of the archived primary run in ``results/primary``, and the
#: default the CLI exposes. Shared so both the solver and the CLI tests assert
#: against one definition.
PRIMARY_ANNEAL = AnnealSettings(
    initial_temperature=1.25,
    alpha=0.95,
    cutoff_temperature=0.001,
    iterations_per_t=10,
    candidate_count=8,
)


class FakeSolver:
    """Stands in for :class:`ising_ep.kaiwu_solver.KaiwuSolver` in unit tests.

    Returns scripted physical states instead of annealing, and records every
    call so pairing rules such as the shared seed and the nudged warm start can
    be asserted exactly.

    Attributes:
        scripted: Physical spin vectors returned in round-robin order.
        calls: One record per :meth:`solve` call, holding ``Q``, ``seed`` and
            ``init_solution``.
    """

    def __init__(self, *physical_per_call: Any) -> None:
        """Create a solver that cycles through the given physical states.

        Args:
            *physical_per_call: One physical spin vector per expected call.
        """
        self.scripted = [np.asarray(state, dtype=np.float64) for state in physical_per_call]
        self.calls: list[dict[str, Any]] = []

    def solve(
        self,
        Q: Any,
        *,
        seed: int,
        init_solution: np.ndarray | None = None,
    ) -> np.ndarray:
        """Record the call and return the next scripted candidate block.

        Args:
            Q: QUBO matrix the caller built.
            seed: Anneal seed the caller passed.
            init_solution: Warm start the caller passed, if any.

        Returns:
            A ``(1, n + 1)`` candidate block. The auxiliary spin is ``+1``, so
            the leading entries are already the physical spins.
        """
        state = self.scripted[len(self.calls) % len(self.scripted)]
        self.calls.append({"Q": Q, "seed": seed, "init_solution": init_solution})
        return np.concatenate([state, [1.0]]).reshape(1, -1)


def iter_physical_states(n: int) -> Iterator[torch.Tensor]:
    """Yield every +-1 assignment of length ``n`` in ``(-1, +1)`` order.

    Exhaustive enumeration is only tractable for the tiny models used in tests,
    which is why this helper lives here rather than in the package.

    Args:
        n: Number of spins per assignment.

    Yields:
        One ``float64`` spin vector of length ``n`` per assignment.
    """
    for bits in itertools.product((-1.0, 1.0), repeat=n):
        yield torch.tensor(bits, dtype=torch.float64)


def qubo_score(matrix: Any, constant: float, spins: Any) -> float:
    """Score physical spins under an upper-triangular QUBO.

    This stays in the tests. The package exposes the energy through
    :meth:`ising_ep.model.IsingEPModel.energy` and the matrix through
    :meth:`ising_ep.model.IsingEPModel.qubo`.

    Args:
        matrix: Upper-triangular QUBO matrix.
        constant: Scalar offset returned with the matrix.
        spins: Physical spins in ``{-1, +1}``.

    Returns:
        ``q @ Q @ q + constant`` with ``q = (spins + 1) / 2``.
    """
    bits = (to_numpy(spins).reshape(-1) + 1.0) * 0.5
    values = to_numpy(matrix)
    return float(bits @ values @ bits + float(constant))


def to_numpy(value: Any) -> np.ndarray:
    """Convert a NumPy array or Torch tensor to a detached NumPy array.

    Args:
        value: Array or tensor to convert.

    Returns:
        A ``float64`` NumPy array.
    """
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def set_parameter(parameter: Any, value: Any) -> None:
    """Overwrite a parameter in place, keeping its tensor identity.

    Tests use this to pin a parameter to a known value without replacing the
    object the optimizer holds a reference to.

    Args:
        parameter: Torch parameter or NumPy array to overwrite.
        value: Replacement values, broadcastable to ``parameter``.
    """
    array = np.asarray(value, dtype=np.float64)
    if hasattr(parameter, "data") and hasattr(parameter, "copy_"):
        with torch.no_grad():
            parameter.copy_(torch.as_tensor(array, dtype=parameter.dtype))
        return
    parameter[...] = array
