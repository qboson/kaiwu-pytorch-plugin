"""QUBO in, ranked spin candidates out. The only module that imports ``kaiwu``.

One solve calls ``qubo_matrix_to_qubo_model``, ``qubo_model_to_ising_model``,
``SimulatedAnnealingOptimizer.single_process_solve``, and
``get_sorted_solutions``. Kaiwu 1.4 exposes the three helpers on ``kaiwu.core``.
Kaiwu 1.3.1 keeps the converters on ``kaiwu.qubo`` and ``kaiwu.conversion`` and
has no sorter, so that case uses :func:`_sorted_solutions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Name Kaiwu gives the gauge spin appended by ``qubo_model_to_ising_model``.
AUXILIARY_VARIABLE = "__spin__"


@dataclass(frozen=True)
class AnnealSettings:
    """Simulated-annealing schedule shared by paired free and nudged solves.

    Paired equilibrium-propagation phases must anneal identically, so one
    instance is reused for both phases of a sample; only the seed and the warm
    start differ between them.

    Attributes:
        initial_temperature: Starting temperature of the anneal.
        alpha: Geometric cooling factor applied after each temperature step.
        cutoff_temperature: Temperature at which annealing stops.
        iterations_per_t: Sweeps performed at each temperature.
        candidate_count: Number of candidate solutions to return per solve,
            passed to Kaiwu as ``size_limit``.
    """

    initial_temperature: float = 1.25
    alpha: float = 0.95
    cutoff_temperature: float = 0.001
    iterations_per_t: int = 10
    candidate_count: int = 8


def _import_kaiwu() -> Any:
    """Import the licensed Kaiwu package.

    The import is deferred so unit tests can inject a fake module and so the
    rest of the package remains importable without the enterprise wheel.

    Returns:
        The imported ``kaiwu`` module.
    """
    import kaiwu as kw

    return kw


def init_license(user_id: str, sdk_code: str) -> None:
    """Activate the Kaiwu license for this process.

    Args:
        user_id: Value of ``KAIWU_USER_ID`` from the quantum cloud platform.
        sdk_code: Value of ``KAIWU_SDK_CODE`` from the quantum cloud platform.
    """
    _import_kaiwu().license.init(user_id=user_id, sdk_code=sdk_code)


def _sorted_solutions(
    matrix: np.ndarray,
    solutions: np.ndarray,
    bias: float = 0.0,
    negtail_ff: bool = True,
    sort_solutions: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort spin rows the way Kaiwu 1.4's ``get_sorted_solutions`` does.

    Kaiwu 1.3.1 does not ship that helper. The energy is ``-s^T J s + bias``,
    lowest first, and a negative auxiliary spin flips the whole row.

    Args:
        matrix: Ising coupling matrix, including the auxiliary spin.
        solutions: Candidate rows. A single flat row is accepted.
        bias: Constant added to every energy.
        negtail_ff: Flip every row whose auxiliary spin is ``-1``.
        sort_solutions: Sort by energy. ``False`` keeps the input order.

    Returns:
        The rows and their energies.
    """
    rows = np.atleast_2d(np.asarray(solutions, dtype=np.float64))
    coupling = np.asarray(matrix, dtype=np.float64)
    hamilton = -np.einsum("ij,ij->i", rows.dot(coupling), rows) + bias
    if sort_solutions:
        order = np.argsort(hamilton)
        rows = rows[order]
        hamilton = hamilton[order]
    if negtail_ff:
        rows = rows * rows[:, [-1]]
    return rows, hamilton


def _sdk_symbol(owner: Any, name: str) -> Any:
    """Return ``owner.name`` when the installed SDK defines it."""
    if owner is None:
        return None
    return getattr(owner, name, None)


def _sdk_api(kw: Any) -> tuple[Any, Any, Any]:
    """Resolve the QUBO converters and the solution sorter for this SDK.

    Args:
        kw: Imported ``kaiwu`` module, real or injected by a test.

    Returns:
        ``(qubo_matrix_to_qubo_model, qubo_model_to_ising_model, sorter)``.
        The sorter is ``kaiwu.core.get_sorted_solutions`` on 1.4 and
        :func:`_sorted_solutions` on 1.3.1.
    """
    core = getattr(kw, "core", None)
    to_model = _sdk_symbol(core, "qubo_matrix_to_qubo_model")
    to_ising = _sdk_symbol(core, "qubo_model_to_ising_model")
    sorter = _sdk_symbol(core, "get_sorted_solutions")
    return (
        to_model or kw.qubo.qubo_matrix_to_qubo_model,
        to_ising or kw.conversion.qubo_model_to_ising_model,
        sorter or _sorted_solutions,
    )


def _as_float64_matrix(value: Any) -> np.ndarray:
    """Convert a CPU Torch tensor or array-like into a contiguous float64 matrix.

    Args:
        value: QUBO matrix as a Torch tensor or any array-like.

    Returns:
        A C-contiguous ``float64`` NumPy array suitable for the Kaiwu API.
    """
    return np.ascontiguousarray(value, dtype=np.float64)


class KaiwuSolver:
    """Solves QUBO problems on the official Kaiwu classical annealer.

    One instance holds the annealing schedule, which is what makes paired
    free/nudged solves comparable: callers vary only the seed and the warm
    start.

    Attributes:
        settings: Annealing schedule applied to every :meth:`solve` call.
    """

    def __init__(self, settings: AnnealSettings | None = None) -> None:
        """Create a solver bound to one annealing schedule.

        Args:
            settings: Annealing schedule. Defaults to :class:`AnnealSettings`.
        """
        self.settings = settings or AnnealSettings()

    def solve(
        self,
        Q: Any,
        *,
        seed: int,
        init_solution: np.ndarray | None = None,
    ) -> np.ndarray:
        """Anneal one QUBO matrix and return candidate spin vectors.

        Args:
            Q: Upper-triangular QUBO matrix of shape ``(n, n)``, as a Torch
                tensor or array-like.
            seed: Random seed for the anneal. Paired free/nudged solves of the
                same sample pass the same value so the two phases stay
                comparable.
            init_solution: Optional warm start in solver order, length
                ``n + 1``. The nudged phase passes the free phase's selected
                candidate row here.

        Returns:
            A ``float64`` array of shape ``(candidate_count, n + 1)``, sorted
            so row 0 has the lowest Ising objective. Entries are ``-1.0`` or
            ``+1.0``, and the trailing auxiliary spin is fixed at ``+1.0`` by
            Kaiwu's negative-tail flip. The leading ``n`` entries of a row are
            the physical spins; the whole row is a valid ``init_solution``.

        Raises:
            RuntimeError: If the Ising model Kaiwu returns does not place the
                auxiliary spin last, which would invalidate the negative-tail
                flip.
        """
        kw = _import_kaiwu()
        matrix = _as_float64_matrix(Q)
        to_model, to_ising, sort = _sdk_api(kw)

        qubo_model = to_model(matrix)
        ising_model = to_ising(qubo_model)
        self._require_trailing_auxiliary(ising_model.get_variables(), matrix.shape[0])
        ising_matrix = _as_float64_matrix(ising_model.get_matrix())

        rows = self._anneal(kw, ising_matrix, seed, init_solution)
        ranked, _energies = sort(
            ising_matrix, rows, negtail_ff=True, sort_solutions=True
        )
        return np.asarray(ranked, dtype=np.float64)

    def _anneal(
        self,
        kw: Any,
        ising_matrix: np.ndarray,
        seed: int,
        init_solution: np.ndarray | None,
    ) -> np.ndarray:
        """Anneal once, retrying when Kaiwu returns an empty pool.

        A warm start sometimes comes back with no candidates. The extra
        attempts flip that start and then drop it. They repeat the same solve;
        they are not a log of failed runs.

        Args:
            kw: Imported ``kaiwu`` module.
            ising_matrix: Ising coupling matrix, including the auxiliary spin.
            seed: Random seed forwarded to the optimizer and the solve call.
            init_solution: Warm start in solver order, or ``None`` to start cold.

        Returns:
            Candidate rows whose width matches ``ising_matrix``.

        Raises:
            RuntimeError: If a cold start still returns no candidates.
        """
        width = int(ising_matrix.shape[0])
        attempts: list[np.ndarray | None] = [init_solution]
        if init_solution is not None:
            attempts.append(-np.asarray(init_solution, dtype=np.float64))
            attempts.append(None)
        for attempt in attempts:
            optimizer = kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=self.settings.initial_temperature,
                alpha=self.settings.alpha,
                cutoff_temperature=self.settings.cutoff_temperature,
                iterations_per_t=self.settings.iterations_per_t,
                size_limit=self.settings.candidate_count,
                rand_seed=int(seed),
                # Paired EP solves must stay reproducible, so never fan out.
                process_num=1,
            )
            raw = optimizer.single_process_solve(
                ising_matrix=ising_matrix,
                init_solution=attempt,
                rand_seed=int(seed),
            )
            rows = np.atleast_2d(np.asarray(raw, dtype=np.float64))
            if rows.ndim == 2 and rows.shape[-1] == width and rows.shape[0] > 0:
                return rows
        raise RuntimeError("Kaiwu annealer returned no candidates")

    @staticmethod
    def _require_trailing_auxiliary(variables: dict, n_qubo: int) -> None:
        """Verify the auxiliary spin occupies the final solver position.

        Args:
            variables: Name-to-index map from ``ising_model.get_variables()``.
            n_qubo: Number of QUBO variables before the Ising rewrite.

        Raises:
            RuntimeError: If the map is not ``{b[i]: i, __spin__: n_qubo}``.
        """
        if len(variables) != n_qubo + 1 or variables.get(AUXILIARY_VARIABLE) != n_qubo:
            raise RuntimeError(
                "Kaiwu Ising model must expose exactly one trailing auxiliary "
                f"spin {AUXILIARY_VARIABLE!r} at index {n_qubo}; got {variables!r}"
            )
