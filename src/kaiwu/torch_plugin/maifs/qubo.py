from __future__ import annotations

# pylint: disable=too-many-lines,import-outside-toplevel

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from time import time_ns
from typing import Any, cast

import numpy as np

AVAILABLE_SOLVERS = (
    "local_search",
    "sa",
    "kaiwu_cim",
)

DEFAULT_CIM_TARGET_PRECISION = 14
DEFAULT_CIM_MAX_BITS = 1000
DEFAULT_CIM_MAX_PRECISION = 32
DEFAULT_CIM_PRECISION_STEP = 4
DEFAULT_CIM_SAMPLE_NUMBER = 512


@dataclass
class PrecisionSplitPlan:
    """Store a Kaiwu CIM precision-adjustment and variable-splitting plan.

    The plan records one precision search result. It does not execute the CIM
    solver by itself.

    Args:
        source_precision: Source precision used to adjust the original Ising
            matrix.
        target_precision: Target precision for the split matrix submitted to
            Kaiwu CIM.
        max_bits: Maximum allowed variable count after splitting.
        adjusted_matrix: Ising matrix after precision adjustment.
        split_matrix: Ising matrix after variable splitting.
        last_var_idx: Mapping from split variables back to original variables.
        split_size: Variable count after splitting.
        precision_info: Metadata returned by Kaiwu precision calculation.
        history: Precision search attempt history.
    """

    source_precision: int
    target_precision: int
    max_bits: int
    adjusted_matrix: np.ndarray
    split_matrix: np.ndarray
    last_var_idx: np.ndarray
    split_size: int
    precision_info: dict[str, Any]
    history: list[dict[str, Any]]


class PrecisionSplitExplorer:
    """Find a feasible Kaiwu precision/split plan under a bit-size limit.

    This is a local copy of the helper used in ``kaiwu_test``. It delays Kaiwu
    imports until use, so the default non-CIM QUBO backends do not require Kaiwu.

    Args:
        target_precision: Target precision after variable splitting.
        max_bits: Maximum allowed variable count after splitting.
        max_precision: Maximum source precision to test during search.
        min_precision: Backward-compatible alias for the starting precision.
        min_increment: Optional minimum increment passed to Kaiwu splitting.
        penalty: Optional splitting penalty coefficient.
        round_to_increment: Whether Kaiwu should round values to the increment.
        start_precision: Starting source precision for the search.
        precision_step: Coarse-search precision step.

    Raises:
        ValueError: If precision bounds, precision step, or bit limit are invalid.
    """

    def __init__(
        self,
        target_precision: int = 8,
        max_bits: int | None = None,
        max_precision: int = 32,
        min_precision: int | None = None,
        min_increment: float | None = None,
        penalty: float | None = None,
        round_to_increment: bool = True,
        start_precision: int | None = None,
        precision_step: int = 4,
    ) -> None:
        """Initialize a Kaiwu precision-split explorer.

        Args:
            target_precision: Target precision after variable splitting.
            max_bits: Maximum allowed variable count after splitting.
            max_precision: Maximum source precision to test during search.
            min_precision: Backward-compatible alias for the starting precision.
            min_increment: Optional minimum increment passed to Kaiwu splitting.
            penalty: Optional splitting penalty coefficient.
            round_to_increment: Whether Kaiwu should round values to the
                increment.
            start_precision: Starting source precision for the search.
            precision_step: Coarse-search precision step.

        Raises:
            ValueError: If precision or bit-count settings are invalid.
            TypeError: If ``precision_step`` is not an integer.
        """
        if max_bits is None:
            raise ValueError("max_bits is required")
        if target_precision < 2:
            raise ValueError("target_precision must be at least 2")
        if start_precision is None:
            start_precision = min_precision
        if start_precision is None:
            start_precision = target_precision
        if start_precision < 2:
            raise ValueError("start_precision must be at least 2")
        if max_precision < start_precision:
            raise ValueError("max_precision must be no smaller than start_precision")
        if not isinstance(precision_step, Integral):
            raise TypeError("precision_step must be an integer")
        if precision_step < 1:
            raise ValueError("precision_step must be at least 1")

        self.target_precision = int(target_precision)
        self.max_bits = int(max_bits)
        self.max_precision = int(max_precision)
        self.start_precision = int(start_precision)
        self.min_precision = self.start_precision
        self.precision_step = int(precision_step)
        self.min_increment = min_increment
        self.penalty = penalty
        self.round_to_increment = round_to_increment

        self.plan: PrecisionSplitPlan | None = None
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _resolve_min_increment(
        ising_matrix: np.ndarray,
        min_increment: float | None,
    ) -> float:
        """Resolve the minimum increment used during precision splitting.

        Args:
            ising_matrix: Input Ising matrix.
            min_increment: User-provided minimum increment, if any.

        Returns:
            Minimum increment passed to Kaiwu splitting.
        """
        if min_increment is not None:
            return min_increment
        params = np.unique(ising_matrix)
        if params.shape[0] < 2:
            return 1
        diff = np.diff(params)
        positive_diff = diff[diff > 0]
        if positive_diff.shape[0] == 0:
            return 1
        return float(positive_diff.min())

    @staticmethod
    def _adjust_to_precision(
        ising_matrix: np.ndarray,
        precision: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Adjust an Ising matrix to a target integer precision.

        Args:
            ising_matrix: Original Ising matrix.
            precision: Target integer precision.

        Returns:
            A tuple containing the adjusted matrix and Kaiwu precision metadata.

        Raises:
            ImportError: If the Kaiwu precision helpers are unavailable.
        """
        try:
            from kaiwu.preprocess import (
                adjust_ising_matrix_precision,
                calculate_ising_matrix_bit_width,
            )
        except ImportError:
            from kaiwu.ising import (
                adjust_ising_matrix_precision,
                calculate_ising_matrix_bit_width,
            )

        precision_info = calculate_ising_matrix_bit_width(ising_matrix, precision)
        if precision_info.get("multiplier") != float("inf"):
            adjusted_matrix = ising_matrix * precision_info.get("multiplier")
        else:
            adjusted_matrix = adjust_ising_matrix_precision(ising_matrix, precision)
        adjusted_matrix = np.round(adjusted_matrix).astype(int)
        return adjusted_matrix, precision_info

    def _build_plan(
        self,
        ising_matrix: np.ndarray,
        source_precision: int,
    ) -> PrecisionSplitPlan:
        """Build one precision-adjustment and variable-splitting plan.

        Args:
            ising_matrix: Original Ising matrix.
            source_precision: Source precision for this attempt.

        Returns:
            The split matrix, variable mapping, and related search metadata.

        Raises:
            ImportError: If the Kaiwu splitting helper is unavailable.
        """
        from kaiwu.preprocess import perform_precision_adaption_split

        adjusted_matrix, precision_info = self._adjust_to_precision(
            ising_matrix,
            source_precision,
        )
        min_increment = self._resolve_min_increment(
            adjusted_matrix,
            self.min_increment,
        )
        split_matrix, last_var_idx = perform_precision_adaption_split(
            adjusted_matrix,
            param_bit=self.target_precision,
            min_increment=min_increment,
            penalty=self.penalty,
            round_to_increment=self.round_to_increment,
        )
        return PrecisionSplitPlan(
            source_precision=int(source_precision),
            target_precision=self.target_precision,
            max_bits=self.max_bits,
            adjusted_matrix=adjusted_matrix,
            split_matrix=split_matrix,
            last_var_idx=last_var_idx,
            split_size=int(split_matrix.shape[0]),
            precision_info=precision_info,
            history=list(self.history),
        )

    def _evaluate(
        self,
        ising_matrix: np.ndarray,
        source_precision: int,
        phase: str,
    ) -> PrecisionSplitPlan:
        """Build a split plan and record the precision-search attempt.

        Args:
            ising_matrix: Original Ising matrix.
            source_precision: Source precision for this attempt.
            phase: Search phase name, such as ``"coarse"`` or ``"fine"``.

        Returns:
            The precision-split plan produced by this attempt.

        Raises:
            ValueError: If the Ising matrix or precision arguments are invalid.
            RuntimeError: If a Kaiwu preprocessing helper fails.
        """
        plan = self._build_plan(ising_matrix, source_precision)
        attempt = {
            "source_precision": int(source_precision),
            "split_size": int(plan.split_size),
            "phase": phase,
            "precision_info": plan.precision_info,
        }
        self.history.append(attempt)
        plan.history = list(self.history)
        return plan

    def search(
        self,
        ising_matrix: np.ndarray,
    ) -> PrecisionSplitPlan:
        """Search for the highest feasible split precision under ``max_bits``.

        Args:
            ising_matrix: Original Ising matrix.

        Returns:
            The best feasible precision-split plan found by the search.

        Raises:
            ValueError: If the original matrix is not square or already exceeds
                ``max_bits``.
            RuntimeError: If no feasible precision is found.
        """
        ising_matrix = np.asarray(ising_matrix)
        if ising_matrix.ndim != 2 or ising_matrix.shape[0] != ising_matrix.shape[1]:
            raise ValueError("ising_matrix must be a square matrix")
        if ising_matrix.shape[0] > self.max_bits:
            raise ValueError("The original matrix size cannot be larger than max_bits")

        self.history = []
        best_plan = None
        previous_coarse_precision = None
        source_precision = self.start_precision

        while True:
            plan = self._evaluate(
                ising_matrix,
                source_precision,
                "coarse",
            )
            if plan.split_size <= self.max_bits:
                best_plan = plan
                previous_coarse_precision = source_precision
                if source_precision == self.max_precision:
                    self.plan = best_plan
                    return best_plan
                source_precision = min(
                    source_precision + self.precision_step,
                    self.max_precision,
                )
                continue

            if previous_coarse_precision is not None:
                for fine_precision in range(
                    previous_coarse_precision + 1,
                    source_precision,
                ):
                    fine_plan = self._evaluate(
                        ising_matrix,
                        fine_precision,
                        "fine",
                    )
                    if fine_plan.split_size <= self.max_bits:
                        best_plan = fine_plan
                    else:
                        break

            if best_plan is not None:
                best_plan.history = list(self.history)
                self.plan = best_plan
                return best_plan
            break

        msg = (
            "No feasible precision found: split matrix size exceeds max_bits "
            f"for source_precision in [{self.start_precision}, {self.max_precision}]"
        )
        if self.history:
            msg += (
                "; first attempted precision produced size "
                f"{self.history[0]['split_size']}"
            )
        raise RuntimeError(msg)

    def restore_solution(self, solution: np.ndarray, vote: bool = False) -> np.ndarray:
        """Restore a split-problem solution to the original variable space.

        Args:
            solution: Solution vector for the split problem.
            vote: Whether to use Kaiwu voting restoration when supported.

        Returns:
            Restored solution over the original variables.

        Raises:
            ValueError: If ``search`` has not been called.
            ImportError: If the Kaiwu restoration helper is unavailable.
        """
        try:
            from kaiwu.preprocess import restore_splitted_solution

            if self.plan is None:
                raise ValueError("search or fit must be called before restoring")
            return restore_splitted_solution(solution, self.plan.last_var_idx, vote)
        except ImportError:
            from kaiwu.preprocess import restore_split_solution

        if self.plan is None:
            raise ValueError("search or fit must be called before restoring")
        return restore_split_solution(solution, self.plan.last_var_idx)


class QuadraticLinearSolver:  # pylint: disable=too-few-public-methods
    """Convert QUBO quadratic and linear terms to an Ising matrix."""

    @staticmethod
    def _qubo_matrix_to_ising_matrix(qubo_matrix: np.ndarray) -> np.ndarray:
        """Convert an upper-triangular QUBO matrix to an auxiliary-spin Ising matrix.

        Args:
            qubo_matrix: Upper-triangular QUBO matrix.

        Returns:
            Ising matrix with one auxiliary spin appended.

        Raises:
            ValueError: If ``qubo_matrix`` is not a finite square matrix.
        """
        qubo_matrix = np.asarray(qubo_matrix, dtype=float)
        if qubo_matrix.ndim != 2 or qubo_matrix.shape[0] != qubo_matrix.shape[1]:
            raise ValueError("qubo_matrix must be a square matrix")
        if not np.all(np.isfinite(qubo_matrix)):
            raise ValueError("qubo_matrix must contain only finite values")

        num_variables = qubo_matrix.shape[0]
        auxiliary_index = num_variables
        ising_matrix = np.zeros((num_variables + 1, num_variables + 1), dtype=float)

        for row in range(num_variables):
            diagonal = float(qubo_matrix[row, row])
            ising_matrix[row, auxiliary_index] += 0.5 * diagonal
            for col in range(row + 1, num_variables):
                coefficient = float(qubo_matrix[row, col])
                pair_weight = 0.25 * coefficient
                ising_matrix[row, col] += pair_weight
                ising_matrix[row, auxiliary_index] += pair_weight
                ising_matrix[col, auxiliary_index] += pair_weight

        return ising_matrix

    def solve(
        self,
        quadratic_matrix: np.ndarray,
        linear_vector: np.ndarray,
    ) -> np.ndarray:
        """Convert QUBO terms to the equivalent Ising matrix.

        Args:
            quadratic_matrix: QUBO quadratic term.
            linear_vector: QUBO linear term.

        Returns:
            Ising matrix with one auxiliary spin appended.
        """
        qubo_matrix = _qubo_terms_to_matrix(quadratic_matrix, linear_vector)
        return self._qubo_matrix_to_ising_matrix(qubo_matrix)


def _solve_ising_local_search(
    ising_matrix: np.ndarray,
    initial_binary: np.ndarray | None = None,
    max_iter: int = 2000,
) -> np.ndarray:
    """Solve an Ising matrix with local greedy spin flips.

    Args:
        ising_matrix: Square Ising matrix with an auxiliary spin.
        initial_binary: Optional initial binary state for the original QUBO
            variables.
        max_iter: Maximum number of greedy-improvement iterations.

    Returns:
        One or more spin solutions encoded as ``-1`` and ``1`` values.

    Raises:
        ValueError: If input shapes or values are invalid.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    matrix = np.asarray(ising_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("ising_matrix must be a square matrix")

    if initial_binary is None:
        spins = np.ones(matrix.shape[0], dtype=int)
    else:
        binary = np.asarray(initial_binary, dtype=int)
        if binary.ndim != 1 or binary.shape[0] != matrix.shape[0] - 1:
            raise ValueError("initial_binary must match the QUBO variable size")
        if not np.all((binary == 0) | (binary == 1)):
            raise ValueError("initial_binary must contain only 0/1 values")
        spins = np.r_[2 * binary - 1, 1].astype(int)

    weights = np.triu(matrix)

    def objective(candidate: np.ndarray) -> float:
        """Evaluate the Ising objective for a candidate spin vector.

        Args:
            candidate: Candidate spin vector encoded with ``-1`` and ``1``.

        Returns:
            Ising objective value for ``candidate``.
        """
        return float(np.sum(weights * np.outer(candidate, candidate)))

    current_value = objective(spins)

    for _ in range(int(max_iter)):
        best_delta = 0.0
        best_index = -1
        best_value = current_value
        for index in range(spins.size):
            candidate = spins.copy()
            candidate[index] *= -1
            candidate_value = objective(candidate)
            delta = candidate_value - current_value
            if delta < best_delta:
                best_delta = float(delta)
                best_index = int(index)
                best_value = float(candidate_value)
        if best_index < 0:
            break
        spins[best_index] *= -1
        current_value = best_value

    return spins.reshape(1, -1).astype(int)


def _solve_ising_sa(
    ising_matrix: np.ndarray,
    initial_binary: np.ndarray | None = None,
    max_iter: int = 2000,
    random_state: int = 0,
) -> np.ndarray:
    """Solve an Ising matrix with local simulated annealing.

    Args:
        ising_matrix: Square Ising matrix with an auxiliary spin.
        initial_binary: Optional initial binary state for the original QUBO
            variables.
        max_iter: Maximum number of annealing steps.
        random_state: Seed for the NumPy random number generator.

    Returns:
        One or more spin solutions encoded as ``-1`` and ``1`` values.

    Raises:
        ValueError: If input shapes or values are invalid.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    rng = np.random.default_rng(int(random_state))
    matrix = np.asarray(ising_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("ising_matrix must be a square matrix")

    if initial_binary is None:
        spins = np.ones(matrix.shape[0], dtype=int)
    else:
        binary = np.asarray(initial_binary, dtype=int)
        if binary.ndim != 1 or binary.shape[0] != matrix.shape[0] - 1:
            raise ValueError("initial_binary must match the QUBO variable size")
        if not np.all((binary == 0) | (binary == 1)):
            raise ValueError("initial_binary must contain only 0/1 values")
        spins = np.r_[2 * binary - 1, 1].astype(int)

    weights = np.triu(matrix)

    def objective(candidate: np.ndarray) -> float:
        """Evaluate the Ising objective for a candidate spin vector.

        Args:
            candidate: Candidate spin vector encoded with ``-1`` and ``1``.

        Returns:
            Ising objective value for ``candidate``.
        """
        return float(np.sum(weights * np.outer(candidate, candidate)))

    current_value = objective(spins)
    best_spins = spins.copy()
    best_value = current_value

    for step in range(int(max_iter)):
        index = int(rng.integers(spins.size))
        candidate = spins.copy()
        candidate[index] *= -1
        candidate_value = objective(candidate)
        delta = candidate_value - current_value
        temperature = max(1e-6, 1.0 * (0.995**step))

        if delta <= 0.0 or rng.random() < float(np.exp(-delta / temperature)):
            spins = candidate
            current_value = float(candidate_value)
            if current_value < best_value:
                best_value = current_value
                best_spins = spins.copy()

    return best_spins.reshape(1, -1).astype(int)


def _solve_ising_kaiwu_cim(
    ising_matrix: np.ndarray,
    target_precision: int = DEFAULT_CIM_TARGET_PRECISION,
    max_bits: int | None = DEFAULT_CIM_MAX_BITS,
    max_precision: int = DEFAULT_CIM_MAX_PRECISION,
    precision_step: int = DEFAULT_CIM_PRECISION_STEP,
    sample_number: int = DEFAULT_CIM_SAMPLE_NUMBER,
    save_dir: str | Path | None = None,
    cleanup_records: bool = True,
    project_no: str | None = None,
    task_mode: Any = "sample",
    interval: int | None = None,
) -> np.ndarray:
    """Split Ising precision and solve directly with Kaiwu CIMOptimizer.

    Args:
        ising_matrix: Ising matrix to submit after precision splitting.
        target_precision: Target precision for the split Ising matrix.
        max_bits: Maximum allowed split variable count.
        max_precision: Maximum source precision to test.
        precision_step: Coarse-search precision step.
        sample_number: Number of solutions requested from Kaiwu CIM.
        save_dir: Optional directory used by Kaiwu checkpoint records.
        cleanup_records: Whether to delete generated checkpoint records.
        project_no: Optional Kaiwu project number.
        task_mode: Kaiwu CIM task mode.
        interval: Optional Kaiwu polling interval.

    Returns:
        Restored spin solutions over the original variables.

    Raises:
        ImportError: If the optional Kaiwu package is unavailable.
        RuntimeError: If Kaiwu CIM does not return a solution.
    """
    try:
        import kaiwu as kw
    except ImportError as exc:
        raise ImportError(
            "MAIFS solver 'kaiwu_cim' requires the optional 'kaiwu' package."
        ) from exc

    explorer = PrecisionSplitExplorer(
        target_precision=target_precision,
        max_bits=max_bits,
        max_precision=max_precision,
        precision_step=precision_step,
    )
    plan = explorer.search(ising_matrix)

    submit_matrix = np.asarray(np.round(plan.split_matrix), dtype=int)
    resolved_save_dir = Path(
        save_dir
        if save_dir is not None
        else Path(tempfile.gettempdir()) / "feature_selection_kaiwu_cim"
    )
    resolved_save_dir.mkdir(parents=True, exist_ok=True)
    if save_dir is not None or kw.common.CheckpointManager.save_dir is None:
        kw.common.CheckpointManager.save_dir = str(resolved_save_dir)

    task_hash = hashlib.md5(
        np.ascontiguousarray(submit_matrix).tobytes()
    ).hexdigest()
    task_name = f"feature_selection_qubo_cim_{task_hash[:16]}_{time_ns()}"
    cim_kwargs = {
        "task_name": task_name,
        "wait": True,
        "project_no": None if project_no is None else str(project_no),
        "task_mode": task_mode,
        "sample_number": int(sample_number),
    }
    if interval is not None:
        cim_kwargs["interval"] = int(interval)
    optimizer = kw.cim.CIMOptimizer(**cim_kwargs)

    try:
        result = optimizer.solve(submit_matrix)
        if result is None:
            raise RuntimeError("CIMOptimizer did not return a solution.")
        result = np.asarray(result)
        result = result.reshape(1, -1) if result.ndim == 1 else result
        result = result[: int(sample_number)]
        return np.asarray([explorer.restore_solution(solution) for solution in result])
    finally:
        if cleanup_records:
            for child in resolved_save_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink()
                    except FileNotFoundError:
                        pass


def qubo_objective(
    binary_state: np.ndarray,
    quadratic_matrix: np.ndarray,
    linear_vector: np.ndarray,
) -> float:
    """Compute a binary QUBO objective value.

    Args:
        binary_state: Binary candidate vector.
        quadratic_matrix: QUBO quadratic term.
        linear_vector: QUBO linear term.

    Returns:
        Objective value computed from the binary state, quadratic matrix, and
        linear vector.
    """
    return float(
        0.5 * binary_state @ quadratic_matrix @ binary_state
        + linear_vector @ binary_state
    )


def _qubo_terms_to_matrix(
    quadratic_matrix: np.ndarray,
    linear_vector: np.ndarray,
) -> np.ndarray:
    """Convert QUBO quadratic and linear terms to upper-triangular form.

    Args:
        quadratic_matrix: QUBO quadratic term.
        linear_vector: QUBO linear term.

    Returns:
        Upper-triangular QUBO matrix.

    Raises:
        ValueError: If terms have incompatible shapes or non-finite values.
    """
    quadratic_matrix = np.asarray(quadratic_matrix, dtype=float)
    linear_vector = np.asarray(linear_vector, dtype=float)
    if quadratic_matrix.ndim != 2 or (
        quadratic_matrix.shape[0] != quadratic_matrix.shape[1]
    ):
        raise ValueError("quadratic_matrix must be a square matrix")
    if (
        linear_vector.ndim != 1
        or linear_vector.shape[0] != quadratic_matrix.shape[0]
    ):
        raise ValueError(
            "linear_vector must be a vector with length matching quadratic_matrix"
        )
    if not np.all(np.isfinite(quadratic_matrix)) or not np.all(
        np.isfinite(linear_vector)
    ):
        raise ValueError(
            "quadratic_matrix and linear_vector must contain only finite values"
        )

    symmetric_quadratic = 0.5 * (quadratic_matrix + quadratic_matrix.T)
    qubo_matrix = np.triu(symmetric_quadratic, 1)
    np.fill_diagonal(
        qubo_matrix,
        linear_vector + 0.5 * np.diag(symmetric_quadratic),
    )
    return qubo_matrix


def solve_qubo(
    quadratic_matrix: np.ndarray,
    linear_vector: np.ndarray,
    initial_state: np.ndarray,
    solver: str = "local_search",
    **solver_kwargs: object,
) -> np.ndarray:
    # pylint: disable=too-many-branches,too-many-statements
    """Validate QUBO inputs and solve them with a built-in solver.

    Args:
        quadratic_matrix: QUBO quadratic term.
        linear_vector: QUBO linear term.
        initial_state: Initial binary state used by local solvers.
        solver: Built-in solver name. Supported values are listed in
            ``AVAILABLE_SOLVERS``.
        **solver_kwargs: Solver-specific keyword arguments.

    Returns:
        Best binary state found by the selected solver.

    Raises:
        ValueError: If inputs or the solver name are invalid.
        ImportError: If ``solver="kaiwu_cim"`` is requested without Kaiwu.
        RuntimeError: If solver execution fails or returns an invalid solution.
    """
    quadratic_matrix = np.asarray(quadratic_matrix, dtype=float)
    linear_vector = np.asarray(linear_vector, dtype=float)
    initial_state = np.asarray(initial_state, dtype=int)
    if quadratic_matrix.ndim != 2 or (
        quadratic_matrix.shape[0] != quadratic_matrix.shape[1]
    ):
        raise ValueError("quadratic_matrix must be a square matrix")
    if linear_vector.shape != (quadratic_matrix.shape[0],):
        raise ValueError(
            "linear_vector must have one entry per QUBO variable"
        )
    if initial_state.shape != linear_vector.shape:
        raise ValueError("initial_state must have the same shape as linear_vector")
    if not np.all(np.isfinite(quadratic_matrix)) or not np.all(
        np.isfinite(linear_vector)
    ):
        raise ValueError(
            "quadratic_matrix and linear_vector must contain only finite values"
        )
    if not np.all((initial_state == 0) | (initial_state == 1)):
        raise ValueError("initial_state must contain only binary 0/1 values")
    quadratic_matrix = 0.5 * (quadratic_matrix + quadratic_matrix.T)

    solver_name = str(solver)
    if solver_name not in AVAILABLE_SOLVERS:
        choices = ", ".join(AVAILABLE_SOLVERS)
        raise ValueError(
            f"Unsupported solver {solver_name!r}. Available solvers: {choices}"
        )

    try:
        ising_matrix = QuadraticLinearSolver().solve(
            quadratic_matrix,
            linear_vector,
        )

        if solver_name == "local_search":
            spin_solutions = _solve_ising_local_search(
                ising_matrix,
                initial_binary=initial_state,
                max_iter=int(solver_kwargs.get("max_iter", 2000)),
            )
        elif solver_name == "sa":
            spin_solutions = _solve_ising_sa(
                ising_matrix,
                initial_binary=initial_state,
                max_iter=int(solver_kwargs.get("max_iter", 2000)),
                random_state=int(solver_kwargs.get("random_state", 0)),
            )
        else:
            max_bits = solver_kwargs.get("max_bits", DEFAULT_CIM_MAX_BITS)
            spin_solutions = _solve_ising_kaiwu_cim(
                ising_matrix,
                target_precision=int(
                    solver_kwargs.get(
                        "target_precision",
                        DEFAULT_CIM_TARGET_PRECISION,
                    )
                ),
                max_bits=None if max_bits is None else int(max_bits),
                max_precision=int(
                    solver_kwargs.get("max_precision", DEFAULT_CIM_MAX_PRECISION)
                ),
                precision_step=int(
                    solver_kwargs.get("precision_step", DEFAULT_CIM_PRECISION_STEP)
                ),
                sample_number=int(
                    solver_kwargs.get("sample_number", DEFAULT_CIM_SAMPLE_NUMBER)
                ),
                save_dir=cast(str | Path | None, solver_kwargs.get("save_dir", None)),
                cleanup_records=bool(solver_kwargs.get("cleanup_records", True)),
                project_no=cast(str | None, solver_kwargs.get("project_no", None)),
                task_mode=solver_kwargs.get("task_mode", "sample"),
                interval=cast(int | None, solver_kwargs.get("interval", None)),
            )
    except ImportError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"MAIFS solver '{solver_name}' failed. "
            f"Problem shape: quadratic_matrix={quadratic_matrix.shape}, "
            f"linear_vector={linear_vector.shape}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    result = initial_state.copy()
    best_value = float("inf")
    spin_array = np.asarray(spin_solutions)
    if spin_array.ndim == 1:
        spin_array = spin_array.reshape(1, -1)
    for spin_solution in spin_array:
        spin_solution = np.asarray(spin_solution, dtype=float)
        if spin_solution.shape != (linear_vector.shape[0] + 1,):
            raise RuntimeError("spin solution must contain one auxiliary spin")
        if not np.all((spin_solution == -1) | (spin_solution == 1)):
            raise RuntimeError("spin solution must contain only -1/1 values")
        binary = np.rint((spin_solution[:-1] * spin_solution[-1] + 1.0) / 2.0)
        binary = binary.astype(int)
        value = qubo_objective(binary, quadratic_matrix, linear_vector)
        if value < best_value:
            best_value = value
            result = binary

    result = np.asarray(result, dtype=int)
    if result.shape != linear_vector.shape or not np.all((result == 0) | (result == 1)):
        raise RuntimeError(
            "MAIFS solver must return a binary vector matching linear_vector."
        )
    return result
