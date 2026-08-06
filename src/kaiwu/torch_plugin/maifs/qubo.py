from __future__ import annotations

# pylint: disable=too-many-lines

import os
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


def _init_kaiwu_license_from_env() -> None:
    """使用环境变量初始化 Kaiwu license。

    该函数只在同时存在 LICENSE_USER_ID 和 LICENSE_SDK_CODE 时调用 Kaiwu 初始化，
    不会读取或打印具体密钥内容。

    Args:
        无。

    Returns:
        None: 函数只修改 Kaiwu 运行时授权状态，不返回值。

    Raises:
        RuntimeError: 当 Kaiwu license 初始化失败时抛出。
    """
    user_id = os.environ.get("LICENSE_USER_ID")
    sdk_code = os.environ.get("LICENSE_SDK_CODE")
    if not user_id or not sdk_code:
        return

    import kaiwu.license as license_manager

    try:
        license_manager.init(user_id, sdk_code)
    except Exception as exc:
        raise RuntimeError(
            "Kaiwu license initialization failed. Check whether LICENSE_USER_ID "
            "and LICENSE_SDK_CODE are correct, and whether the machine can reach "
            "the Kaiwu license server."
        ) from exc


@dataclass
class PrecisionSplitPlan:
    """保存 Kaiwu CIM 提交前的精度调整和变量拆分方案。

    该类只保存一次精度拆分搜索的结果，不负责执行 CIM 求解。

    Args:
        source_precision (int): 原始 Ising 矩阵被调整到的源精度。
        target_precision (int): 拆分后提交矩阵的目标精度。
        max_bits (int): 拆分后允许的最大变量数量。
        adjusted_matrix (np.ndarray): 精度调整后的 Ising 矩阵。
        split_matrix (np.ndarray): 变量拆分后的 Ising 矩阵。
        last_var_idx (np.ndarray): 拆分变量到原始变量的映射。
        split_size (int): 拆分后矩阵的变量数量。
        precision_info (dict[str, Any]): Kaiwu 精度计算返回的信息。
        history (list[dict[str, Any]]): 精度搜索过程记录。

    Returns:
        PrecisionSplitPlan: 精度拆分方案数据对象。

    Raises:
        无。
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
    """Find a feasible kaiwu precision/split plan under a bit-size limit.

    This is a local copy of the helper used in ``kaiwu_test``. It delays kaiwu
    imports until use, so the default non-CIM QUBO backends do not require kaiwu.

    Args:
        target_precision (int, optional): 拆分后的目标精度。默认为 8。
        max_bits (int | None, optional): 拆分后允许的最大变量数量。默认为 None。
        max_precision (int, optional): 搜索时允许的最大源精度。默认为 32。
        min_precision (int | None, optional): 兼容原版的起始精度。默认为 None。
        min_increment (float | None, optional): 指定拆分最小增量。默认为 None。
        penalty (float | None, optional): 拆分惩罚系数。默认为 None。
        round_to_increment (bool, optional): 是否按增量取整。默认为 True。
        start_precision (int | None, optional): 搜索起始精度。默认为 None。
        precision_step (int, optional): 精度搜索步长。默认为 4。

    Returns:
        PrecisionSplitExplorer: 精度拆分搜索器实例。

    Raises:
        ValueError: 当精度范围、步长或变量数量限制不合法时抛出。

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
        """初始化 Kaiwu 精度拆分搜索器。

        该函数用于 MAIFS 特征选择流程，保持输入校验、计算逻辑和返回结果一致。

        Args:
            target_precision (int, optional): 拆分后的目标精度。默认为 8。
            max_bits (int | None, optional): 拆分后允许的最大变量数量。默认为 None。
            max_precision (int, optional): 搜索时允许的最大源精度。默认为 32。
            min_precision (int | None, optional): 兼容原版的起始精度。默认为 None。
            min_increment (float | None, optional): 指定拆分最小增量。默认为 None。
            penalty (float | None, optional): 拆分惩罚系数。默认为 None。
            round_to_increment (bool, optional): 是否按增量取整。默认为 True。
            start_precision (int | None, optional): 搜索起始精度。默认为 None。
            precision_step (int, optional): 粗搜索时精度递增步长。默认为 4。

        Returns:
            None: 函数只修改对象状态，不返回值。

        Raises:
            ValueError: 当精度或变量数量配置不合法时抛出。
            TypeError: 当 precision_step 不是整数时抛出。
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
        """计算精度拆分时使用的最小增量。

        该函数用于 MAIFS 特征选择流程，保持输入校验、计算逻辑和返回结果一致。

        Args:
            ising_matrix (np.ndarray): 输入 Ising 矩阵。
            min_increment (float | None): 用户指定的最小增量。

        Returns:
            float: 用于 Kaiwu 拆分接口的最小增量。

        Raises:
            ValueError: 当输入参数不合法时抛出。

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
        """把 Ising 矩阵调整到指定整数精度。

        该函数用于 MAIFS 特征选择流程，保持输入校验、计算逻辑和返回结果一致。

        Args:
            ising_matrix (np.ndarray): 原始 Ising 矩阵。
            precision (int): 目标整数精度。
        Returns:
            tuple[np.ndarray, dict[str, Any]]: 精度调整后的矩阵和精度信息。

        Raises:
            ImportError: 当 Kaiwu 精度处理接口不可用时抛出。

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
        """构造一次精度调整和变量拆分方案。

        该函数用于 MAIFS 特征选择流程，保持输入校验、计算逻辑和返回结果一致。

        Args:
            ising_matrix (np.ndarray): 原始 Ising 矩阵。
            source_precision (int): 本次尝试的源精度。
        Returns:
            PrecisionSplitPlan: 拆分后的矩阵、变量映射和变量数量。

        Raises:
            ImportError: 当 Kaiwu 拆分接口不可用时抛出。

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
        """构造拆分方案并记录本次精度搜索历史。

        该函数是 search() 的内部步骤，用于记录 coarse 或 fine 阶段的尝试结果。

        Args:
            ising_matrix (np.ndarray): 原始 Ising 矩阵。
            source_precision (int): 本次尝试使用的源精度。
            phase (str): 搜索阶段名称，例如 "coarse" 或 "fine"。
        Returns:
            PrecisionSplitPlan: 本次精度尝试得到的拆分方案。

        Raises:
            ValueError: 当 Ising 矩阵或精度参数不合法时抛出。
            RuntimeError: 当 Kaiwu 预处理接口执行失败时抛出。
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
        """搜索满足变量数量限制的最高可行拆分精度。

        该函数用于 MAIFS 特征选择流程，保持输入校验、计算逻辑和返回结果一致。

        Args:
            ising_matrix (np.ndarray): 原始 Ising 矩阵。

        Returns:
            PrecisionSplitPlan: 搜索得到的拆分方案。

        Raises:
            ValueError: 当原始矩阵规模已经超过 max_bits 时抛出。
            RuntimeError: 当没有找到可行精度时抛出。

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
        """把拆分后问题的解恢复到原始变量空间。

        该函数用于 MAIFS 特征选择流程，保持输入校验、计算逻辑和返回结果一致。

        Args:
            solution (np.ndarray): 拆分后问题的解。
        Returns:
            np.ndarray: 恢复后的原始变量解。

        Raises:
            ValueError: 当尚未执行 search 时抛出。
            ImportError: 当 Kaiwu 恢复接口不可用时抛出。

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
    """定义 q、c 到 Ising 矩阵的转换 Adapter。
    """

    @staticmethod
    def _qubo_matrix_to_ising_matrix(qubo_matrix: np.ndarray) -> np.ndarray:
        """用本地公式把 QUBO 上三角矩阵转换为带辅助自旋的 Ising 矩阵。
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
        q: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """接收 q、c 并返回转换后的 Ising 矩阵。
        """
        qubo_matrix = _qubo_terms_to_matrix(q, c)
        return self._qubo_matrix_to_ising_matrix(qubo_matrix)


def _solve_ising_local_search(
    ising_matrix: np.ndarray,
    initial_binary: np.ndarray | None = None,
    max_iter: int = 2000,
) -> np.ndarray:
    """使用本地贪心翻转求解 Ising 矩阵。"""
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
    """使用普通本地模拟退火求解 Ising 矩阵。"""
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
        temperature = max(1e-6, 1.0 * (0.995 ** step))

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
    """精度拆分 Ising 矩阵后直接调用 Kaiwu CIMOptimizer。"""
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
    _init_kaiwu_license_from_env()

    submit_matrix = np.asarray(np.round(plan.split_matrix), dtype=int)
    resolved_save_dir = Path(
        save_dir
        if save_dir is not None
        else Path(tempfile.gettempdir()) / "feature_selection_kaiwu_cim"
    )
    resolved_save_dir.mkdir(parents=True, exist_ok=True)
    if save_dir is not None or kw.common.CheckpointManager.save_dir is None:
        kw.common.CheckpointManager.save_dir = str(resolved_save_dir)

    task_hash = kw.cim.CIMOptimizer._ising_to_md5_hash(submit_matrix)
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


def qubo_objective(s: np.ndarray, q: np.ndarray, c: np.ndarray) -> float:
    """计算二进制 QUBO 目标函数值。
    """
    return float(0.5 * s @ q @ s + c @ s)


def _qubo_terms_to_matrix(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """把 QUBO 二次项和一次项转换为上三角矩阵形式。
    """
    q = np.asarray(q, dtype=float)
    c = np.asarray(c, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be a square matrix")
    if c.ndim != 1 or c.shape[0] != q.shape[0]:
        raise ValueError("c must be a vector with length matching q")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(c)):
        raise ValueError("q and c must contain only finite values")

    hessian = 0.5 * (q + q.T)
    qubo_matrix = np.triu(hessian, 1)
    np.fill_diagonal(qubo_matrix, c + 0.5 * np.diag(hessian))
    return qubo_matrix


def solve_qubo(
    q: np.ndarray,
    c: np.ndarray,
    s0: np.ndarray,
    solver: str = "local_search",
    **solver_kwargs: object,
) -> np.ndarray:
    # pylint: disable=too-many-branches,too-many-statements
    """校验 QUBO 输入并使用指定内置求解器求解。
    """
    q = np.asarray(q, dtype=float)
    c = np.asarray(c, dtype=float)
    s0 = np.asarray(s0, dtype=int)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be a square matrix")
    if c.shape != (q.shape[0],):
        raise ValueError("c must have one entry per QUBO variable")
    if s0.shape != c.shape:
        raise ValueError("s0 must have the same shape as c")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(c)):
        raise ValueError("q and c must contain only finite values")
    if not np.all((s0 == 0) | (s0 == 1)):
        raise ValueError("s0 must contain only binary 0/1 values")
    q = 0.5 * (q + q.T)

    solver_name = str(solver)
    if solver_name not in AVAILABLE_SOLVERS:
        choices = ", ".join(AVAILABLE_SOLVERS)
        raise ValueError(
            f"Unsupported solver {solver_name!r}. Available solvers: {choices}"
        )

    try:
        ising_matrix = QuadraticLinearSolver().solve(q, c)

        if solver_name == "local_search":
            spin_solutions = _solve_ising_local_search(
                ising_matrix,
                initial_binary=s0,
                max_iter=int(solver_kwargs.get("max_iter", 2000)),
            )
        elif solver_name == "sa":
            spin_solutions = _solve_ising_sa(
                ising_matrix,
                initial_binary=s0,
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
            f"Problem shape: q={q.shape}, c={c.shape}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    result = s0.copy()
    best_value = float("inf")
    spin_array = np.asarray(spin_solutions)
    if spin_array.ndim == 1:
        spin_array = spin_array.reshape(1, -1)
    for spin_solution in spin_array:
        spin_solution = np.asarray(spin_solution, dtype=float)
        if spin_solution.shape != (c.shape[0] + 1,):
            raise RuntimeError("spin solution must contain one auxiliary spin")
        if not np.all((spin_solution == -1) | (spin_solution == 1)):
            raise RuntimeError("spin solution must contain only -1/1 values")
        binary = np.rint((spin_solution[:-1] * spin_solution[-1] + 1.0) / 2.0)
        binary = binary.astype(int)
        value = qubo_objective(binary, q, c)
        if value < best_value:
            best_value = value
            result = binary

    result = np.asarray(result, dtype=int)
    if result.shape != c.shape or not np.all((result == 0) | (result == 1)):
        raise RuntimeError("MAIFS solver must return a binary vector matching c.")
    return result
