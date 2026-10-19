from __future__ import annotations

import os
import sys

import numpy as np
import pytest

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

import kaiwu

kaiwu_src_path = os.path.join(src_root, "kaiwu")
if kaiwu_src_path not in kaiwu.__path__:
    kaiwu.__path__ = [kaiwu_src_path] + list(kaiwu.__path__)

for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

from kaiwu.torch_plugin.maifs.qubo import (  # noqa: E402
    PrecisionSplitExplorer,
    _binary_from_solver_solution,
    _solve_ising_local_search,
    solve_qubo,
)


def test_solve_qubo_rejects_unknown_solver() -> None:
    """未知 solver 名应在进入求解前报出可用选项。"""
    quadratic = np.zeros((2, 2))
    linear = np.zeros(2)
    initial = np.zeros(2, dtype=int)
    with pytest.raises(ValueError, match="Available solvers"):
        solve_qubo(quadratic, linear, initial, solver="unknown")


def test_local_search_rejects_invalid_arguments() -> None:
    """局部搜索对迭代次数与矩阵形状的校验。"""
    matrix = np.zeros((3, 3))
    with pytest.raises(ValueError):
        _solve_ising_local_search(matrix, max_iter=0)
    with pytest.raises(ValueError):
        _solve_ising_local_search(np.zeros((2, 3)))


def test_binary_decoding_rejects_invalid_solutions() -> None:
    """解码只接受带一个辅助自旋的 ±1 解。"""
    with pytest.raises(RuntimeError):
        _binary_from_solver_solution(np.array([0, 1, 1]), 2)
    with pytest.raises(RuntimeError):
        _binary_from_solver_solution(np.array([0.5, -1, 1]), 2)
    with pytest.raises(RuntimeError):
        _binary_from_solver_solution(np.zeros((0, 3)), 2)
    with pytest.raises(RuntimeError):
        _binary_from_solver_solution(np.ones((2, 2, 2)), 2)


def test_precision_split_explorer_validates_arguments() -> None:
    """精度搜索器的参数校验（不触发 Kaiwu SDK 调用）。"""
    with pytest.raises(ValueError):
        PrecisionSplitExplorer()
    with pytest.raises(ValueError):
        PrecisionSplitExplorer(target_precision=1, max_bits=100)
    with pytest.raises(ValueError):
        PrecisionSplitExplorer(start_precision=1, max_bits=100)
    with pytest.raises(ValueError):
        PrecisionSplitExplorer(start_precision=8, max_precision=4, max_bits=100)
    with pytest.raises(TypeError):
        PrecisionSplitExplorer(max_bits=100, precision_step=2.5)
    with pytest.raises(ValueError):
        PrecisionSplitExplorer(max_bits=100, precision_step=0)

    explorer = PrecisionSplitExplorer(target_precision=6, max_bits=100)
    assert explorer.min_precision == 6
    assert PrecisionSplitExplorer(
        target_precision=8, min_precision=6, max_bits=100
    ).start_precision == 6


def test_min_increment_resolution() -> None:
    """最小增量默认取矩阵取值间的最小正差。"""
    matrix = np.array([[0.0, 2.0], [0.0, 5.0]])
    assert PrecisionSplitExplorer._resolve_min_increment(matrix, 0.25) == 0.25
    assert PrecisionSplitExplorer._resolve_min_increment(matrix, None) == 2
    assert PrecisionSplitExplorer._resolve_min_increment(np.zeros((2, 2)), None) == 1
