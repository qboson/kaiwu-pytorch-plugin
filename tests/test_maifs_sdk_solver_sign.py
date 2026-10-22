# -*- coding: utf-8 -*-
"""Kaiwu SDK 求解器符号约定的判别性测试。

Kaiwu SDK 求解器(kw.classical / kw.cim)按 ``s^T M s`` 的**最大化**约定求解:
SDK 自带的 ``kw.conversion.qubo_matrix_to_ising_matrix`` 对 QUBO 取负编码,
配合 ``kw.classical.BruteForceOptimizer`` 会返回 QUBO 的最小值点;而
``QuadraticLinearSolver`` 生成的是正向编码(``s^T M s = f(x) + const``),
直接提交给 SDK 求解器会得到 QUBO 的**最大值点**——即最差特征组合。
本文件用按 SDK 约定行为的桩求解器钉住该契约。
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

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

from kaiwu.torch_plugin.maifs import qubo
from kaiwu.torch_plugin.maifs.qubo import solve_qubo

# f(x) = 2 x0 x1 + x0 + x1:最小值点 [0, 0](f=0),最大值点 [1, 1](f=4)
QUADRATIC = np.array([[0.0, 2.0], [2.0, 0.0]])
LINEAR = np.array([1.0, 1.0])


def _enumerate_argmax_spins(matrix: np.ndarray) -> np.ndarray:
    """按 SDK 约定(最大化 s^T M s)暴力枚举最优自旋解。"""
    best_spins, best_value = None, -np.inf
    for spins in itertools.product((-1, 1), repeat=matrix.shape[0]):
        value = float(np.asarray(spins) @ matrix @ np.asarray(spins))
        if value > best_value:
            best_spins, best_value = np.asarray(spins), value
    return best_spins


def _decode(spins: np.ndarray) -> np.ndarray:
    """按辅助自旋规范解码二元变量:x_i = (s_i * s_aux + 1) / 2。"""
    return (spins[:-1] * spins[-1] + 1) / 2


def test_sa_solver_returns_qubo_minimizer(monkeypatch) -> None:
    """sa 路径必须让 SDK 的最大化约定仍然返回 QUBO 最小值点。"""
    seen: dict[str, np.ndarray] = {}

    class StubSASolver:
        """按 SDK 约定(最大化 s^T M s)求解的桩,替代需要许可证的真 SA。"""

        def __init__(self, **kwargs) -> None:
            pass

        def solve(self, matrix):
            seen["matrix"] = np.asarray(matrix, dtype=float)
            return _enumerate_argmax_spins(seen["matrix"]).reshape(1, -1)

    monkeypatch.setattr(kaiwu.classical, "SimulatedAnnealingOptimizer", StubSASolver)

    selected = solve_qubo(QUADRATIC, LINEAR, np.array([1, 1]), solver="sa")

    assert "matrix" in seen
    assert selected.tolist() == [0, 0]


def test_kaiwu_cim_path_receives_sdk_convention_matrix(monkeypatch) -> None:
    """kaiwu_cim 路径提交给 SDK 的矩阵同样必须满足最大化约定。"""
    seen: dict[str, np.ndarray] = {}

    def fake_cim_solver(ising_matrix, **kwargs):
        seen["matrix"] = np.asarray(ising_matrix, dtype=float)
        return np.ones((1, seen["matrix"].shape[0]), dtype=int)

    monkeypatch.setattr(qubo, "_solve_ising_kaiwu_cim", fake_cim_solver)

    solve_qubo(QUADRATIC, LINEAR, np.array([1, 1]), solver="kaiwu_cim")

    assert "matrix" in seen
    assert _decode(_enumerate_argmax_spins(seen["matrix"])).tolist() == [0.0, 0.0]


def test_local_search_still_minimizes_positive_encoding() -> None:
    """local_search 直接最小化正向编码矩阵,回归保护:仍返回最小值点。"""
    selected = solve_qubo(
        QUADRATIC, LINEAR, np.array([1, 1]), solver="local_search"
    )

    assert selected.tolist() == [0, 0]
