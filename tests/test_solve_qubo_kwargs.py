"""solve_qubo must reject solver options the selected backend does not use."""

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs.qubo import solve_qubo

QUBO = np.array([[1.0, -2.0], [0.0, 1.0]])
LINEAR = np.array([0.5, -0.5])
INITIAL = np.array([0, 0])


def test_local_search_rejects_unknown_option() -> None:
    with pytest.raises(ValueError, match="Unsupported option"):
        solve_qubo(QUBO, LINEAR, INITIAL, "local_search", max_iters=10)


def test_kaiwu_cim_rejects_unknown_option_before_importing_kaiwu() -> None:
    with pytest.raises(ValueError, match="Unsupported option"):
        solve_qubo(QUBO, LINEAR, INITIAL, "kaiwu_cim", bogus_flag=True)


def test_local_search_still_accepts_max_iter() -> None:
    solution = solve_qubo(QUBO, LINEAR, INITIAL, "local_search", max_iter=5)
    assert solution.shape == INITIAL.shape
    assert set(np.unique(solution)).issubset({0, 1})
