"""PrecisionSplitExplorer.restore_solution must validate its inputs."""

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs.qubo import PrecisionSplitExplorer

MATRIX = np.array([[0.0, 1.0], [1.0, 0.0]])


def _explorer_with_plan() -> PrecisionSplitExplorer:
    explorer = PrecisionSplitExplorer(
        target_precision=2,
        max_bits=8,
        max_precision=2,
        start_precision=2,
    )
    explorer.search(MATRIX)
    return explorer


@pytest.mark.parametrize("value", [-1, 2, 0.5])
def test_non_binary_solution_is_rejected(value) -> None:
    explorer = _explorer_with_plan()
    solution = np.full(explorer.plan.split_size, value)
    with pytest.raises(ValueError, match="only binary 0/1 values"):
        explorer.restore_solution(solution)


@pytest.mark.parametrize("vote", ["yes", 1, None])
def test_non_boolean_vote_is_rejected(vote) -> None:
    explorer = _explorer_with_plan()
    solution = np.zeros(explorer.plan.split_size, dtype=int)
    with pytest.raises(TypeError, match="vote must be a boolean"):
        explorer.restore_solution(solution, vote=vote)


def test_binary_solution_still_restores() -> None:
    explorer = _explorer_with_plan()
    for value in (0, 1):
        solution = np.full(explorer.plan.split_size, value, dtype=int)
        restored = np.asarray(explorer.restore_solution(solution))
        assert restored.shape == solution.shape
    restored_vote = np.asarray(
        explorer.restore_solution(np.ones(explorer.plan.split_size, dtype=int), vote=True)
    )
    assert restored_vote.shape[0] == explorer.plan.split_size
