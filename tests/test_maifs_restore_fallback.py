"""Without the vote helper, restore must call the two-argument API."""

import numpy as np

import kaiwu.preprocess as preprocess

from kaiwu.torch_plugin.maifs.qubo import PrecisionSplitExplorer, PrecisionSplitPlan


def _plan():
    return PrecisionSplitPlan(
        source_precision=8,
        target_precision=8,
        max_bits=10,
        adjusted_matrix=np.zeros((2, 2)),
        split_matrix=np.zeros((2, 2)),
        last_var_idx=np.array([1]),
        split_size=2,
        precision_info={},
        history=[],
    )


def test_the_fallback_calls_the_two_argument_helper(monkeypatch):
    calls = []

    def fake_restore(solution, last_var_idx):
        calls.append((solution.copy(), np.array(last_var_idx)))
        return np.array([7])

    monkeypatch.delattr(preprocess, "restore_splitted_solution", raising=False)
    monkeypatch.setattr(preprocess, "restore_split_solution", fake_restore)
    explorer = PrecisionSplitExplorer(target_precision=8, max_bits=10)
    explorer.plan = _plan()

    restored = explorer.restore_solution(np.array([1, 0]), vote=True)

    assert restored.tolist() == [7]
    assert len(calls) == 1
    assert calls[0][0].tolist() == [1, 0]
