"""Pin the search history recorded by PrecisionSplitExplorer."""

import numpy as np

from kaiwu.torch_plugin.maifs.qubo import PrecisionSplitExplorer


def _plan():
    explorer = PrecisionSplitExplorer(
        target_precision=2,
        max_bits=8,
        max_precision=2,
        start_precision=2,
    )
    return explorer.search(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_plan_records_the_search_attempts() -> None:
    plan = _plan()

    assert plan.history
    first = plan.history[0]
    assert first["phase"] == "coarse"
    assert first["source_precision"] == 2
    assert first["split_size"] == plan.split_size
    assert isinstance(plan.precision_info, dict)


def test_every_attempt_records_its_precision_info() -> None:
    plan = _plan()
    for attempt in plan.history:
        assert set(attempt) == {
            "source_precision",
            "split_size",
            "phase",
            "precision_info",
        }
