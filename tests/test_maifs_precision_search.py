"""Precision search and lifecycle regressions without a licensed solver."""
import itertools
import weakref

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs import qubo


def install_sizes(monkeypatch, explorer, sizes):
    """Return unique mappings for deterministic split sizes."""
    calls = []
    references = []

    def build(matrix, precision):
        # Completed failures must not accumulate full split plans.
        assert sum(ref() is not None for ref in references) <= 1
        calls.append(precision)
        size = sizes[precision]
        plan = qubo.PrecisionSplitPlan(
            source_precision=precision, target_precision=explorer.target_precision,
            max_bits=explorer.max_bits, adjusted_matrix=matrix.copy(),
            split_matrix=np.zeros((size, size)), last_var_idx=[precision],
            split_size=size, precision_info={"marker": precision}, history=[],
        )
        references.append(weakref.ref(plan))
        return plan

    monkeypatch.setattr(explorer, "_build_plan", build)
    return calls


@pytest.mark.parametrize("feasible", list(itertools.product((False, True), repeat=5)))
@pytest.mark.parametrize("step", [1, 2, 4, 10])
def test_highest_feasible_without_monotonicity(monkeypatch, feasible, step):
    explorer = qubo.PrecisionSplitExplorer(
        target_precision=3, max_bits=5, start_precision=3,
        max_precision=7, precision_step=step,
    )
    sizes = {precision: 5 if fits else 6 for precision, fits in zip(range(3, 8), feasible)}
    calls = install_sizes(monkeypatch, explorer, sizes)
    candidates = [p for p, size in sizes.items() if size <= 5]
    if not candidates:
        with pytest.raises(RuntimeError, match="No feasible precision"):
            explorer.search(np.eye(2))
        assert explorer.plan is None
        assert set(calls) == set(sizes)
    else:
        plan = explorer.search(np.eye(2))
        assert plan.source_precision == max(candidates)
        assert explorer.plan is plan
        assert plan.last_var_idx == [max(candidates)]
        assert plan.history == explorer.history
        assert all(p in calls for p in range(max(candidates) + 1, 8))
    assert len(calls) == len(set(calls))
    assert len(calls) <= 5
    assert [x["source_precision"] for x in explorer.history] == calls
    assert all(x["phase"] in ("coarse", "fine") for x in explorer.history)


@pytest.mark.parametrize("step", [1, 3, 8])
def test_single_precision_interval(monkeypatch, step):
    explorer = qubo.PrecisionSplitExplorer(
        max_bits=4, start_precision=5, max_precision=5, precision_step=step,
    )
    calls = install_sizes(monkeypatch, explorer, {5: 4})
    assert explorer.search(np.eye(2)).source_precision == 5
    assert calls == [5]


@pytest.mark.parametrize("failure", ["shape", "capacity", "conversion", "helper", "infeasible"])
def test_failed_reuse_cannot_restore_previous_mapping(monkeypatch, failure):
    explorer = qubo.PrecisionSplitExplorer(max_bits=5, start_precision=3, max_precision=5)
    install_sizes(monkeypatch, explorer, {3: 4, 4: 4, 5: 4})
    old_plan = explorer.search(np.eye(2))
    old_history = list(old_plan.history)
    restores = []
    monkeypatch.setattr(qubo, "_restore_kaiwu_split_solution",
                        lambda solution, mapping, vote: restores.append((mapping, vote)) or solution)
    explorer.restore_solution(np.ones(4), vote=True)
    assert restores == [([5], True)]
    if failure == "shape":
        value = np.ones((2, 3))
    elif failure == "capacity":
        value = np.eye(6)
    elif failure == "conversion":
        class BrokenArray:
            def __array__(self, *args, **kwargs):
                raise ValueError("conversion failed")
        value = BrokenArray()
    elif failure == "helper":
        value = np.eye(2)
        def broken(matrix, precision):
            raise RuntimeError("preprocessing failed")
        monkeypatch.setattr(explorer, "_build_plan", broken)
    else:
        value = np.eye(2)
        install_sizes(monkeypatch, explorer, {3: 6, 4: 6, 5: 6})
    with pytest.raises((ValueError, RuntimeError)):
        explorer.search(value)
    assert explorer.plan is None
    with pytest.raises(ValueError, match="successfully"):
        explorer.restore_solution(np.ones(4))
    assert len(restores) == 1
    assert old_plan.history == old_history
    if failure != "infeasible":
        assert explorer.history == []
    else:
        assert len(explorer.history) == 3
    install_sizes(monkeypatch, explorer, {3: 4, 4: 6, 5: 6})
    new_plan = explorer.search(np.eye(2))
    assert new_plan.source_precision == 3
    explorer.restore_solution(np.ones(4))
    assert restores[-1] == ([3], False)
    assert new_plan.history is not explorer.history


def test_partial_failure_retains_only_current_attempts(monkeypatch):
    explorer = qubo.PrecisionSplitExplorer(max_bits=5, start_precision=3, max_precision=7)
    install_sizes(monkeypatch, explorer, {3: 4, 7: 6})
    original = explorer._build_plan
    def fail_later(matrix, precision):
        if precision == 7:
            raise RuntimeError("second attempt failed")
        return original(matrix, precision)
    monkeypatch.setattr(explorer, "_build_plan", fail_later)
    with pytest.raises(RuntimeError, match="second attempt"):
        explorer.search(np.eye(2))
    assert explorer.plan is None
    assert [x["source_precision"] for x in explorer.history] == [3]


def test_real_sdk_nonmonotonic_split_and_selected_restoration():
    """Pinned SDK preprocessing on four variables, never a sampler/QPU."""
    if not hasattr(qubo.kw, "preprocess"):
        pytest.skip("Kaiwu preprocessing is unavailable")
    matrix = np.triu(np.random.default_rng(8).normal(size=(4, 4)), 1)
    explorer = qubo.PrecisionSplitExplorer(
        target_precision=3, min_precision=3, max_precision=12, max_bits=11,
    )
    oracle = {p: explorer._build_plan(matrix, p) for p in range(3, 13)}
    feasible = [p for p, plan in oracle.items() if plan.split_size <= 11]
    assert max(feasible) == 9
    assert oracle[7].split_size == 14
    plan = explorer.search(matrix)
    assert plan.source_precision == 9
    np.testing.assert_array_equal(plan.split_matrix, oracle[9].split_matrix)
    solution = np.ones(plan.split_size, dtype=int)
    np.testing.assert_array_equal(
        explorer.restore_solution(solution),
        qubo._restore_kaiwu_split_solution(solution, oracle[9].last_var_idx, False),
    )
    assert len({x["source_precision"] for x in plan.history}) == len(plan.history)
    assert len(plan.history) <= 10
