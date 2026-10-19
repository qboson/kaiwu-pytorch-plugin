"""The digits examples must label confusion-matrix axes with actual classes."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


def unavailable_solver(*args, **kwargs):
    """Fail if a plotting-only test tries to construct an external solver."""
    del args, kwargs
    raise AssertionError("Confusion-matrix plotting does not need a sampler")


@pytest.fixture(name="plotter", params=["rbm", "dbn"])
def fixture_plotter(request, monkeypatch, tmp_path):
    """Import each real visualizer and isolate only unavailable SDK imports."""
    for dependency in ("scipy", "sklearn", "seaborn"):
        pytest.importorskip(dependency)
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    plt = pytest.importorskip("matplotlib.pyplot")
    for name, exports in (
        ("kaiwu.classical", ("SimulatedAnnealingOptimizer",)),
        ("kaiwu.cim", ("CIMOptimizer", "PrecisionReducer")),
    ):
        boundary = ModuleType(name)
        for export in exports:
            setattr(boundary, export, unavailable_solver)
        monkeypatch.setitem(sys.modules, name, boundary)
    folder = Path(__file__).resolve().parents[1] / "example"
    if request.param == "rbm":
        path = folder / "rbm_digits/rbm_digits.py"
    else:
        path = folder / "dbn_digits/supervised_dbn_digits.py"
        monkeypatch.syspath_prepend(str(path.parent))
    saved_trainer = sys.modules.pop("dbn_trainer", None)
    spec = importlib.util.spec_from_file_location(f"confusion_{request.param}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        if request.param == "rbm":
            # This helper uses no estimator state or trained model.
            instance = module.RBMRunner.__new__(module.RBMRunner)
        else:
            instance = module.RBMVisualizer(str(tmp_path / "dbn_results"))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(plt, "show", lambda: None)
        yield instance.plot_confusion_matrix, plt
    finally:
        sys.modules.pop("dbn_trainer", None)
        if saved_trainer is not None:
            sys.modules["dbn_trainer"] = saved_trainer
        plt.close("all")


@pytest.mark.parametrize("labels,truth,predicted,expected", [
    ([2, 7], [0, 0, 1], [0, 1, 1], [[1, 1], [0, 1]]),
    (["B cell", "T cell"], [0, 0, 1], [0, 1, 1], [[1, 1], [0, 1]]),
    ([2, 7, 9], [0, 0, 1, 1, 1], [0, 1, 1, 2, 2],
     [[1, 1, 0], [0, 1, 2], [0, 0, 0]]),
    (list(range(10)), list(range(10)), list(range(10)), np.eye(10, dtype=int)),
])
def test_actual_labels_and_counts_are_rendered(plotter, tmp_path, labels,
                                               truth, predicted, expected):
    """Check real heatmap counts, both tick axes and the string-label PDF."""
    plot, plt = plotter
    y_true = np.array([labels[index] for index in truth])
    y_pred = np.array([labels[index] for index in predicted])
    original_true, original_pred = y_true.copy(), y_pred.copy()
    save_pdf = isinstance(labels[0], str)

    plot(y_true, y_pred, save_pdf=save_pdf)

    axes = plt.gcf().axes[0]
    expected_ticks = [str(label) for label in labels]
    assert [tick.get_text() for tick in axes.get_xticklabels()] == expected_ticks
    assert [tick.get_text() for tick in axes.get_yticklabels()] == expected_ticks
    counts = np.asarray(axes.collections[0].get_array()).reshape(len(labels), -1)
    np.testing.assert_array_equal(counts, expected)
    assert [text.get_text() for text in axes.texts] == [
        str(count) for count in np.asarray(expected).flat
    ]
    np.testing.assert_array_equal(y_true, original_true)
    np.testing.assert_array_equal(y_pred, original_pred)
    if save_pdf:
        saved = list(tmp_path.rglob("*.pdf"))
        assert len(saved) == 1
        assert saved[0].read_bytes().startswith(b"%PDF-")
