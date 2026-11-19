"""Generation plots tolerate repeated properties and empty target groups."""
import importlib
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
@pytest.fixture(params=["cim_sa", "sqa"])
def plots(request, monkeypatch):
    prefix = "qc_mol_design"
    saved = {k: v for k, v in sys.modules.items()
             if k == prefix or k.startswith(prefix + ".")}
    for name in saved:
        sys.modules.pop(name)
    root = Path(__file__).resolve().parents[1] / "example/molecular_design" / request.param
    monkeypatch.syspath_prepend(str(root))
    try:
        with warnings.catch_warnings():
            module = importlib.import_module("qc_mol_design.evaluation.visualization")
            try:
                yield module
            finally:
                module.plt.close("all")
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)



@pytest.mark.parametrize("values", [[2.] * 5, [1., 2., 3., 4., np.inf, 5., np.nan], [1., 2., 3., 4., 5.]])
def test_kde_writes_plot_for_degenerate_or_nonfinite_values(plots, tmp_path, monkeypatch, values):
    path = tmp_path / "density.png"
    data = pd.DataFrame({"logp": values, "qed": values})
    captured = []
    save = plots.plt.savefig

    def save_and_capture(*args, **kwargs):
        captured.extend(plots.plt.gcf().axes)
        return save(*args, **kwargs)

    monkeypatch.setattr(plots.plt, "savefig", save_and_capture)
    plots.plot_kde(data, data, save_path=str(path))
    for axis in captured:
        assert len(axis.lines) == 2
        for line in axis.lines:
            assert np.isfinite(line.get_xdata()).all()
            assert np.isfinite(line.get_ydata()).all()
            if values == [2.] * 5:
                np.testing.assert_array_equal(line.get_xdata(), [2., 2.])
            else:
                assert len(line.get_xdata()) == 300
    assert path.read_bytes().startswith(b"\x89PNG")
    assert not plots.plt.get_fignums()


def test_violin_skips_empty_target_groups(plots, tmp_path):
    path = tmp_path / "violin.png"
    plots.plot_violin_sas({"empty": [], "valid": [1., 2., 3.]}, save_path=str(path))
    assert path.read_bytes().startswith(b"\x89PNG")
    assert not plots.plt.get_fignums()


def test_violin_with_only_empty_groups_creates_no_figure(plots, tmp_path):
    path = tmp_path / "empty.png"
    plots.plot_violin_sas({"empty": []}, save_path=str(path))
    assert not path.exists()
    assert not plots.plt.get_fignums()
