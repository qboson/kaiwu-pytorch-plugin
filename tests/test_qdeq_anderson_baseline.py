"""The example's Anderson baseline must match the library solver."""

import sys
import types
from pathlib import Path

import pytest
import torch

from kaiwu.torch_plugin.qdeq import _anderson_solver

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qdeq" / "simple"


@pytest.fixture
def etth1(monkeypatch):
    """Imports etth1_common with a stubbed pandas module."""
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda path: None
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("etth1_common", None)
    import etth1_common

    yield etth1_common
    sys.modules.pop("etth1_common", None)


def test_baseline_and_library_solver_reach_the_same_point(etth1):
    torch.manual_seed(0)
    layer = etth1.EquilibriumBlock(d_model=4, n_heads=2, d_ff=8)
    config = etth1.build_config()
    history = torch.randn(2, 3, 4)
    baseline = etth1.AndersonDEQ(layer, config).eval()

    with torch.no_grad():
        out = baseline(history)
        reference, stats = _anderson_solver(
            lambda z: layer(z, history),
            torch.zeros_like(history),
            config,
            config.f_thres,
            config.f_eps,
        )

    assert torch.isfinite(out).all()
    assert stats["nstep"] >= 1
    expected = layer(reference, history)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
