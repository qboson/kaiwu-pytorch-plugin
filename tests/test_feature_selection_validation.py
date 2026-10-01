"""Constructor contract and feasible-count regressions for MAIFS."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

src_root = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_root))

import kaiwu

kaiwu.__path__ = [str(src_root / "kaiwu")] + list(kaiwu.__path__)
for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith("kaiwu.torch_plugin."):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper


@pytest.mark.parametrize("model", [None, object(), lambda inputs: inputs])
def test_constructor_rejects_non_module_models(model):
    """The documented model TypeError must occur during construction."""
    with pytest.raises(TypeError, match="model"):
        FeatureSelectionWrapper(model, feature_dim=4)


@pytest.mark.parametrize(
    "options, parameter",
    [
        ({"feature_dim": 0}, "feature_dim"),
        ({"feature_dim": -1}, "feature_dim"),
        ({"feature_dim": None}, "feature_dim"),
        ({"min_selected_features": -1}, "min_selected_features"),
        ({"min_selected_features": 5}, "min_selected_features"),
        ({"max_selected_features": -1}, "max_selected_features"),
        ({"max_selected_features": 5}, "max_selected_features"),
        ({"min_selected_features": 3, "max_selected_features": 2}, "min_selected_features"),
        ({"cardinality_k": -1}, "cardinality_k"),
        ({"cardinality_k": 5}, "cardinality_k"),
        ({"cardinality_k": 1, "min_selected_features": 2}, "cardinality_k"),
        ({"cardinality_k": 3, "max_selected_features": 2}, "cardinality_k"),
        ({"cardinality_k": 0, "min_selected_features": 1}, "cardinality_k"),
    ],
)
def test_constructor_rejects_infeasible_feature_counts(options, parameter):
    """Invalid domains and conflicting bounds must fail before any training."""
    arguments = {"feature_dim": 4, **options}
    with pytest.raises(ValueError, match=parameter):
        FeatureSelectionWrapper(nn.Identity(), **arguments)


@pytest.mark.parametrize(
    "parameter",
    ["feature_dim", "min_selected_features", "max_selected_features", "cardinality_k"],
)
@pytest.mark.parametrize("value", [1.5, -0.5, np.nan, np.inf, "invalid"])
def test_constructor_reports_invalid_count_types(parameter, value):
    """Counts must be finite integers with errors naming the supplied argument."""
    arguments = {"feature_dim": 4, parameter: value}
    with pytest.raises(ValueError, match=parameter):
        FeatureSelectionWrapper(nn.Identity(), **arguments)


@pytest.mark.parametrize("solver", ["unknown", "SA", None])
def test_constructor_rejects_unknown_solver_names(solver):
    """Unsupported solver names should not be deferred until update_mask()."""
    with pytest.raises(ValueError, match="solver"):
        FeatureSelectionWrapper(nn.Identity(), 4, solver=solver)


@pytest.mark.parametrize("solver", ["local_search", "sa", "kaiwu_cim"])
@pytest.mark.parametrize("cardinality", [np.float64(3.0), "3"])
def test_constructor_preserves_integer_like_inputs_and_supported_solvers(solver, cardinality):
    """Integer strings, NumPy scalars and integral floats remain usable."""
    wrapper = FeatureSelectionWrapper(
        nn.Identity(), feature_dim="10", cardinality_k=cardinality,
        min_selected_features=np.int64(2), max_selected_features=8.0, solver=solver,
    )
    assert wrapper.feature_dim == 10
    assert wrapper.cardinality_k == 3
    assert wrapper.min_selected_features == 2
    assert wrapper.max_selected_features == 8
    assert wrapper.solver == solver
    quadratic, linear = wrapper._build_qubo(np.zeros(10), np.zeros((10, 10)))
    assert np.isfinite(quadratic).all()
    assert np.isfinite(linear).all()
    torch.testing.assert_close(wrapper(torch.ones(2, 10)), torch.ones(2, 10))


@pytest.mark.parametrize(
    "feature_dim, options, minimum, maximum",
    [
        (1, {}, 1, 1),
        (10, {}, 2, 10),
        (10, {"max_selected_features": 1}, 1, 1),
        (10, {"max_selected_features": 0}, 0, 0),
        (10, {"cardinality_k": 0}, 0, 10),
        (100, {"cardinality_k": 1}, 1, 100),
        (10, {"cardinality_k": 0, "min_selected_features": 0}, 0, 10),
        (10, {"cardinality_k": 10}, 2, 10),
    ],
)
def test_default_bounds_respect_feasible_cardinality_targets(
    feature_dim, options, minimum, maximum
):
    """Implicit defaults must allow both empty and small cardinality targets."""
    wrapper = FeatureSelectionWrapper(nn.Identity(), feature_dim, **options)
    assert wrapper.min_selected_features == minimum
    assert wrapper.max_selected_features == maximum


@pytest.mark.parametrize(
    "options",
    [
        {"cardinality_k": 0},
        {"cardinality_k": 0, "max_selected_features": 0},
        {"cardinality_k": 2, "min_selected_features": 0, "lambda_reg": 1000},
    ],
)
def test_real_local_search_can_select_no_features_and_keeps_cardinality_soft(options):
    """A zero target permits an empty mask, while a positive target remains soft."""
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    wrapper = FeatureSelectionWrapper(model, 4, solver="local_search", **options)
    inputs = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    targets = torch.zeros(3, 1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    selected = wrapper.update_mask(loader, nn.MSELoss())

    np.testing.assert_array_equal(selected, np.zeros(4, dtype=int))
    assert wrapper.num_selected() == 0
    assert not wrapper.get_support().any()
    torch.testing.assert_close(wrapper(inputs), targets)
