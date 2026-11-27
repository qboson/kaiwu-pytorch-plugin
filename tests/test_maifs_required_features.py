"""Required MAIFS features affect the reduced objective and remain selected."""

from itertools import product

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper
from kaiwu.torch_plugin.maifs import plugin


def test_required_feature_is_substituted_into_solver_objective(monkeypatch):
    selector = FeatureSelectionWrapper(
        nn.Linear(3, 1), 3, max_selected_features=2, required_feature_indices=(1,)
    )
    quadratic = np.array([[2., 4., 1.], [2., 0., -6.], [3., 2., 1.]])
    linear = np.array([-0.6, 20., 0.1])
    monkeypatch.setattr(selector, "compute_mask_derivatives", lambda *_args: (None, None))
    monkeypatch.setattr(selector, "_build_qubo", lambda *_args: (quadratic, linear))

    def objective(bits):
        return 0.5 * bits @ quadratic @ bits + linear @ bits

    def solver(reduced_quadratic, reduced_linear, mask, name, **kwargs):
        np.testing.assert_array_equal(mask, [1, 1])
        for bits in map(np.asarray, product((0., 1.), repeat=2)):
            full = np.array([bits[0], 1, bits[1]])
            actual = 0.5 * bits @ reduced_quadratic @ bits + reduced_linear @ bits
            assert actual == pytest.approx(objective(full) - objective(np.array([0, 1, 0])))
        return np.ones(2, dtype=int)

    monkeypatch.setattr(plugin, "solve_qubo", solver)
    expected = min((np.array([0, 1, 1]), np.array([1, 1, 0])), key=objective)
    np.testing.assert_array_equal(selector.update_mask([], nn.MSELoss()), expected)
    np.testing.assert_array_equal(selector.get_support(), expected.astype(bool))


def test_all_required_features_skip_solving_and_survive_older_mask():
    model = nn.Linear(2, 1)
    original = FeatureSelectionWrapper(model, 2)
    original.mask.zero_()
    selector = FeatureSelectionWrapper(
        model, 2, cardinality_k=2, required_feature_indices=(0, 1)
    )
    selector.load_state_dict(original.state_dict(), strict=True)
    np.testing.assert_array_equal(selector.get_support(), [True, True])
    np.testing.assert_array_equal(selector.update_mask([], nn.MSELoss()), [1, 1])


@pytest.mark.parametrize(
    "required, options, error",
    [((0, 0), {}, "distinct"), ((-1,), {}, "valid"), ((3,), {}, "valid"),
     ((1.5,), {}, "integers"), ((0, 1), {"cardinality_k": 1}, "feature count"),
     ((0, 1), {"max_selected_features": 1}, "feature count")],
)
def test_required_features_must_fit_configuration(required, options, error):
    with pytest.raises(ValueError, match=error):
        FeatureSelectionWrapper(nn.Linear(3, 1), 3, required_feature_indices=required, **options)
