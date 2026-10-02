"""Independent references and resource controls for cardinality projection."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from torch import nn

from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper


def reference(wrapper, candidate, quadratic, linear):
    mask = candidate.copy()

    def value(index, change):
        changed = mask.copy()
        changed[index] += change
        return float(0.5 * np.einsum("i,ij,j->", changed, quadratic, changed)
                     + np.einsum("i,i->", linear, changed)), int(index)

    count = int(mask.sum())
    while count > wrapper.max_selected_features:
        index = min(np.flatnonzero(mask), key=lambda index: value(index, -1))
        mask[index] = 0
        count -= 1
    enforce_minimum = wrapper._min_selected_features_explicit or count == 0
    while enforce_minimum and count < wrapper.min_selected_features:
        index = min(np.flatnonzero(mask == 0), key=lambda index: value(index, 1))
        mask[index] = 1
        count += 1
    return mask


@pytest.mark.parametrize("size", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("seed", range(20))
def test_projection_matches_independent_full_energy(size, seed):
    rng = np.random.default_rng(seed)
    wrapper = FeatureSelectionWrapper(nn.Identity(), size, min_selected_features=1,
                                      max_selected_features=max(1, size // 2))
    for integer in (True, False):
        quadratic = (rng.integers(-10, 11, (size, size)).astype(float) if integer
                     else rng.normal(size=(size, size)))
        linear = (rng.integers(-10, 11, size).astype(float) if integer
                  else rng.normal(size=size))
        for candidate in [np.ones(size, dtype=int), np.zeros(size, dtype=int),
                          rng.integers(0, 2, size)]:
            saved = [array.copy() for array in (candidate, quadratic, linear)]
            actual = wrapper._project_selected_feature_count(candidate, quadratic, linear)
            np.testing.assert_array_equal(actual, reference(wrapper, candidate, quadratic, linear))
            for array, original in zip((candidate, quadratic, linear), saved):
                np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("initial,expected", [
    ([1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]),
    ([0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]),
])
def test_exact_ties_choose_lowest_index(initial, expected):
    wrapper = FeatureSelectionWrapper(nn.Identity(), 6, min_selected_features=3,
                                      max_selected_features=3)
    np.testing.assert_array_equal(wrapper._project_selected_feature_count(
        np.array(initial), np.zeros((6, 6)), np.zeros(6)), expected)


def test_implicit_minimum_only_applies_to_empty_mask():
    wrapper = FeatureSelectionWrapper(nn.Identity(), 10)
    for count, expected in [(0, 2), (1, 1), (2, 2)]:
        candidate = np.r_[np.ones(count), np.zeros(10 - count)]
        actual = wrapper._project_selected_feature_count(candidate, np.eye(10), np.zeros(10))
        assert actual.sum() == expected


def test_fixed_large_energy_does_not_hide_the_better_addition():
    wrapper = FeatureSelectionWrapper(nn.Identity(), 3, min_selected_features=2)
    quadratic = np.diag([2e20, 0., 0.])
    linear = np.array([0., 2., 1.])
    np.testing.assert_array_equal(wrapper._project_selected_feature_count(
        np.array([1, 0, 0]), quadratic, linear), [1, 0, 1])


@pytest.mark.parametrize("scale", [1., 1e8, 1e16])
def test_cancelling_row_preserves_small_gain(scale):
    wrapper = FeatureSelectionWrapper(nn.Identity(), 5, min_selected_features=3)
    quadratic = np.zeros((5, 5))
    quadratic[2, :2] = [scale, -scale]
    quadratic[:2, 2] = [scale, -scale]
    linear = np.array([0., 0., 1., 2., 3.])
    candidate = np.array([1, 1, 0, 0, 0])
    actual = wrapper._project_selected_feature_count(candidate, quadratic, linear)
    with localcontext() as ctx:
        ctx.prec = 2048
        def energy(mask):
            return sum((Decimal.from_float(float(quadratic[i, j])) * int(mask[i]) * int(mask[j]) / 2
                        for i in range(5) for j in range(5)), Decimal(0)) + sum(
                            Decimal.from_float(float(linear[i])) * int(mask[i]) for i in range(5))
        chosen = energy(actual)
        for index in [2, 3, 4]:
            alternative = candidate.copy()
            alternative[index] = 1
            assert chosen <= energy(alternative)
    np.testing.assert_array_equal(actual, [1, 1, 1, 0, 0])


def test_projection_avoids_per_candidate_matrix_products():
    calls = []
    class TracedMatrix(np.ndarray):
        __array_priority__ = 1000
        def __rmatmul__(self, other):
            calls.append(np.shape(other))
            return np.asarray(other) @ np.asarray(self)
    quadratic = np.eye(16).view(TracedMatrix)
    wrapper = FeatureSelectionWrapper(nn.Identity(), 16, max_selected_features=8)
    result = wrapper._project_selected_feature_count(np.ones(16), quadratic, np.zeros(16))
    assert result.sum() == 8
    assert len(calls) <= 1


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e308])
def test_unsafe_projection_coefficients_are_rejected(value):
    wrapper = FeatureSelectionWrapper(nn.Identity(), 2, max_selected_features=1)
    with pytest.raises(ValueError, match="finite|safe float64"):
        wrapper._project_selected_feature_count(np.ones(2), np.eye(2), np.array([value, 0.]))
