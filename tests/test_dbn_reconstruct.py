"""Contract test for ``UnsupervisedDBN.reconstruct`` layer selection."""

from __future__ import annotations

import pytest

from kaiwu.torch_plugin.dbn import UnsupervisedDBN


class _FakeDBN:
    def __init__(self, layers) -> None:
        self.rbm_layers = layers
        self.device = "cpu"
        self.calls = []

    def reconstruct_with_rbm(self, rbm, data_in, device=None):
        self.calls.append((rbm, data_in, device))
        return "reconstructed"


def test_requires_at_least_one_fitted_layer():
    with pytest.raises(ValueError, match="No RBM layers"):
        UnsupervisedDBN.reconstruct(_FakeDBN(None), "data")
    with pytest.raises(ValueError, match="No RBM layers"):
        UnsupervisedDBN.reconstruct(_FakeDBN([]), "data")


def test_rejects_an_out_of_range_layer_index():
    model = _FakeDBN(["first"])

    with pytest.raises(ValueError, match="out of range"):
        UnsupervisedDBN.reconstruct(model, "data", layer_index=1)


def test_delegates_to_the_selected_layer_and_device():
    model = _FakeDBN(["first", "second"])

    result = UnsupervisedDBN.reconstruct(model, "data", layer_index=1)

    assert result == "reconstructed"
    assert model.calls == [("second", "data", "cpu")]
