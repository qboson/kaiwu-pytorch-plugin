"""EnergyModel must reject non-positive BM dimensions at construction."""

import pytest

from kaiwu.torch_plugin.qdiffusion import EnergyModel


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bm_num_visible": 0, "bm_num_hidden": 3},
        {"bm_num_visible": -2, "bm_num_hidden": 3},
        {"bm_num_visible": 2, "bm_num_hidden": 0},
        {"bm_num_visible": 2.5},
        {"bm_num_hidden": True},
    ],
)
def test_invalid_dimensions_are_rejected(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        EnergyModel(**kwargs)


def test_a_partial_configuration_is_still_allowed():
    model = EnergyModel(bm_num_visible=2)

    assert model.bm_num_visible == 2
    assert model.bm_num_hidden is None
