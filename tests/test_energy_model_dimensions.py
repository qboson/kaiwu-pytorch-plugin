# -*- coding: utf-8 -*-
"""EnergyModel requires paired, positive Boltzmann-machine dimensions."""

from __future__ import annotations

import pytest

from kaiwu.torch_plugin.qdiffusion import EnergyModel


@pytest.mark.parametrize(
    "visible,hidden",
    [(2, None), (None, 2), (-1, 2), (2, 0), (2, 2.0), (True, 2)],
)
def test_invalid_bm_dimensions_are_rejected(visible, hidden) -> None:
    with pytest.raises(ValueError):
        EnergyModel(bm_num_visible=visible, bm_num_hidden=hidden)


def test_valid_dimensions_build_a_boltzmann_machine() -> None:
    model = EnergyModel(bm_num_visible=2, bm_num_hidden=3)

    assert model.bm_num_visible == 2
    assert model.bm_num_hidden == 3
    assert model.energy_bm.num_nodes == 5


def test_no_bm_configuration_remains_valid() -> None:
    model = EnergyModel()

    assert model.bm_num_visible is None
    assert model.bm_num_hidden is None
    assert not hasattr(model, "energy_bm")
