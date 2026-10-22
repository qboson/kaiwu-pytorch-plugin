"""QDEQ must reject solver configurations that cannot run."""

import pytest
import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"f_thres": 0},
        {"f_thres": 1.5},
        {"f_thres": True},
        {"b_thres": 0},
        {"num_basis": 0},
        {"coeff_bits": 0},
        {"anderson_m": 0},
        {"history_size": 0},
        {"jac_loss_vecs": -1},
        {"f_eps": 0.0},
        {"b_eps": float("nan")},
        {"coeff_range": 0.0},
        {"fd_eps": 0.0},
        {"anderson_lam": -1.0},
    ],
)
def test_invalid_config_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        QDEQ(torch.nn.Identity(), None, QDEQConfig(**kwargs))


def test_default_config_constructs() -> None:
    model = QDEQ(torch.nn.Identity(), None)
    assert model.config.num_basis >= 1
