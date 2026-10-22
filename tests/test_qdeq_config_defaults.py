"""QDEQConfig must keep its documented defaults."""

import pytest

from kaiwu.torch_plugin import QDEQConfig


def test_documented_defaults_are_unchanged():
    config = QDEQConfig()

    assert config.f_thres == 12
    assert config.b_thres == 25
    assert config.f_eps == pytest.approx(1e-3)
    assert config.b_eps == pytest.approx(1e-3)
    assert config.stop_mode == "rel"
    assert config.num_basis == 8
    assert config.coeff_bits == 8
    assert config.coeff_range == pytest.approx(2.0)
    assert config.fd_eps == pytest.approx(1e-3)
    assert config.history_size == 4
    assert config.anderson_m == 6
    assert config.anderson_lam == pytest.approx(1e-4)
    assert config.jac_loss_vecs == 0
