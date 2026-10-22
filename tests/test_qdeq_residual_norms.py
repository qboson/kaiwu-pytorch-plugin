"""The residual norms must match the stopping rules they feed."""

import pytest
import torch

from kaiwu.torch_plugin.qdeq import _residual_norms


def test_absolute_and_relative_norms_use_the_function_value():
    residual = torch.tensor([3.0, 4.0])
    f_value = torch.tensor([0.0, 10.0])

    abs_res, rel_res = _residual_norms(residual, f_value)

    assert abs_res == 5.0
    assert rel_res == pytest.approx(0.5, rel=1e-9)


def test_zero_function_value_falls_back_to_the_epsilon_floor():
    residual = torch.tensor([3.0, 4.0])

    abs_res, rel_res = _residual_norms(residual, torch.zeros(2))

    assert abs_res == 5.0
    assert rel_res == pytest.approx(5.0 / 1e-9, rel=1e-6)


def test_norms_use_all_elements():
    residual = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

    abs_res, _ = _residual_norms(residual, residual)

    assert abs_res == 5.0
