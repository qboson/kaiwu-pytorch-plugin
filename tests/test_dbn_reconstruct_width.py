"""UnsupervisedDBN.reconstruct_with_rbm must validate the input width."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine, UnsupervisedDBN


def _rbm() -> RestrictedBoltzmannMachine:
    return RestrictedBoltzmannMachine(num_visible=4, num_hidden=3)


def test_wrong_width_input_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"data_in must have shape \(batch, 4\)"):
        UnsupervisedDBN.reconstruct_with_rbm(_rbm(), np.zeros((2, 5), dtype=np.float32))


def test_one_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"data_in must have shape \(batch, 4\)"):
        UnsupervisedDBN.reconstruct_with_rbm(_rbm(), np.zeros(4, dtype=np.float32))


def test_matching_input_still_reconstructs() -> None:
    visible, errors = UnsupervisedDBN.reconstruct_with_rbm(
        _rbm(), np.zeros((2, 4), dtype=np.float32)
    )
    assert visible.shape == (2, 4)
    assert errors.shape == (2,)

    visible_tensor, _ = UnsupervisedDBN.reconstruct_with_rbm(
        _rbm(), torch.zeros(2, 4)
    )
    assert visible_tensor.shape == (2, 4)
