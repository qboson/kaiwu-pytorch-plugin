"""UnsupervisedDBN.transform must accept array-like input and validate its width."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import UnsupervisedDBN


def _trained() -> UnsupervisedDBN:
    return UnsupervisedDBN([3, 2]).create_rbm_layer(4).mark_as_trained()


def test_numpy_input_still_transforms() -> None:
    out = _trained().transform(np.zeros((2, 4), dtype=np.float32))
    assert out.shape == (2, 2)


def test_list_input_transforms() -> None:
    out = _trained().transform([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    assert out.shape == (2, 2)


def test_torch_input_transforms() -> None:
    out = _trained().transform(torch.zeros(2, 4))
    assert out.shape == (2, 2)


def test_wrong_width_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"data_in must have shape \(n_samples, 4\)"):
        _trained().transform(np.zeros((2, 5), dtype=np.float32))


def test_one_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"data_in must have shape \(n_samples, 4\)"):
        _trained().transform(np.zeros(4, dtype=np.float32))
