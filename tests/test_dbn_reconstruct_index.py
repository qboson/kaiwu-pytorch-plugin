"""UnsupervisedDBN.reconstruct must reject negative layer indices."""

import numpy as np
import pytest

from kaiwu.torch_plugin import UnsupervisedDBN


def _trained_dbn() -> UnsupervisedDBN:
    return UnsupervisedDBN([3, 2]).create_rbm_layer(4).mark_as_trained()


@pytest.mark.parametrize("index", [-1, -2])
def test_negative_layer_index_is_rejected(index) -> None:
    with pytest.raises(ValueError, match=f"Layer index {index} out of range"):
        _trained_dbn().reconstruct(np.zeros((2, 4), dtype=np.float32), layer_index=index)


def test_too_large_layer_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="Layer index 2 out of range"):
        _trained_dbn().reconstruct(np.zeros((2, 4), dtype=np.float32), layer_index=2)


def test_first_layer_still_reconstructs() -> None:
    visible, errors = _trained_dbn().reconstruct(
        np.zeros((2, 4), dtype=np.float32), layer_index=0
    )
    assert visible.shape == (2, 4)
    assert errors.shape == (2,)
