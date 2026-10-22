"""Pin layer selection in UnsupervisedDBN.reconstruct."""

import numpy as np

from kaiwu.torch_plugin import UnsupervisedDBN


def _dbn() -> UnsupervisedDBN:
    return UnsupervisedDBN([3, 2]).create_rbm_layer(4).mark_as_trained()


def test_reconstruct_uses_the_requested_layer_width() -> None:
    visible, errors = _dbn().reconstruct(
        np.zeros((2, 3), dtype=np.float32), layer_index=1
    )

    assert visible.shape == (2, 3)
    assert errors.shape == (2,)


def test_first_layer_reconstructs_the_full_input_width() -> None:
    visible, errors = _dbn().reconstruct(
        np.zeros((2, 4), dtype=np.float32), layer_index=0
    )

    assert visible.shape == (2, 4)
    assert errors.shape == (2,)
