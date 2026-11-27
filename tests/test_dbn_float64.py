from __future__ import annotations

import os
import sys

import numpy as np

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

from kaiwu.torch_plugin.dbn import UnsupervisedDBN


def _built(hidden: int = 10, dim: int = 4) -> UnsupervisedDBN:
    dbn = UnsupervisedDBN(hidden_layers_structure=[hidden])
    dbn.create_rbm_layer(dim).mark_as_trained()
    return dbn


def test_double_model_accepts_float64_input() -> None:
    """A .double() DBN must transform float64 input without a dtype crash."""
    dbn = _built().double()

    out = dbn.forward(np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64))

    assert out.dtype == np.float64
    assert out.shape == (1, 10)


def test_double_model_reconstructs_any_input_dtype() -> None:
    """A .double() DBN must reconstruct numpy input of either dtype."""
    dbn = _built().double()

    for value in (np.float64, np.float32):
        visible, errors = dbn.reconstruct(
            np.array([[0.1, 0.2, 0.3, 0.4]], dtype=value), layer_index=0
        )

        assert visible.dtype == np.float64
        assert errors.dtype == np.float64


def test_float32_model_paths_unchanged() -> None:
    """The default float32 pipeline keeps its dtype and shapes."""
    dbn = _built()
    x = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    out = dbn.forward(x)
    visible, errors = dbn.reconstruct(x, layer_index=0)

    assert out.dtype == np.float32
    assert visible.dtype == np.float32
    assert errors.dtype == np.float32
