from __future__ import annotations

import os
import sys

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

from kaiwu.torch_plugin.dbn import UnsupervisedDBN


def test_unbuilt_accessors_fall_back_instead_of_crashing() -> None:
    """未 build 的 DBN 访问层信息应回退，不应抛 TypeError。"""
    dbn = UnsupervisedDBN(hidden_layers_structure=[10])

    assert dbn.get_rbm_layer(0) is None
    assert dbn.num_layers == 0
    assert dbn.output_dim is None


def test_forward_guard_unchanged() -> None:
    """forward 的既有明确错误保持不变。"""
    import pytest

    dbn = UnsupervisedDBN(hidden_layers_structure=[10])
    with pytest.raises(ValueError, match="Model not built"):
        dbn.forward([[1.0, 2.0]])


def test_built_accessors_unchanged() -> None:
    """build 后的正常路径行为零变化。"""
    import numpy as np

    dbn = UnsupervisedDBN(hidden_layers_structure=[10])
    dbn.create_rbm_layer(4).mark_as_trained()

    assert dbn.num_layers == 1
    assert dbn.output_dim == 10
    assert dbn.get_rbm_layer(0) is dbn.rbm_layers[0]
    assert dbn.get_rbm_layer(5) is None

    out = dbn.forward(np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64))
    assert out.shape == (1, 10)
