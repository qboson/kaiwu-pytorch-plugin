"""Contract test for ``UnsupervisedDBN.transform``."""

from __future__ import annotations

import numpy as np

from kaiwu.torch_plugin.dbn import UnsupervisedDBN


class _FakeDBN:
    def __init__(self) -> None:
        self.calls = []

    def forward(self, data_in):
        self.calls.append(data_in)
        return "features"


def test_transform_delegates_to_forward_and_returns_its_result():
    model = _FakeDBN()
    data = np.zeros((4, 3), dtype=np.float32)

    result = UnsupervisedDBN.transform(model, data)

    assert result == "features"
    assert len(model.calls) == 1
    assert model.calls[0] is data
