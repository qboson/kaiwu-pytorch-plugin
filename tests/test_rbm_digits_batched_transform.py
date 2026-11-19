"""The digits pipeline preserves feature order with bounded device batches."""

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("seaborn")

from example.rbm_digits.rbm_digits import RBMRunner
from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def test_transform_batches_on_the_rbm_device_and_keeps_the_tail(monkeypatch):
    runner = RBMRunner(n_components=2, batch_size=3)
    runner.rbm = RestrictedBoltzmannMachine(4, 2, device="cpu").double()
    data = np.linspace(0, 1, 32, dtype=np.float32).reshape(8, 4)
    original = runner.rbm.get_hidden
    expected = original(torch.tensor(data, dtype=torch.float64))[:, 4:].numpy()
    batch_sizes = []

    def record(batch):
        batch_sizes.append(len(batch))
        assert batch.device == runner.rbm.quadratic_coef.device
        assert batch.dtype == runner.rbm.quadratic_coef.dtype
        return original(batch)

    monkeypatch.setattr(runner.rbm, "get_hidden", record)
    np.testing.assert_allclose(runner.transform(data), expected)
    assert batch_sizes == [3, 3, 2]
    assert runner.transform(data[:0]).shape == (0, 2)
