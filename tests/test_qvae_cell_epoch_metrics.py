"""Regression test for the CellQVAE epoch-metric aggregation."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("kaiwu")
for _module in ("sklearn", "tqdm"):
    pytest.importorskip(_module)
kaiwu_preprocess = pytest.importorskip("kaiwu.preprocess")

if not hasattr(kaiwu_preprocess, "PrecisionReducer"):
    # The example imports PrecisionReducer from kaiwu.preprocess, which the
    # pinned SDK does not export (separate, already-reported issue).
    from kaiwu.cim import PrecisionReducer

    kaiwu_preprocess.PrecisionReducer = PrecisionReducer

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/qvae_cell"))
)
from trainer import Trainer  # noqa: E402  pylint: disable=wrong-import-position


class _ConstantLossModel(torch.nn.Module):
    """Model whose per-batch loss equals the batch mean of the input."""

    def __init__(self):
        super().__init__()
        self.last_kl_loss = torch.tensor(0.0)
        self.last_recon_loss = torch.tensor(0.0)

    def forward(self, x, batch_idx):
        return x, None, x, x

    def loss(self, x, output, posterior):
        self.last_kl_loss = torch.tensor(x.mean().item() * 0.1)
        self.last_recon_loss = torch.tensor(x.mean().item() * 0.9)
        return x.mean()

    def bm_loss(self, q, weight_decay):  # pragma: no cover - train path unused
        return q.mean()


def _loader_with_remainder():
    return [
        (torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)),
        (torch.full((4, 2), 0.4), torch.zeros(4, dtype=torch.long)),
        (torch.ones(2, 2), torch.zeros(2, dtype=torch.long)),
    ]


def test_epoch_metrics_are_weighted_by_observations():
    trainer = Trainer(SimpleNamespace(bm_weight_decay=0.0), torch.device("cpu"))

    metrics = trainer.run_epoch(
        _ConstantLossModel(), _loader_with_remainder(), None, None, train=False
    )

    expected = (0.0 * 4 + 0.4 * 4 + 1.0 * 2) / 10
    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["kl"] == pytest.approx(expected * 0.1)
    assert metrics["recon_loss"] == pytest.approx(expected * 0.9)
