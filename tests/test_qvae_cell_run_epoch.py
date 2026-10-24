"""Epoch-aggregation contract for the qvae_cell trainer."""

import importlib
import importlib.util
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_cell"))

for _name in [m for m in list(sys.modules) if m == "models" or m.startswith("models.")]:
    del sys.modules[_name]
pre = importlib.import_module("kaiwu.preprocess")
if not hasattr(pre, "PrecisionReducer"):
    pre.PrecisionReducer = type(
        "PrecisionReducer", (), {"__init__": lambda self, *args, **kwargs: None}
    )

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_trainer_epoch", os.path.join(REPO_ROOT, "example", "qvae_cell", "trainer.py")
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer


class StubModel:
    """Returns fixed losses for every batch."""

    def __init__(self):
        self.last_kl_loss = torch.tensor(1.0)
        self.last_recon_loss = torch.tensor(2.0)
        self.mode = None

    def train(self, mode=True):
        self.mode = mode
        return self

    def __call__(self, x, batch_idx):
        return ("OUTPUT", "POSTERIOR", "Q", "ZETA")

    def loss(self, x, output, posterior):
        return torch.tensor(3.0)

    def bm_loss(self, q, weight_decay):
        raise AssertionError("BM loss must not run in evaluation mode")


class ForbiddenOptimizer:
    """Fails if touched during evaluation."""

    def zero_grad(self, *args, **kwargs):
        raise AssertionError("evaluation must not touch optimizers")

    def step(self, *args, **kwargs):
        raise AssertionError("evaluation must not touch optimizers")


def build_trainer():
    """Builds only the attributes ``run_epoch`` reads."""
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.args = type("Args", (), {"bm_weight_decay": 0.0})()
    trainer.metric_keys = ("loss", "neg_elbo", "kl", "recon_loss", "bm_loss")
    return trainer


def test_evaluation_averages_metrics_and_skips_the_bm_phase():
    """Metrics are averaged over batches with a zero BM loss and no optimizer calls."""
    loader = [
        (torch.zeros(2, 3), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(2, 3), torch.zeros(2, dtype=torch.long)),
    ]
    metrics = build_trainer().run_epoch(
        StubModel(), loader, ForbiddenOptimizer(), ForbiddenOptimizer(), train=False
    )
    assert metrics == {
        "loss": 3.0,
        "neg_elbo": 3.0,
        "kl": 1.0,
        "recon_loss": 2.0,
        "bm_loss": 0.0,
    }
