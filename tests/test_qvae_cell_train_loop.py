"""Early-stopping/history contract for the qvae_cell training loop."""

import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

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
    "qvae_cell_trainer_loop", os.path.join(REPO_ROOT, "example", "qvae_cell", "trainer.py")
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer


class StubModel:
    """Exposes state dict hooks used by checkpointing."""

    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"w": torch.tensor([1.0])}

    def load_state_dict(self, state):
        self.loaded = {key: value.clone() for key, value in state.items()}


def test_history_best_checkpoint_and_early_stop(tmp_path, capsys):
    """Four epochs record history, restore the best state and stop after patience."""
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        epochs=4,
        disable_tqdm=True,
        train_log_every=1,
        checkpoint_every=0,
        early_stopping_patience=2,
    )
    trainer.metric_keys = ("loss", "kl", "recon_loss", "bm_loss", "neg_elbo")
    val_losses = [1.0, 0.5, 0.6, 0.7]
    calls = {"val": 0}

    def fake_run_epoch(model, loader, vae_optimizer, bm_optimizer, train=True):
        if train:
            return {key: 0.1 for key in trainer.metric_keys}
        loss = val_losses[calls["val"]]
        calls["val"] += 1
        return {
            "loss": loss,
            "kl": 1.0,
            "recon_loss": 2.0,
            "bm_loss": 0.0,
            "neg_elbo": loss,
        }

    trainer.run_epoch = fake_run_epoch
    model = StubModel()
    checkpoint = tmp_path / "best.pth"
    history = trainer.train(
        model, object(), object(), object(), object(), str(checkpoint)
    )

    assert calls["val"] == 4
    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 4
    assert history["val_loss"] == val_losses
    assert checkpoint.is_file()
    assert model.loaded is not None
    assert "Early stopping triggered at epoch 4" in capsys.readouterr().out
