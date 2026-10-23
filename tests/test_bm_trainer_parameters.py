"""Parameter-sync contract for the BM generation trainer."""

import os
from types import SimpleNamespace

import importlib.util
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "bm_generation_trainer_params",
    os.path.join(REPO_ROOT, "example", "bm_generation", "trainer.py"),
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer


class ForbiddenCpuCopy:
    """Fails if the CPU parameter iterator is touched."""

    def parameters(self):
        raise AssertionError("cpu mirror must not be touched on a CPU-only run")


def test_learning_parameters_are_recorded():
    """The three hyperparameters are stored under their documented keys."""
    trainer = Trainer.__new__(Trainer)
    trainer.set_learning_parameters(0.01, 0.02, 0.9)
    assert trainer.learning_parameters == {
        "learning_rate": 0.01,
        "weight_decay_rate": 0.02,
        "momentum_rate": 0.9,
    }


def test_cpu_model_skips_the_gpu_to_cpu_copy():
    """A CPU-resident model returns before touching the mirror parameters."""
    trainer = Trainer.__new__(Trainer)
    trainer.bm_net = SimpleNamespace(device=torch.device("cpu"))
    trainer.bm_net_cpu = ForbiddenCpuCopy()
    assert trainer._sync_gpu_to_cpu() is None


def test_gpu_model_copies_weights_into_the_mirror():
    """A non-CPU model copies every parameter into the CPU mirror."""
    trainer = Trainer.__new__(Trainer)
    trainer.bm_net = SimpleNamespace(
        device=torch.device("meta"), parameters=lambda: iter([torch.ones(2)])
    )
    mirror = torch.zeros(2)
    trainer.bm_net_cpu = SimpleNamespace(parameters=lambda: iter([mirror]))
    trainer._sync_gpu_to_cpu()
    assert mirror.tolist() == [1.0, 1.0]
