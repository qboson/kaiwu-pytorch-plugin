"""Training and optional diagnostics must keep independent run state."""

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

for dependency in ("matplotlib", "sklearn", "pandas", "seaborn", "torchvision",
                   "gif", "imageio", "torchmetrics"):
    pytest.importorskip(dependency)

import matplotlib
import kaiwu

matplotlib.use("Agg")


@pytest.fixture
def trainer_module(monkeypatch):
    """Import the real example, whose imports expect its directory on sys.path."""
    example = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
    monkeypatch.syspath_prepend(str(example))
    src_root = example.parents[1] / "src"
    monkeypatch.setattr(kaiwu, "__path__", [str(src_root / "kaiwu")] + list(kaiwu.__path__))
    # Restore both an installed plugin and its package attribute after this test.
    monkeypatch.setattr(kaiwu, "torch_plugin", None, raising=False)
    monkeypatch.delattr(kaiwu, "torch_plugin")
    # These generic package names may also belong to another example test.
    prefixes = ("model", "utils", "trainer", "kaiwu.torch_plugin")

    def belongs_to_example(name):
        return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)

    previous = {name: value for name, value in sys.modules.items()
                if belongs_to_example(name)}
    for name in previous:
        del sys.modules[name]
    try:
        yield importlib.import_module("trainer.trainer")
    finally:
        for name in list(sys.modules):
            if belongs_to_example(name):
                del sys.modules[name]
        sys.modules.update(previous)


class OfflineSolver:
    """Enumerate the tiny Ising system at the external SDK solve boundary."""

    def solve(self, matrix):
        states = np.array([
            [1 if state >> bit & 1 else -1 for bit in range(len(matrix))]
            for state in range(2 ** len(matrix))
        ])
        energies = np.einsum("bi,ij,bj->b", states, matrix, states)
        return states[np.argsort(energies)[:8]]


@pytest.fixture
def make_trainer(trainer_module, monkeypatch, tmp_path):
    monkeypatch.setattr(trainer_module.QVAE, "_create_sampler", lambda self, *args: OfflineSolver())
    # Keep actual training, posterior sampling, t-SNE and frame rendering;
    # reconstruction plots and GIF encoding are unrelated to state isolation.
    monkeypatch.setattr(trainer_module.Trainer, "_save_reconstruction", lambda *args: None)
    monkeypatch.setattr(trainer_module.Trainer, "_plot_training_curve", lambda *args: None)
    monkeypatch.setattr(trainer_module, "create_tsne_animation", lambda *args, **kwargs: None)

    def make(name):
        config = trainer_module.Config(
            num_latent_units=4, encoder_hidden_nodes=[6], decoder_hidden_nodes=[6],
            batch_size=32, num_epochs=10, output_dir=str(tmp_path / name),
        )
        data = np.linspace(0.01, 0.99, 128 * 8, dtype=np.float32).reshape(128, 8)
        labels = np.arange(128) % 2
        return trainer_module.Trainer(config, (data[:64], labels[:64]), (data[64:], labels[64:]))

    return make


def test_animation_preserves_training_losses_parameters_and_rng(make_trainer, trainer_module, monkeypatch):
    recorded_frames = []

    def record_animation(frames, **kwargs):
        assert [Path(frame).name for frame in frames] == ["tsne_epoch_005.png", "tsne_epoch_010.png"]
        for frame in frames:
            assert Path(frame).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        recorded_frames.extend(frames)

    monkeypatch.setattr(trainer_module, "create_tsne_animation", record_animation)

    def run(animation):
        torch.manual_seed(812)
        trainer = make_trainer(str(animation))
        model, train_losses, test_losses = trainer.train(
            generate_animation=animation, tsne_interval=5,
        )
        parameters = torch.cat([value.detach().flatten() for value in model.parameters()])
        return train_losses, test_losses, parameters, torch.get_rng_state()

    baseline = run(False)
    animated = run(True)
    assert len(recorded_frames) == 2
    assert baseline[:2] == animated[:2]
    assert torch.equal(baseline[2], animated[2])
    assert torch.equal(baseline[3], animated[3])


def test_new_training_run_keeps_previous_returned_histories(make_trainer):
    trainer = make_trainer("histories")
    first_model, first_train, first_test = trainer.train()
    train_snapshot, test_snapshot = first_train.copy(), first_test.copy()

    second_model, second_train, second_test = trainer.train()
    trainer.save_results()
    saved = json.loads((Path(trainer.output_dir) / "results.json").read_text())

    assert second_model is not first_model
    assert first_train == train_snapshot
    assert first_test == test_snapshot
    assert second_train is not first_train
    assert second_test is not first_test
    assert len(second_train) == len(second_test) == trainer.config.num_epochs
    assert saved["train_losses"] == second_train
    assert saved["test_losses"] == second_test


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
)])
def test_frame_restores_rng_even_when_visualization_fails(trainer_module, tmp_path, monkeypatch, device):
    trainer = trainer_module.Trainer(trainer_module.Config(output_dir=str(tmp_path)))
    trainer.model = torch.nn.Linear(2, 2).to(device)

    def fail_after_sampling(**kwargs):
        torch.rand(5)
        torch.rand(5, device=device)
        raise RuntimeError("plot failed")

    monkeypatch.setattr(trainer_module, "t_SNE", fail_after_sampling)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(trainer.model.weight.device) if device == "cuda" else None
    with pytest.raises(RuntimeError, match="plot failed"):
        trainer._save_tsne_frame(1)

    assert torch.equal(torch.get_rng_state(), cpu_state)
    if cuda_state is not None:
        assert torch.equal(torch.cuda.get_rng_state(trainer.model.weight.device), cuda_state)
