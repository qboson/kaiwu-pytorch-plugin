"""Exact conditional-learning gradients through the real BM generation example."""
from itertools import product
from pathlib import Path
from functools import partial
import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
SOURCE = Path(__file__).resolve().parents[1] / "src"
EXAMPLE = SOURCE.parent / "example" / "bm_generation"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(EXAMPLE))
import kaiwu

kaiwu.__path__.insert(0, str(SOURCE / "kaiwu"))
import saver as generation_saver
import data_loader as generation_data

for module, filename in (
    (generation_saver, "saver.py"),
    (generation_data, "data_loader.py"),
):
    assert Path(module.__file__).resolve() == (EXAMPLE / filename).resolve()


@pytest.fixture
def generation_trainer(monkeypatch):
    """Load after collection, which reloads Kaiwu in existing tests."""
    import trainer
    from kaiwu.torch_plugin import BoltzmannMachine

    assert Path(trainer.__file__).resolve() == (EXAMPLE / "trainer.py").resolve()
    assert trainer.BoltzmannMachine is BoltzmannMachine
    assert Path(sys.modules[BoltzmannMachine.__module__].__file__).resolve() == (
        SOURCE / "kaiwu" / "torch_plugin" / "full_boltzmann_machine.py"
    ).resolve()
    # Use the notebook's real spawn Pool on every platform. Its initializer
    # verifies the selected class before the BM task arguments are deserialized.
    spawn_pool = trainer.mp.get_context("spawn").Pool
    monkeypatch.setattr(trainer.mp, "Pool", partial(spawn_pool, initializer=initialize_worker_source))
    return trainer


def initialize_worker_source():
    """Select the same source in the worker before it receives pickled models."""
    sys.path.insert(0, str(SOURCE))
    kaiwu.__path__.insert(0, str(SOURCE / "kaiwu"))
    from kaiwu.torch_plugin import BoltzmannMachine

    assert Path(sys.modules[BoltzmannMachine.__module__].__file__).resolve() == (
        SOURCE / "kaiwu" / "torch_plugin" / "full_boltzmann_machine.py"
    ).resolve()


class UniformEnumeratingSampler:
    """Enumerate exact uniform conditionals only at the tested zero parameters."""

    def __init__(self, record_directory):
        self.record_directory = record_directory

    def solve(self, matrix):
        from kaiwu.torch_plugin import BoltzmannMachine

        source = Path(sys.modules[BoltzmannMachine.__module__].__file__).resolve()
        assert source == (SOURCE / "kaiwu" / "torch_plugin" / "full_boltzmann_machine.py").resolve()
        # Separate per-process logs avoid concurrent writes to the same file.
        record = Path(self.record_directory) / f"calls-{os.getpid()}.txt"
        if not record.exists():
            print("SAMPLER_SOURCE", os.getpid(), source, flush=True)
        with record.open("a", encoding="utf-8") as stream:
            stream.write(f"{len(matrix)}|{source}\n")
        spins = np.array(list(product((-1.0, 1.0), repeat=len(matrix) - 1)))
        return np.column_stack((spins, np.ones(len(spins))))


class RecordingSaver(generation_saver.Saver):
    """Retain real text/checkpoint writes while exposing the reported training loss."""

    def __init__(self, log_path):
        super().__init__(log_path)
        self.losses = []

    def output_loss(self, output_i, kl_div, ncl, func):
        super().output_loss(output_i, kl_div, ncl, func)
        self.losses.append((kl_div, ncl, func))


def exact_likelihood_losses(quadratic, linear, observations, num_output):
    """Independently sum elementary energies over observed/input/latent assignments."""
    states = torch.tensor(list(product((0.0, 1.0), repeat=len(linear))))
    energy = -(states * linear).sum(dim=1)
    for first in range(len(linear)):
        for second in range(first + 1, len(linear)):
            energy = energy - quadratic[first, second] * states[:, first] * states[:, second]
    joint_losses, conditional_losses = [], []
    num_visible = observations.shape[1]
    num_input = num_visible - num_output
    for observation in observations:
        data_matches = (states[:, :num_visible] == observation).all(dim=1)
        input_matches = (states[:, :num_input] == observation[:num_input]).all(dim=1)
        observed_log_partition = torch.logsumexp(-energy[data_matches], 0)
        joint_losses.append(torch.logsumexp(-energy, 0) - observed_log_partition)
        conditional_losses.append(
            torch.logsumexp(-energy[input_matches], 0) - observed_log_partition
        )
    return torch.stack(joint_losses).mean(), torch.stack(conditional_losses).mean()


@pytest.mark.parametrize("num_hidden", [0, 1])
@pytest.mark.parametrize("alpha", [0.0, 0.35, 1.0])
@pytest.mark.parametrize("num_output", [0, 1])
def test_public_training_matches_exact_likelihood_gradient(
    tmp_path, generation_trainer, num_hidden, alpha, num_output
):
    """Real multiprocess training must retain every conditional sufficient statistic."""
    # All files belong to this fresh pytest directory; no deletion is performed.
    checkpoint_directory = tmp_path / "checkpoints"
    checkpoint_directory.mkdir()
    record_directory = tmp_path / "sampler-records"
    record_directory.mkdir()
    persistence = RecordingSaver(str(tmp_path / "logs"))
    observations = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    dataset = generation_data.CSVDataset(observations)
    assert len(dataset) == 2
    assert dataset[0].dtype == torch.float32
    np.testing.assert_array_equal(dataset[0].numpy(), observations[0])
    sampler = UniformEnumeratingSampler(str(record_directory))
    model_trainer = generation_trainer.Trainer(
        DataLoader(dataset, batch_size=2, shuffle=False), persistence, sampler,
        num_visible=2, num_hidden=num_hidden, num_output=num_output,
    )
    with torch.no_grad():
        model_trainer.bm_net.quadratic_coef.zero_()
        model_trainer.bm_net.linear_bias.zero_()
    model_trainer.set_cost_parameter(alpha=alpha, beta=1.0)
    model_trainer.set_learning_parameters(0.2, 0.0, 0.0)
    quadratic = model_trainer.bm_net.quadratic_coef.detach().clone().requires_grad_()
    linear = model_trainer.bm_net.linear_bias.detach().clone().requires_grad_()
    joint_loss, conditional_loss = exact_likelihood_losses(
        quadratic, linear, observations, num_output
    )
    expected_quadratic, expected_linear = torch.autograd.grad(
        alpha * joint_loss + (1 - alpha) * conditional_loss, (quadratic, linear)
    )

    model_trainer.train(max_steps=1, save_path=str(checkpoint_directory), num_processes=2)

    # The enumerated sampler is exact at initial zero parameters, when these
    # gradients were computed. It does not claim Boltzmann sampling after the update.
    torch.testing.assert_close(model_trainer.bm_net.quadratic_coef.grad, expected_quadratic)
    torch.testing.assert_close(model_trainer.bm_net.linear_bias.grad, expected_linear)
    joint_after, conditional_after = exact_likelihood_losses(
        model_trainer.bm_net.quadratic_coef,
        model_trainer.bm_net.linear_bias, observations, num_output,
    )
    if num_output:
        assert conditional_after.item() < conditional_loss.item()
    elif alpha > 0:
        assert joint_after.item() < joint_loss.item()
    else:
        # A zero conditional term must still permit backward through the KL mixture.
        assert torch.count_nonzero(model_trainer.bm_net.quadratic_coef.grad) == 0
        assert torch.count_nonzero(model_trainer.bm_net.linear_bias.grad) == 0
        assert torch.count_nonzero(model_trainer.bm_net.quadratic_coef) == 0
        assert torch.count_nonzero(model_trainer.bm_net.linear_bias) == 0
    if num_output == 0:
        assert persistence.losses[0][1] == 0.0
    records = [line.split("|", 1) for record in record_directory.glob("calls-*.txt")
               for line in record.read_text(encoding="utf-8").splitlines()]
    calls = [int(size) for size, _ in records]
    assert all(Path(source) == (SOURCE / "kaiwu" / "torch_plugin" / "full_boltzmann_machine.py")
               for _, source in records)
    assert len(calls) == (5 if num_output else 3)
    assert calls.count(1 + num_hidden) == 2
    assert len(list(record_directory.glob("calls-*.txt"))) >= 2

    # The actual persistence format is the format consumed by sample_bm.ipynb.
    assert (checkpoint_directory / "rbm_model0.pth").is_file()
    model_trainer.saver.save_info(model_trainer.bm_net, str(checkpoint_directory), 1, 0.0)
    restored = torch.load(checkpoint_directory / "rbm_model1.pth", weights_only=False)
    for name, parameter in model_trainer.bm_net.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], parameter)
    assert (tmp_path / "logs" / "loss.txt").is_file()
