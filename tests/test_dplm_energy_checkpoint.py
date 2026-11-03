"""Pin compact energy-checkpoint contents and actual save/load round trips."""

import importlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("omegaconf")
runtime = importlib.import_module("example.qdiffusion.dplm.utils.runtime")


def make_generator():
    """Use real Torch modules with an independently owned proposal network."""
    energy = torch.nn.Module()
    energy.encoder = torch.nn.Module()
    energy.encoder.backbone = torch.nn.Sequential(
        torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2),
    )
    energy.feature_projector = torch.nn.Linear(2, 2)
    energy.energy_bm = torch.nn.Linear(2, 1)
    energy.sampler_type = "sa"
    energy.sampler_kwargs = {"rand_seed": 42, "size_limit": 10}
    energy.eval()
    return SimpleNamespace(energy_model=energy, proposal_model=torch.nn.Linear(3, 7))


def score(generator, values):
    """Execute each persisted energy-side component without any remote solver."""
    energy = generator.energy_model
    return energy.energy_bm(energy.feature_projector(energy.encoder.backbone(values)))


def test_checkpoint_round_trip_preserves_energy_and_excludes_proposal(tmp_path):
    """Load real bytes into different modules while leaving the proposal untouched."""
    with torch.random.fork_rng():
        torch.manual_seed(314)
        source = make_generator()
        target = make_generator()
    with torch.no_grad():
        source.energy_model.encoder.backbone[1].running_mean.fill_(2)
        source.energy_model.encoder.backbone[1].running_var.fill_(3)
        source.energy_model.encoder.backbone[1].num_batches_tracked.fill_(7)
    for name, value in source.energy_model.named_buffers():
        assert not torch.equal(value, dict(target.energy_model.named_buffers())[name])
    values = torch.tensor([[1., 2., 3.], [-1., 0., 1.]])
    expected = score(source, values).detach()
    assert not torch.allclose(score(target, values), expected)
    proposal_before = {key: val.clone() for key, val in target.proposal_model.state_dict().items()}
    path = runtime.save_checkpoint(
        tmp_path / "nested", "best.pt", generator=source, epoch=3, metric=0.25,
    )
    payload = torch.load(path, weights_only=True)
    assert {"epoch", "metric", "metadata", "state_dict"} <= set(payload)
    assert {"energy_encoder", "feature_projector", "energy_bm"} <= set(payload["state_dict"])
    assert "proposal" not in payload["state_dict"]
    assert "proposal_model" not in payload["state_dict"]
    assert payload["epoch"] == 3
    assert payload["metric"] == 0.25
    metadata = runtime.load_trained_energy_weights(target, str(path), "cpu")
    assert metadata == {
        "bm_sampler_type": "sa",
        "bm_sampler_kwargs": {"rand_seed": 42, "size_limit": 10},
    }
    torch.testing.assert_close(score(target, values), expected, rtol=0, atol=0)
    for key, value in target.proposal_model.state_dict().items():
        torch.testing.assert_close(value, proposal_before[key], rtol=0, atol=0)
    for key, value in source.energy_model.state_dict().items():
        torch.testing.assert_close(value, target.energy_model.state_dict()[key], rtol=0, atol=0)


def test_saved_checkpoint_is_not_changed_by_later_training(tmp_path):
    """The on-disk best snapshot survives subsequent in-place optimizer changes."""
    with torch.random.fork_rng():
        source = make_generator()
        restored = make_generator()
    expected = {key: val.clone() for key, val in source.energy_model.state_dict().items()}
    path = runtime.save_checkpoint(tmp_path, "best.pt", generator=source, epoch=1, metric=1.0)
    with torch.no_grad():
        for parameter in source.energy_model.parameters():
            parameter.add_(10)
        for buffer in source.energy_model.buffers():
            buffer.add_(10)
    runtime.load_trained_energy_weights(restored, str(path), "cpu")
    for key, value in restored.energy_model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
