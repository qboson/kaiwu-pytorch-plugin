"""A failed compact checkpoint write must leave the published file intact."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from example.qdiffusion.dplm.utils.runtime import load_trained_energy_weights, save_checkpoint


def make_generator():
    return SimpleNamespace(energy_model=SimpleNamespace(
        encoder=SimpleNamespace(backbone=nn.Linear(2, 2)),
        feature_projector=nn.Linear(4, 2),
        energy_bm=nn.Linear(2, 1),
        sampler_type="sa",
        sampler_kwargs={"rand_seed": 42},
    ))


@pytest.mark.parametrize("existing", [False, True])
def test_partial_save_preserves_destination_and_removes_temporary_file(tmp_path, monkeypatch, existing):
    final_path = tmp_path / "权重.pt"
    original = b"previous checkpoint"
    if existing:
        final_path.write_bytes(original)

    def incomplete_save(payload, target):
        if hasattr(target, "write"):
            target.write(b"partial data")
        else:
            target.write_bytes(b"partial data")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", incomplete_save)
    with pytest.raises(OSError, match="simulated write failure"):
        save_checkpoint(tmp_path, final_path.name, generator=make_generator(), epoch=1, metric=1.0)

    if existing:
        assert final_path.read_bytes() == original
        assert list(tmp_path.iterdir()) == [final_path]
    else:
        assert list(tmp_path.iterdir()) == []


def test_successful_save_replaces_existing_file_and_round_trips(tmp_path):
    final_path = tmp_path / "best.pt"
    final_path.write_bytes(b"old checkpoint")
    source, restored = make_generator(), make_generator()
    result = save_checkpoint(tmp_path, final_path.name, generator=source, epoch=2, metric=0.25)

    assert result == final_path
    assert list(tmp_path.iterdir()) == [final_path]
    assert load_trained_energy_weights(restored, str(result), "cpu") == {
        "bm_sampler_type": "sa", "bm_sampler_kwargs": {"rand_seed": 42},
    }
    torch.testing.assert_close(restored.energy_model.energy_bm.weight, source.energy_model.energy_bm.weight)
