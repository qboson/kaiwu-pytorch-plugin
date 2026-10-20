"""Checkpoint restoration stages tensors on CPU without changing model state."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('dplm_runtime',
    Path(__file__).resolve().parents[1] / 'example/qdiffusion/dplm/utils/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def generator():
    return SimpleNamespace(
        proposal_model=nn.Linear(3, 2),
        energy_model=SimpleNamespace(
            encoder=SimpleNamespace(backbone=nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))),
            feature_projector=nn.Linear(3, 2), energy_bm=nn.Linear(2, 1),
            sampler_type='sa', sampler_kwargs={'alpha': .9}))


def energy_modules(model):
    energy = model.energy_model
    return [energy.encoder.backbone, energy.feature_projector, energy.energy_bm]


@pytest.mark.parametrize('requested_device', ['cpu', 'cuda:0'])
def test_cpu_staging_restores_actual_checkpoint_and_retains_destinations(tmp_path, requested_device):
    source, target = generator(), generator()
    source.energy_model.encoder.backbone[1].running_mean.fill_(2)
    source.energy_model.encoder.backbone[1].num_batches_tracked.fill_(7)
    before_proposal = {k: t.clone() for k, t in target.proposal_model.state_dict().items()}
    path = runtime.save_checkpoint(tmp_path, 'energy.pt', generator=source, epoch=3, metric=.5)
    original = torch.load
    locations = []

    def load(*args, **kwargs):
        locations.append(kwargs.get('map_location'))
        # Keep this policy regression runnable without CUDA. The actual loader,
        # serialization and restored modules are real CPU implementations.
        kwargs['map_location'] = 'cpu'
        return original(*args, **kwargs)

    with patch.object(runtime.torch, 'load', side_effect=load):
        metadata = runtime.load_trained_energy_weights(target, str(path), requested_device)
    assert locations == ['cpu']
    assert metadata == {'bm_sampler_type': 'sa', 'bm_sampler_kwargs': {'alpha': .9}}
    for actual, expected in zip(energy_modules(target), energy_modules(source)):
        for key, tensor in actual.state_dict().items():
            torch.testing.assert_close(tensor, expected.state_dict()[key])
            assert tensor.device.type == 'cpu'
    for key, tensor in target.proposal_model.state_dict().items():
        torch.testing.assert_close(tensor, before_proposal[key])
    assert target.energy_model.encoder.backbone[1].num_batches_tracked.item() == 7


def test_unwrapped_cpu_roundtrip(tmp_path):
    source, target = generator(), generator()
    path = runtime.save_checkpoint(tmp_path, 'energy.pt', generator=source, epoch=1, metric=1.)
    runtime.load_trained_energy_weights(target, str(path), 'cpu')
    x = torch.randn(4, 3)
    for model in (source, target):
        model.energy_model.encoder.backbone.eval()
    def score(model):
        e = model.energy_model
        return e.energy_bm(e.feature_projector(e.encoder.backbone(x)))
    torch.testing.assert_close(score(target), score(source))
