"""Energy-model wiring contract for the DPLM example builder."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import pytest

import utils.dplm_builder as builder  # noqa: E402


class RecordingEncoder:
    """Captures the backbone handed to the feature encoder."""

    def __init__(self, backbone):
        self.backbone = backbone


class RecordingEnergyModel:
    """Captures the keyword arguments handed to the energy model."""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingEnergyModel.last = self


@pytest.fixture()
def recording_classes(monkeypatch):
    """Replaces both model classes with recorders."""
    monkeypatch.setattr(builder, "DPLMFeatureEncoder", RecordingEncoder)
    monkeypatch.setattr(builder, "BMConditionedEnergyModel", RecordingEnergyModel)
    RecordingEnergyModel.last = None
    return RecordingEnergyModel


def test_energy_model_wraps_the_encoder_and_forwards_the_bm_settings(recording_classes):
    """Visible/hidden widths, sampler object and sampler config all pass through."""
    backbone = object()
    sampler = object()
    model = builder._build_energy_model(
        energy_backbone=backbone,
        bm_num_visible=7,
        bm_num_hidden=3,
        bm_sampler=sampler,
        bm_sampler_type="cim",
        bm_sampler_kwargs={"tmp_dir": "checkpoints"},
    )
    assert isinstance(model, RecordingEnergyModel)
    encoder = model.kwargs["encoder"]
    assert isinstance(encoder, RecordingEncoder)
    assert encoder.backbone is backbone
    assert model.kwargs["bm_num_visible"] == 7
    assert model.kwargs["bm_num_hidden"] == 3
    assert model.kwargs["sampler"] is sampler
    assert model.kwargs["sampler_type"] == "cim"
    assert model.kwargs["sampler_kwargs"] == {"tmp_dir": "checkpoints"}
