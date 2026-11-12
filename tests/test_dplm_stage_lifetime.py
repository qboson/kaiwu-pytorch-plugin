"""Keep only one DPLM stage resident when the workflow changes models."""

import weakref

import pytest
import torch
from torch import nn

pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from example.qdiffusion.dplm.workflows import train


def test_workflow_releases_models_between_stages_and_evaluates_inference(tmp_path, monkeypatch):
    fasta = tmp_path / "sequences.fasta"
    fasta.write_text(">first\nAAA\n>second\nCCC\n>third\nDDD\n", encoding="utf-8")
    config = train.WorkflowConfig(
        data=train.DataConfig(str(fasta), min_length=1, max_length=10),
        model=train.ModelConfig("proposal", "energy"),
        train=train.TrainConfig(epochs=1, require_cuda=False),
    )
    references = []
    retained_counts = []
    inference_modes = []

    def build(*args, **kwargs):
        retained_counts.append(sum(ref() is not None for ref in references))
        model = nn.Linear(2, 2)
        references.extend((weakref.ref(model), weakref.ref(model.weight)))
        return model

    def validate(generator, *args, **kwargs):
        inference_modes.append(generator.training)
        return {}

    def generate(generator, records, **kwargs):
        inference_modes.append(generator.training)
        return records, []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "build_default_workflow_config", lambda: config)
    monkeypatch.setattr(train, "default_outputs_root", lambda: tmp_path)
    monkeypatch.setattr(train, "build_generator_from_config", build)
    monkeypatch.setattr(train, "run_structural_validation", validate)
    monkeypatch.setattr(train, "build_data_loader_from_records", lambda model, *a, **k: (model,))
    monkeypatch.setattr(train, "run_epoch", lambda *a, **k: {"energy_objective_mean": 1.0})
    monkeypatch.setattr(train, "save_checkpoint", lambda output, name, **kwargs: output / name)
    monkeypatch.setattr(train, "load_trained_energy_weights", lambda *args: None)
    monkeypatch.setattr(train, "run_generation_over_records", generate)

    train.main()

    assert retained_counts == [0, 0, 0, 0]
    assert inference_modes == [False, False, False]
