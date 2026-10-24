"""Feature-extraction delegation contract for the MNIST trainer."""

import importlib
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_PATH = list(sys.path)
sys.path[:] = [
    entry
    for entry in ORIGINAL_PATH
    if not entry.endswith("qdiffusion") and not entry.endswith("dplm")
]
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]
for _name in [m for m in list(sys.modules) if m == "trainer" or m.startswith("trainer.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

trainer_module = importlib.import_module("trainer.trainer")
sys.path[:] = ORIGINAL_PATH
Trainer = trainer_module.Trainer


class RecordingExtractor:
    """Records the model/feature type it is built with."""

    calls = []

    def __init__(self, model, feature_type):
        self.model = model
        self.feature_type = feature_type
        RecordingExtractor.calls.append((model, feature_type))

    def extract(self, dataloader):
        return ("features", dataloader)


def test_extract_features_builds_an_extractor_and_delegates(monkeypatch):
    """The trainer wraps the model in a FeatureExtractor and calls extract."""
    RecordingExtractor.calls = []
    monkeypatch.setattr(trainer_module, "FeatureExtractor", RecordingExtractor)
    trainer = Trainer.__new__(Trainer)
    trainer.model = object()
    loader = object()
    result = trainer.extract_features(loader, feature_type="zeta")
    assert result == ("features", loader)
    assert RecordingExtractor.calls == [(trainer.model, "zeta")]
