"""ESM2 model-loading contract for the distance workflow."""

import os
import sys
import types

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPLM = os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm")
sys.path[:0] = [DPLM, os.path.join(DPLM, "workflows")]

_esm = types.ModuleType("esm")
_esm.pretrained = types.SimpleNamespace()
_added = "esm" not in sys.modules
if _added:
    sys.modules["esm"] = _esm
import workflows.esm2_eval_helpers as helpers  # noqa: E402

load_esm2_model = helpers.load_esm2_model
if _added:
    sys.modules.pop("esm", None)


class RecordingModel(torch.nn.Module):
    """Records eval()/to() calls made by the loader."""

    def __init__(self):
        super().__init__()
        self.eval_called = False
        self.device_arg = None

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device_arg = device
        return self


def install_pretrained(monkeypatch):
    """Registers two fake pretrained constructors and returns them."""
    model = RecordingModel()
    alphabet = object()

    def loader_a():
        return model, alphabet

    def loader_b():
        return "other-model", "other-alphabet"

    monkeypatch.setattr(
        helpers.esm,
        "pretrained",
        types.SimpleNamespace(esm2_fake=loader_a, esm2_other=loader_b),
        raising=False,
    )
    return model, alphabet


def test_supported_model_is_evaluated_and_moved(monkeypatch):
    """The loader builds the model, switches it to eval and moves it to the device."""
    model, alphabet = install_pretrained(monkeypatch)
    device = torch.device("cpu")
    result_model, result_alphabet = load_esm2_model("esm2_fake", device)
    assert result_model is model
    assert result_alphabet is alphabet
    assert model.eval_called is True
    assert model.device_arg == device


def test_unsupported_model_lists_available_examples(monkeypatch):
    """An unknown name fails with the available esm2_* constructors named."""
    install_pretrained(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported esm.pretrained model") as info:
        load_esm2_model("esm2_missing", torch.device("cpu"))
    assert "esm2_fake" in str(info.value)
    assert "esm2_other" in str(info.value)
