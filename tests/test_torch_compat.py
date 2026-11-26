"""Cover the PyTorch range declared in requirements/requirements.txt.

The suite runs once per torch version in the CI matrix (2.3 / 2.7 / 2.14),
so these tests deliberately stay SDK-free (no CIM samplers, no license
stubs): a matrix failure should point at the torch side of the dependency
window, not at an SDK release.
"""

import inspect
import pickle
import re

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


SUPPORTED_MIN = (2, 3)
SUPPORTED_MAX = (2, 15)  # exclusive, mirrors the requirements.txt upper bound


def installed_torch_version() -> tuple[int, int]:
    """Parses the running torch version into a (major, minor) pair."""
    release = re.match(r"\d+\.\d+", torch.__version__)
    assert release, torch.__version__
    major, minor = (int(part) for part in release.group(0).split("."))
    return major, minor


def test_installed_torch_is_within_supported_range():
    """The running torch satisfies torch>=2.3,<2.15."""
    assert SUPPORTED_MIN <= installed_torch_version() < SUPPORTED_MAX, (
        torch.__version__
    )


def test_torch_load_accepts_weights_only_kwarg():
    """torch.load exposes weights_only across the whole supported range."""
    assert "weights_only" in inspect.signature(torch.load).parameters


def build_model(boltzmann: bool):
    """Builds a small BM (4 nodes) or RBM (2 visible + 2 hidden) on CPU."""
    if boltzmann:
        return BoltzmannMachine(4, device="cpu")
    return RestrictedBoltzmannMachine(2, 2, device="cpu")


@pytest.mark.parametrize("boltzmann", [False, True], ids=["rbm", "bm"])
def test_plugin_forward_on_installed_torch(boltzmann):
    """Models build and evaluate a forward pass on the installed torch."""
    torch.manual_seed(7)
    model = build_model(boltzmann)
    states = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    energy = model(states)
    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()


@pytest.mark.parametrize("boltzmann", [False, True], ids=["rbm", "bm"])
def test_state_dict_checkpoint_roundtrip(tmp_path, boltzmann):
    """State_dict files round-trip through torch.load(weights_only=True).

    Every fixed example now loads checkpoints this way (see the
    fix(compat) commit), so the round-trip must hold on each torch in the
    supported range, including the versions where weights_only=True is no
    longer the implicit default.
    """
    torch.manual_seed(7)
    source = build_model(boltzmann)
    checkpoint = tmp_path / "model_state.pt"
    torch.save(source.state_dict(), checkpoint)

    restored = build_model(boltzmann)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))

    states = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    torch.testing.assert_close(source(states), restored(states))


def test_example_checkpoint_payload_roundtrip(tmp_path):
    """The qdeq example payload (state_dict + sampler string) survives the
    safe loader used by simple_forecast_example.py."""
    torch.manual_seed(7)
    source = RestrictedBoltzmannMachine(2, 2, device="cpu")
    checkpoint = tmp_path / "deq_forecaster.pt"
    torch.save(
        {"state_dict": source.state_dict(), "sampler": "sa"}, checkpoint
    )

    payload = torch.load(checkpoint, weights_only=True)
    restored = RestrictedBoltzmannMachine(2, 2, device="cpu")
    restored.load_state_dict(payload["state_dict"])
    assert payload["sampler"] == "sa"

    states = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    torch.testing.assert_close(source(states), restored(states))


class _ArbitraryObject:
    """Stands in for pickled payloads no safe loader may restore."""

    def __init__(self, value):
        self.value = value


def test_weights_only_true_rejects_arbitrary_objects(tmp_path):
    """weights_only=True refuses non-allowlisted payloads.

    The state_dict call sites pass weights_only=True on purpose; this pins
    that the flag is actually enforced (it would silently downgrade to an
    unrestricted pickle load if the flag regressed to a no-op).
    """
    checkpoint = tmp_path / "arbitrary.pt"
    torch.save({"payload": _ArbitraryObject(42)}, checkpoint)
    with pytest.raises(pickle.UnpicklingError):
        torch.load(checkpoint, weights_only=True)


def test_weights_only_false_restores_arbitrary_objects(tmp_path):
    """weights_only=False restores full pickled objects.

    molecular_design/run_generation_strict.py loads checkpoints containing
    arbitrary objects and therefore passes weights_only=False explicitly;
    this pins the contract those call sites rely on.
    """
    checkpoint = tmp_path / "arbitrary.pt"
    torch.save({"payload": _ArbitraryObject(42)}, checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["payload"].value == 42
