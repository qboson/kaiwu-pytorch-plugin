"""QVAE centering-state checkpoint regressions with real neural components."""

from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

src_root = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_root))

import kaiwu

kaiwu.__path__ = [str(src_root / "kaiwu")] + list(kaiwu.__path__)
for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith("kaiwu.torch_plugin."):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine
from kaiwu.torch_plugin.qvae import QVAE


INPUTS = torch.tensor([[0.4, 0.6]])


def make_model(mean=None):
    """Use recognizable centered logits and nontrivial binary-state energies."""
    encoder, decoder = nn.Linear(2, 2), nn.Linear(2, 2)
    with torch.no_grad():
        encoder.weight.copy_(10 * torch.eye(2))
        encoder.bias.zero_()
        decoder.weight.copy_(torch.eye(2))
        decoder.bias.zero_()
    bm = BoltzmannMachine(
        2, quadratic_coef=torch.zeros(2, 2), linear_bias=torch.tensor([1.0, 2.0]),
        device=torch.device("cpu"),
    )
    config = SimpleNamespace(
        num_latent_units=2, loss_type="bernoulli", dist_beta=1.0,
        kl_beta=0.0, weight_decay=0.0,
    )
    model = QVAE(
        2, None, config, encoder=encoder, decoder=decoder, bm=bm, sampler=object(),
    ).eval()
    model.set_dataset_mean(mean)
    # Deliberately independent from the centering mean: it cannot recover that state.
    model.set_train_bias(0.2)
    return model


def forward_values(model):
    """Compare complete forward outputs under identical posterior samples."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        reconstruction, posterior, logits, latent = model(INPUTS)
    return reconstruction, posterior.logit_mu, logits, latent


def assert_same_forward(original, restored):
    for expected, actual in zip(forward_values(original), forward_values(restored)):
        torch.testing.assert_close(actual, expected)


def serialize_state(model):
    """Exercise ordinary tensor checkpoint serialization with weights_only=True."""
    checkpoint = BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    return torch.load(checkpoint, weights_only=True)


@pytest.mark.parametrize("mean", [0.8, [0.8, 0.8], torch.tensor([0.8, 0.8])])
@pytest.mark.parametrize("initial_mean", [None, 0.0])
def test_strict_checkpoint_roundtrip_preserves_centering_and_real_outputs(mean, initial_mean):
    """Fresh and already-configured models must recover the checkpoint's mean."""
    original, restored = make_model(mean), make_model(initial_mean)
    restored.load_state_dict(serialize_state(original), strict=True)

    assert_same_forward(original, restored)
    torch.testing.assert_close(original.energy(INPUTS), restored.energy(INPUTS))
    torch.testing.assert_close(restored._train_bias, original._train_bias)


def test_unset_mean_checkpoint_clears_an_existing_mean():
    """Loading an uncentered model must restore its actual forward behavior."""
    original, restored = make_model(), make_model(0.8)
    restored.load_state_dict(serialize_state(original), strict=True)

    assert restored._dataset_mean is None
    assert_same_forward(original, restored)


@pytest.mark.parametrize("current_mean", [None, 0.3])
def test_legacy_strict_checkpoint_preserves_the_receiving_models_mean(current_mean):
    """Legacy checkpoints cannot reconstruct a mean they never serialized."""
    legacy = {
        key: value for key, value in make_model(0.8).state_dict().items()
        if key != "_extra_state"
    }
    original_keys = set(legacy)
    expected, restored = make_model(current_mean), make_model(current_mean)
    restored.load_state_dict(legacy, strict=True)

    assert set(legacy) == original_keys
    assert_same_forward(expected, restored)
    if current_mean is None:
        assert restored._dataset_mean is None
    else:
        torch.testing.assert_close(expected.energy(INPUTS), restored.energy(INPUTS))


def test_nested_model_checkpoint_restores_centering_with_a_prefix():
    """Extra state must load correctly inside another registered module."""
    original, restored = nn.Sequential(make_model(0.8)), nn.Sequential(make_model(0.0))
    restored.load_state_dict(serialize_state(original), strict=True)

    assert_same_forward(original[0], restored[0])
    torch.testing.assert_close(original[0].energy(INPUTS), restored[0].energy(INPUTS))


def test_nested_legacy_checkpoint_preserves_configured_mean():
    """Legacy compatibility must inject only the correctly prefixed own key."""
    original, restored = nn.Sequential(make_model(0.3)), nn.Sequential(make_model(0.3))
    legacy = {key: value for key, value in original.state_dict().items() if key != "0._extra_state"}
    restored.load_state_dict(legacy, strict=True)

    assert_same_forward(original[0], restored[0])


@pytest.mark.parametrize("corruption", ["missing", "unexpected"])
def test_legacy_compatibility_does_not_weaken_other_strict_checks(corruption):
    """Only absence of the new mean state is compatible with strict legacy loads."""
    legacy = {key: value for key, value in make_model(0.8).state_dict().items() if key != "_extra_state"}
    if corruption == "missing":
        del legacy["encoder.weight"]
        expected_key = "encoder.weight"
    else:
        legacy["unrecognized"] = torch.tensor(0.0)
        expected_key = "unrecognized"
    with pytest.raises(RuntimeError, match=expected_key):
        make_model(0.0).load_state_dict(legacy, strict=True)


def test_checkpoint_supports_official_trainers_tensor_only_clone():
    """The existing CellQVAE best-state cloning expression must remain usable."""
    original, restored = make_model(0.8), make_model(0.0)
    checkpoint = {
        key: value.detach().cpu().clone()
        for key, value in original.state_dict().items()
    }
    restored.load_state_dict(checkpoint, strict=True)

    assert_same_forward(original, restored)
    torch.testing.assert_close(original.energy(INPUTS), restored.energy(INPUTS))


def test_mean_is_nontrainable_model_state_and_follows_dtype_migration():
    """Centering statistics should not receive gradients and should move with buffers."""
    supplied_mean = torch.tensor([0.8, 0.8], requires_grad=True)
    model = make_model(supplied_mean)
    model(INPUTS)[2].sum().backward()

    assert supplied_mean.grad is None
    assert model.encoder.weight.grad is not None
    assert "_dataset_mean" in dict(model.named_buffers())
    assert "_dataset_mean" not in dict(model.named_parameters())
    model.to(dtype=torch.float64)
    assert model._dataset_mean.dtype == torch.float64


def test_checkpoint_mean_is_an_independent_snapshot():
    """Captured statistics must not alias the live or subsequently restored mean."""
    model = make_model(torch.tensor([0.8, 0.8]))
    checkpoint = model.state_dict()
    model._dataset_mean.fill_(0.0)
    restored = make_model(0.0)
    restored.load_state_dict(checkpoint, strict=True)
    expected = make_model(0.8)

    assert_same_forward(expected, restored)
    checkpoint["_extra_state"].fill_(0.0)
    assert_same_forward(expected, restored)


@pytest.mark.parametrize("initial_mean", [None, torch.tensor([0.1, 0.2])])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("assign", [False, True])
def test_loading_checkpoint_mean_follows_receiving_precision(initial_mean, nested, assign):
    """The mean follows parameter properties in both supported load modes."""
    source = make_model(torch.tensor([0.8, 0.8], dtype=torch.float32))
    receiving = make_model(initial_mean).double()
    source_container = nn.Sequential(source) if nested else source
    target_container = nn.Sequential(receiving) if nested else receiving
    checkpoint = serialize_state(source_container)

    target_container.load_state_dict(checkpoint, strict=True, assign=assign)
    expected_dtype = torch.float32 if assign else torch.float64
    inputs = INPUTS.to(expected_dtype)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        reconstruction, posterior, logits, latent = receiving(inputs)

    mean_key = "0._extra_state" if nested else "_extra_state"
    expected_mean = checkpoint[mean_key].to(expected_dtype)
    torch.testing.assert_close(logits, 10.0 * (inputs - expected_mean))
    for output in [reconstruction, posterior.logit_mu, logits, latent]:
        assert output.dtype == expected_dtype
        assert torch.isfinite(output).all()
    assert receiving.encoder.weight.dtype == expected_dtype
    assert receiving._train_bias.dtype == expected_dtype
    assert receiving._dataset_mean.dtype == expected_dtype
    assert receiving._dataset_mean.device == receiving.encoder.weight.device
    torch.testing.assert_close(receiving._dataset_mean, expected_mean, rtol=0.0, atol=0.0)
