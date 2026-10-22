"""The DPLM SDPA patch must preserve ESM's train/eval dropout contract."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
from transformers import EsmConfig
from transformers.models.esm.modeling_esm import EsmModel, EsmSelfAttention


@pytest.fixture(scope="module")
def esm_patch():
    """Load the actual example without constructing or downloading a backbone."""
    path = (
        Path(__file__).resolve().parents[1]
        / "example/qdiffusion/dplm/models/esm_patch.py"
    )
    spec = importlib.util.spec_from_file_location("dplm_dropout_esm_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_config(probability):
    """Disable other dropout so that attention is the only stochastic path."""
    return EsmConfig(
        vocab_size=33,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=24,
        attention_probs_dropout_prob=probability,
        hidden_dropout_prob=0.0,
        position_embedding_type="rotary",
        pad_token_id=1,
        mask_token_id=4,
        token_dropout=False,
    )


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_attention_matches_reference_at_dropout_endpoints(
    esm_patch, training, probability
):
    """Use deterministic endpoints to check outputs and backward, not RNG layout."""
    torch.manual_seed(41)
    reference = EsmSelfAttention(tiny_config(probability)).train(training)
    actual = esm_patch._ModifiedEsmSelfAttention(tiny_config(probability)).train(training)
    actual.load_state_dict(reference.state_dict())
    reference_input = torch.randn(2, 5, 16, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_()
    mask = torch.zeros(2, 1, 1, 5)
    mask[1, :, :, -1] = torch.finfo(torch.float32).min

    expected = reference(reference_input, attention_mask=mask)[0]
    observed = actual(actual_input, attention_mask=mask)[0]
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    expected.sum().backward()
    observed.sum().backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad)
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad)


def test_nonzero_attention_dropout_is_training_only(esm_patch):
    """A normal nonzero rate changes training samples but never evaluation."""
    torch.manual_seed(41)
    actual = esm_patch._ModifiedEsmSelfAttention(tiny_config(0.5))
    inputs = torch.randn(2, 5, 16)
    actual.train()
    torch.manual_seed(1)
    first = actual(inputs)[0]
    torch.manual_seed(2)
    second = actual(inputs)[0]
    assert not torch.equal(first, second)
    second.square().sum().backward()
    assert torch.count_nonzero(actual.value.weight.grad) > 0
    for parameter in actual.parameters():
        assert torch.isfinite(parameter.grad).all()
    actual.eval()
    torch.manual_seed(1)
    first = actual(inputs)[0]
    torch.manual_seed(2)
    second = actual(inputs)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    reference = EsmSelfAttention(tiny_config(0.5)).eval()
    reference.load_state_dict(actual.state_dict())
    torch.testing.assert_close(first, reference(inputs)[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("training", [False, True])
def test_full_backbone_matches_reference_with_attention_dropout(esm_patch, training):
    """Exercise the real embedding, encoder and backward path with padding."""
    torch.manual_seed(41)
    reference = EsmModel(tiny_config(1.0), add_pooling_layer=False).train(training)
    actual = esm_patch._ModifiedEsmModel(
        tiny_config(1.0), add_pooling_layer=False
    ).train(training)
    actual.load_state_dict(reference.state_dict())
    tokens = torch.tensor([[0, 5, 6, 7, 2], [0, 8, 9, 2, 1]])
    mask = tokens.ne(1)
    expected = reference(tokens, attention_mask=mask).last_hidden_state
    observed = actual(tokens, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    readout = torch.randn_like(expected)
    (expected * readout).sum().backward()
    (observed * readout).sum().backward()
    for name, parameter in actual.named_parameters():
        expected_gradient = reference.get_parameter(name).grad
        if expected_gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(
                parameter.grad, expected_gradient, rtol=1e-4, atol=1e-5
            )
