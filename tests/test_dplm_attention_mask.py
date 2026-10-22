"""Exercise explicit attention masks through the actual DPLM wrapper."""

import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import EsmConfig, EsmTokenizer


@pytest.fixture
def model(tmp_path):
    path = Path(__file__).resolve().parents[1] / "example/qdiffusion/dplm/models/esm_patch.py"
    spec = importlib.util.spec_from_file_location("dplm_mask_esm_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vocabulary = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "X", "<mask>"]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary), encoding="utf-8")
    EsmTokenizer(vocab_file=str(vocab_path)).save_pretrained(tmp_path)
    config = EsmConfig(
        vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=24,
        attention_probs_dropout_prob=0., hidden_dropout_prob=0.,
        position_embedding_type="rotary", token_dropout=False,
        pad_token_id=1, mask_token_id=8,
    )
    config._name_or_path = str(tmp_path)
    torch.manual_seed(42)
    return module._EsmForDPLM(config, dropout=0.).eval()


@pytest.mark.parametrize("dtype", [torch.bool, torch.long, torch.float32])
def test_explicit_mask_matches_underlying_backbone(model, dtype):
    tokens = torch.tensor([[0, 4, 5, 2, 1], [0, 6, 7, 4, 2]])
    mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]], dtype=dtype)
    expected = model.esm(tokens, attention_mask=mask).last_hidden_state
    actual = model(tokens, attention_mask=mask)
    torch.testing.assert_close(actual["last_hidden_state"], expected)
    torch.testing.assert_close(actual["logits"], model.lm_head(expected))


def test_masked_residue_cannot_affect_unmasked_predictions(model):
    tokens = torch.tensor([[0, 4, 5, 2]])
    changed = tokens.clone()
    changed[0, 2] = 6
    mask = torch.tensor([[1, 1, 0, 1]])
    first = model(tokens, attention_mask=mask)["logits"]
    second = model(changed, attention_mask=mask)["logits"]
    torch.testing.assert_close(first[:, [0, 1, 3]], second[:, [0, 1, 3]])
    assert not torch.allclose(model(tokens)["logits"], model(changed)["logits"])


def test_omitted_mask_preserves_padding_inference(model):
    tokens = torch.tensor([[0, 4, 5, 2, 1]])
    expected = model.esm(tokens, attention_mask=tokens.ne(1)).last_hidden_state
    torch.testing.assert_close(model(tokens)["last_hidden_state"], expected)


def test_explicit_all_ones_mask_takes_precedence(model):
    tokens = torch.tensor([[0, 4, 5, 2, 1]])
    mask = torch.ones_like(tokens)
    expected = model.esm(tokens, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(model(tokens, attention_mask=mask)["last_hidden_state"], expected)


def test_conditioned_embeddings_receive_no_gradient_through_masked_keys(model):
    tokens = torch.tensor([[0, 4, 5, 2]])
    mask = torch.tensor([[1, 1, 0, 1]])
    embeddings = model.get_input_embeddings()(tokens).detach().requires_grad_()
    logits = model(tokens, inputs_embeds=embeddings, attention_mask=mask)["logits"]
    logits[:, [0, 1, 3]].square().sum().backward()
    assert torch.count_nonzero(embeddings.grad[:, 2]) == 0
    assert torch.count_nonzero(embeddings.grad[:, [0, 1, 3]]) > 0
    assert torch.isfinite(embeddings.grad).all()
