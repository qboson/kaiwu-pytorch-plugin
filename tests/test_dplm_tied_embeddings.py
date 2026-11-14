"""Offline checkpoint compatibility for the example's patched ESM model."""

import importlib

import pytest
import torch

TRANSFORMERS = pytest.importorskip("transformers", minversion="4.39")
pytest.importorskip("omegaconf")
SAFETENSORS = pytest.importorskip("safetensors.torch")
DPLM_BACKBONE = importlib.import_module(
    "example.qdiffusion.dplm.models.backbone"
).DPLMBackbone
ESM_FOR_DPLM = getattr(
    importlib.import_module("example.qdiffusion.dplm.models.esm_patch"), "_EsmForDPLM"
)


@pytest.fixture(name="esm_checkpoint")
def fixture_esm_checkpoint(tmp_path):
    """Export a real tiny ESM checkpoint and tokenizer without network access."""
    def export(tied=True, explicit_untied=False):
        folder = tmp_path / ("tied" if tied else "untied")
        folder.mkdir()
        vocabulary = [
            "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E",
            "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H",
            "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>",
        ]
        vocab_path = folder / "vocab.txt"
        vocab_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
        tokenizer = TRANSFORMERS.EsmTokenizer(vocab_file=str(vocab_path))
        tokenizer.save_pretrained(folder)
        config = TRANSFORMERS.EsmConfig(
            vocab_size=len(vocabulary), hidden_size=24, num_hidden_layers=1,
            num_attention_heads=3, intermediate_size=32, max_position_embeddings=128,
            position_embedding_type="rotary", pad_token_id=1, mask_token_id=32,
            token_dropout=True, hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0, tie_word_embeddings=tied,
        )
        config.dplm_type = "dplm_esm"
        torch.manual_seed(9)
        reference = TRANSFORMERS.EsmForMaskedLM(config).eval()
        if explicit_untied:
            # Historical custom DPLM models can store both weights despite this flag.
            reference.lm_head.decoder.weight = torch.nn.Parameter(
                reference.lm_head.decoder.weight.detach().clone()
                + 0.1 * torch.randn_like(reference.lm_head.decoder.weight)
            )
        reference.save_pretrained(folder, safe_serialization=True)
        tokens = tokenizer(["ACDEFG", "WQ"], padding=True, return_tensors="pt")["input_ids"]
        tokens[0, 3] = tokenizer.mask_token_id
        with torch.no_grad():
            expected = reference(tokens, attention_mask=tokens.ne(1)).logits
        return folder, tokens, expected
    return export


@pytest.mark.parametrize("tied", [True, False])
def test_safetensors_checkpoint_preserves_logits(esm_checkpoint, tied):
    """Both shared and independently trained output projections reload fully."""
    folder, tokens, expected = esm_checkpoint(tied)
    saved = SAFETENSORS.load_file(str(folder / "model.safetensors"))
    assert ("lm_head.decoder.weight" not in saved) is tied
    model, info = ESM_FOR_DPLM.from_pretrained(
        str(folder), dropout=0.0, output_loading_info=True, local_files_only=True,
    )
    model.eval()
    assert info["missing_keys"] == []
    assert info["unexpected_keys"] == []
    assert info["mismatched_keys"] == []
    shared = model.lm_head.decoder.weight is model.get_input_embeddings().weight
    assert shared is tied
    with torch.no_grad():
        actual = model(tokens)["logits"]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_backbone_restores_tied_proposal_logits(esm_checkpoint):
    """Exercise the actual factory entry used by the DPLM generator builder."""
    folder, tokens, expected = esm_checkpoint()
    model = DPLM_BACKBONE.from_pretrained(
        str(folder), net_override={"dropout": 0.0, "local_files_only": True},
    ).eval()
    with torch.no_grad():
        actual = model(tokens)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


def test_patched_esm_save_reload_keeps_shared_projection(esm_checkpoint, tmp_path):
    """After an embedding update, export and reload the complete tied model."""
    folder, tokens, _ = esm_checkpoint()
    model = ESM_FOR_DPLM.from_pretrained(
        str(folder), dropout=0.0, local_files_only=True, torch_dtype=torch.float64,
        use_safetensors=True,
    )
    assert isinstance(model, ESM_FOR_DPLM)
    model.eval()
    assert model.esm.config.hidden_dropout_prob == 0.0
    assert model.lm_head.decoder.weight is model.get_input_embeddings().weight
    assert model.lm_head.decoder.weight.dtype == torch.float64
    assert model.lm_head.decoder.weight.device.type == "cpu"
    # Stand in for a completed training update and ensure the current weights save.
    with torch.no_grad():
        model.get_input_embeddings().weight.add_(0.05)
        expected = model(tokens)["logits"]
    output = tmp_path / "roundtrip"
    model.save_pretrained(output)
    model.tokenizer.save_pretrained(output)
    restored, info = ESM_FOR_DPLM.from_pretrained(
        str(output), dropout=0.0, output_loading_info=True, local_files_only=True,
        torch_dtype=torch.float64,
    )
    restored.eval()
    assert info["missing_keys"] == []
    assert restored.lm_head.decoder.weight is restored.get_input_embeddings().weight
    assert restored.lm_head.decoder.weight.dtype == torch.float64
    assert restored.lm_head.decoder.weight.device.type == "cpu"
    assert torch.equal(
        restored.state_dict()["lm_head.decoder.weight"],
        model.state_dict()["lm_head.decoder.weight"],
    )
    with torch.no_grad():
        actual = restored(tokens)["logits"]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_explicit_decoder_survives_legacy_tie_flag(esm_checkpoint, tmp_path):
    """Existing independent projections must win over a legacy config default."""
    folder, tokens, expected = esm_checkpoint(explicit_untied=True)
    assert "lm_head.decoder.weight" in SAFETENSORS.load_file(str(folder / "model.safetensors"))
    model, info = ESM_FOR_DPLM.from_pretrained(
        str(folder), dropout=0.0, output_loading_info=True, local_files_only=True,
    )
    model.eval()
    assert info["missing_keys"] == []
    assert model.config.tie_word_embeddings
    assert model.lm_head.decoder.weight is not model.get_input_embeddings().weight
    with torch.no_grad():
        actual = model(tokens)["logits"]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    output = tmp_path / "native_roundtrip"
    model.save_pretrained(output)
    model.tokenizer.save_pretrained(output)
    saved = SAFETENSORS.load_file(str(output / "model.safetensors"))
    assert "lm_head.decoder.weight" in saved
    restored, info = ESM_FOR_DPLM.from_pretrained(
        str(output), dropout=0.0, output_loading_info=True, local_files_only=True,
    )
    assert info["missing_keys"] == []
    assert restored.lm_head.decoder.weight is not restored.get_input_embeddings().weight
    assert torch.equal(model.lm_head.decoder.weight, restored.lm_head.decoder.weight)
    with torch.no_grad():
        reloaded = restored(tokens)["logits"]
    assert torch.equal(reloaded, actual)


def test_loading_info_keeps_unrelated_missing_weight(esm_checkpoint):
    """Restoring the known shared weight must not hide incomplete checkpoints."""
    folder, _, _ = esm_checkpoint()
    path = folder / "model.safetensors"
    weights = SAFETENSORS.load_file(str(path))
    del weights["lm_head.bias"]
    SAFETENSORS.save_file(weights, str(path), metadata={"format": "pt"})
    model, info = ESM_FOR_DPLM.from_pretrained(
        str(folder), dropout=0.0, output_loading_info=True, local_files_only=True,
    )
    assert info["missing_keys"] == ["lm_head.bias"]
    assert model.lm_head.decoder.weight is model.get_input_embeddings().weight


def test_untied_config_keeps_missing_decoder_visible(esm_checkpoint):
    """An incomplete untied checkpoint must not become tied implicitly."""
    folder, _, _ = esm_checkpoint(tied=False)
    path = folder / "model.safetensors"
    weights = SAFETENSORS.load_file(str(path))
    del weights["lm_head.decoder.weight"]
    SAFETENSORS.save_file(weights, str(path), metadata={"format": "pt"})
    model, info = ESM_FOR_DPLM.from_pretrained(
        str(folder), dropout=0.0, output_loading_info=True, local_files_only=True,
    )
    assert not model.config.tie_word_embeddings
    assert info["missing_keys"] == ["lm_head.decoder.weight"]
    assert model.lm_head.decoder.weight is not model.get_input_embeddings().weight
