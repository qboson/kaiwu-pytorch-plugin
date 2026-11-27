"""Pool ESM representations by tokenizer positions."""

import pytest
import torch

esm = pytest.importorskip("esm")

from example.qdiffusion.dplm.workflows.esm2_eval_helpers import embed_sequences


class TokenIdModel(torch.nn.Module):
    """Return a representation identifiable by the token ID at each position."""

    num_layers = 1

    def forward(self, tokens, repr_layers, return_contacts):
        return {"representations": {repr_layers[0]: tokens.to(torch.float32).unsqueeze(-1)}}


def embed(records, alphabet, pooling):
    return embed_sequences(
        records, model=TokenIdModel(), alphabet=alphabet,
        device=torch.device("cpu"), batch_size=4, pooling=pooling,
    )


def test_mean_pooling_excludes_eos_and_padding_after_multicharacter_tokens():
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    sequences = [("unknown", "A<unk>C"), ("masked", "A<mask>"), ("plain", "ACDEF")]
    pooled = embed(sequences, alphabet, "mean")

    assert list(pooled) == ["unknown", "masked", "plain"]
    for header, sequence in sequences:
        expected = torch.tensor(alphabet.encode(sequence), dtype=torch.float32).mean()
        torch.testing.assert_close(pooled[header], expected.unsqueeze(0))


@pytest.mark.parametrize("pooling,token", [("bos", "<cls>"), ("eos", "<eos>")])
def test_boundary_pooling_uses_token_positions_with_variable_length_rows(pooling, token):
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    pooled = embed([("short", "A<unk>"), ("long", "ACDEF")], alphabet, pooling)

    for value in pooled.values():
        torch.testing.assert_close(value, torch.tensor([float(alphabet.get_idx(token))]))


def test_mean_pooling_can_use_an_alphabet_without_eos():
    alphabet = esm.data.Alphabet.from_architecture("ESM-1")
    pooled = embed([("short", "A<unk>"), ("long", "ACDEF")], alphabet, "mean")

    expected = torch.tensor(alphabet.encode("A<unk>"), dtype=torch.float32).mean()
    torch.testing.assert_close(pooled["short"], expected.unsqueeze(0))
    with pytest.raises(ValueError, match="EOS pooling requires"):
        embed([("short", "A<unk>")], alphabet, "eos")
