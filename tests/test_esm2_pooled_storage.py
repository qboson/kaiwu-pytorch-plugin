"""CPU ESM pooling must retain vectors, not full token batches."""
from importlib.util import find_spec

import pytest
import torch

pytestmark = pytest.mark.skipif(
    any(find_spec(name) is None for name in ("esm", "omegaconf", "transformers")),
    reason="requires fair-esm and the Q-Diffusion example dependencies",
)


class Alphabet:
    padding_idx = 0

    def get_batch_converter(self):
        def convert(records):
            width = max(len(seq) for _, seq in records) + 2
            tokens = torch.zeros(len(records), width, dtype=torch.long)
            for i, (_, seq) in enumerate(records):
                tokens[i, :len(seq) + 2] = torch.arange(1, len(seq) + 3)
            return None, None, tokens
        return convert


class Encoder(torch.nn.Module):
    num_layers = 1

    def forward(self, tokens, **kwargs):
        del kwargs
        self.last = tokens.float().unsqueeze(-1).repeat(1, 1, 4)
        return {"representations": {1: self.last}}


@pytest.mark.parametrize("pooling", ["mean", "cls", "bos", "eos"])
def test_pooled_vectors_own_compact_storage(pooling):
    from example.qdiffusion.dplm.workflows.esm2_eval_helpers import embed_sequences

    records = [("short", "AC"), ("long", "A" * 128), ("last", "CDE")]
    model = Encoder()
    result = embed_sequences(records, model=model, alphabet=Alphabet(),
                             device=torch.device("cpu"), batch_size=2, pooling=pooling)
    expected = {"mean": [2.5, 65.5, 3.0], "cls": [1., 1., 1.],
                "bos": [1., 1., 1.], "eos": [4., 130., 5.]}[pooling]
    for (header, _), value in zip(records, expected):
        vector = result[header]
        torch.testing.assert_close(vector, torch.full((4,), value))
        assert vector.untyped_storage().nbytes() == vector.numel() * vector.element_size()
        assert not vector.requires_grad
    # Mutating the model's last batch cannot change a saved evaluation result.
    model.last.fill_(-99)
    torch.testing.assert_close(result["last"], torch.full((4,), expected[-1]))
