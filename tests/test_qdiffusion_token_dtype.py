"""Public decoding accepts the integer dtypes supported by embedding backbones."""
import pytest
import torch
from torch import nn
from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel, QDiffusion, QDiffusionConfig, SequenceTokenSpec,
)

class EmbeddingProposal(nn.Module):
    """A real embedding/head model with a decisive ordinary-token prediction."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(9, 4)
        self.head = nn.Linear(4, 9)
        self.seen_dtypes = []
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.fill_(-100.0)
            self.head.bias[5] = 100.0

    def forward(self, tokens):
        """Run the embedding/head on the caller's token dtype."""
        self.seen_dtypes.append(tokens.dtype)
        return self.head(self.embedding(tokens))

class EmbeddingEnergy(EnergyModel):
    """Score real candidate embeddings with a trainable projection."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(9, 4)
        self.head = nn.Linear(4, 1)

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del noisy_tokens
        features = self.embedding(candidate_tokens) * attention_mask.unsqueeze(-1)
        return self.head(features.sum(1))

def make_generator(resample):
    """Construct a reproducible neural model without remote resources."""
    torch.manual_seed(317)
    return QDiffusion(
        EmbeddingProposal(), EmbeddingEnergy(),
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
        QDiffusionConfig(num_candidates=2, disable_resample=not resample),
    )

@pytest.mark.parametrize('dtype', [torch.int64, torch.int32])
@pytest.mark.parametrize('resample', [False, True])
@pytest.mark.parametrize('steps', [1, 3])
@pytest.mark.parametrize('partial', [False, True])
def test_generate_keeps_supported_token_dtype(dtype, resample, steps, partial):
    """Both embedding integer dtypes decode with exact structure and RNG parity."""
    model = make_generator(resample)
    tokens = torch.tensor([[1, 5, 6, 7, 2, 0], [1, 8, 5, 6, 7, 2]], dtype=dtype)
    fixed = torch.zeros_like(tokens, dtype=torch.bool)
    if partial:
        fixed[:, 1] = True
    expected = tokens.clone()
    editable = model.get_non_special_symbol_mask(tokens, partial_masks=fixed)
    expected[editable] = 5
    # These operations independently establish the actual backbone's accepted ABI.
    assert model(tokens).shape == (2, 6, 9)
    assert model.energy(tokens, tokens).shape == (2, 1)
    torch.manual_seed(921)
    with torch.no_grad():
        result = model.generate(tokens, max_steps=steps, partial_masks=fixed, return_state=True)
    rng = torch.get_rng_state().clone()
    assert result['output_tokens'].dtype == dtype
    assert torch.equal(result['output_tokens'], expected)
    assert result['output_masks'].sum() == 0
    assert all(snapshot.dtype == dtype for snapshot in result['history'])
    assert model.proposal_model.seen_dtypes[1] == dtype
    if resample:
        assert len(model.proposal_model.seen_dtypes) > steps + 1
    control = make_generator(resample)
    torch.manual_seed(921)
    with torch.no_grad():
        reference = control.generate(
            tokens.long(), max_steps=steps, partial_masks=fixed, return_state=True
        )
    assert torch.equal(result['output_tokens'].long(), reference['output_tokens'])
    assert torch.equal(result['output_scores'], reference['output_scores'])
    assert torch.equal(rng, torch.get_rng_state())
