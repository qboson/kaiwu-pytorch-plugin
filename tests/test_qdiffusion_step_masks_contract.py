"""step() must prefer caller-supplied partial masks over the stored ones."""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class _Tokenizer:  # pylint: disable=too-few-public-methods
    def batch_decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(int(t)) for t in row) for row in tokens]


class _Proposal(nn.Module):
    def __init__(self, vocab=8, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, **kwargs):
        del kwargs
        return self.head(self.embedding(input_ids))


class _Energy(EnergyModel):
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        return candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)


def _model() -> QDiffusion:
    return QDiffusion(
        proposal_model=_Proposal(),
        energy_model=_Energy(),
        token_spec=SequenceTokenSpec(
            pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=None, tokenizer=_Tokenizer()
        ),
        config=QDiffusionConfig(num_diffusion_timesteps=4, disable_resample=True),
        freeze_proposal=False,
    )


def test_passed_masks_override_the_stored_masks() -> None:
    torch.manual_seed(0)
    model = _model()
    tokens = torch.tensor([[1, 5, 6, 2, 0]])
    state = model.initialize_state(tokens, max_steps=2)
    initial = state["output_tokens"].clone()
    masks = torch.ones(1, 5, dtype=torch.bool)

    stepped = model.step(state, partial_masks=masks)

    assert torch.equal(stepped["output_tokens"], initial)


def test_stored_masks_are_used_when_none_is_passed() -> None:
    torch.manual_seed(0)
    model = _model()
    tokens = torch.tensor([[1, 5, 6, 2, 0]])
    masks = torch.ones(1, 5, dtype=torch.bool)
    state = model.initialize_state(tokens, partial_masks=masks, max_steps=2)

    stepped = model.step(state)

    assert torch.equal(stepped["output_tokens"], tokens)
