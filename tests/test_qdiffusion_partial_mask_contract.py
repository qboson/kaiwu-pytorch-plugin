"""Pin the partial-mask contract of the Q-Diffusion decode loop."""

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
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del attention_mask
        return (
            candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)
            - noisy_tokens.to(torch.float32).sum(dim=1, keepdim=True)
        )


def _model() -> QDiffusion:
    return QDiffusion(
        proposal_model=_Proposal(),
        energy_model=_Energy(),
        token_spec=SequenceTokenSpec(
            pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=None, tokenizer=_Tokenizer()
        ),
        config=QDiffusionConfig(num_diffusion_timesteps=8, disable_resample=True),
        freeze_proposal=False,
    )


def test_generate_keeps_partial_mask_positions_fixed() -> None:
    torch.manual_seed(0)
    tokens = torch.tensor([[1, 5, 6, 2, 0]])
    masks = torch.tensor([[True, False, True, True, True]])
    fixed = torch.tensor([1, 6, 2, 0])

    generated = _model().generate(tokens, max_steps=4, partial_masks=masks)
    assert torch.equal(generated[0, [0, 2, 3, 4]], fixed)


def test_each_step_keeps_partial_mask_positions_fixed() -> None:
    torch.manual_seed(0)
    tokens = torch.tensor([[1, 5, 6, 2, 0]])
    masks = torch.tensor([[True, False, True, True, True]])
    fixed = torch.tensor([1, 6, 2, 0])

    model = _model()
    state = model.initialize_state(tokens, partial_masks=masks, max_steps=4)
    for _ in range(4):
        state = model.step(state, partial_masks=masks)
        assert torch.equal(state["output_tokens"][0, [0, 2, 3, 4]], fixed)
