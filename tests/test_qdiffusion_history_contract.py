"""Pin the decode-history frames recorded by initialize_state and step."""

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


def test_history_starts_with_the_initial_tokens() -> None:
    tokens = torch.tensor([[1, 3, 3, 2, 0]])
    state = _model().initialize_state(tokens, max_steps=3)

    assert len(state["history"]) == 1
    assert torch.equal(state["history"][0], state["output_tokens"])


def test_each_step_appends_one_independent_frame() -> None:
    torch.manual_seed(0)
    model = _model()
    state = model.initialize_state(torch.tensor([[1, 3, 3, 2, 0]]), max_steps=3)
    initial = state["history"][0].clone()

    for expected_len in (2, 3, 4):
        state = model.step(state)
        assert len(state["history"]) == expected_len
        assert torch.equal(state["history"][0], initial)

    state["history"][-1].fill_(0)
    assert torch.equal(state["history"][0], initial)
