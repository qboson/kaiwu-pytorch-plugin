"""Regression test for the decode temperature of ``QDiffusion``.

``QDiffusion.generate(temperature=...)`` / ``initialize_state(temperature=...)``
documented a sampling temperature and stored it in the decode state, but the
decode path never read it: candidate sampling always used
``config.proposal_temperature``, so ``temperature=0.01`` and ``temperature=99.0``
produced identical output (issue #313).

The fix threads the state temperature into candidate sampling and falls back to
``config.proposal_temperature`` when the caller does not pass one, so the
previous default behaviour is preserved.
"""

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin.qdiffusion import (  # noqa: E402
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class _Proposal(nn.Module):
    def __init__(self, vocab_size=8, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        del kwargs
        return self.head(self.embedding(input_ids))


class _Energy(EnergyModel):
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del attention_mask
        return candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)


def _build(proposal_temperature, seed=0):
    torch.manual_seed(seed)
    model = QDiffusion(
        _Proposal(),
        _Energy(),
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2),
        config=QDiffusionConfig(
            proposal_temperature=proposal_temperature,
            proposal_noise_scale=1.0,
            num_candidates=4,
            disable_resample=True,
        ),
    )
    model.eval()
    return model


TOKENS = torch.tensor([[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]])


def _generate(model, **kwargs):
    torch.manual_seed(1234)
    with torch.no_grad():
        return model.generate(TOKENS, max_steps=3, **kwargs)


def test_decode_temperature_has_an_effect():
    model = _build(proposal_temperature=1.0)

    cold = _generate(model, temperature=0.01)
    hot = _generate(model, temperature=99.0)

    assert not torch.equal(cold, hot), (
        "an explicit decode temperature must change proposal sampling"
    )


def test_explicit_temperature_overrides_the_config():
    greedy = _build(proposal_temperature=0.0)
    stochastic = _build(proposal_temperature=1.0)

    config_driven = _generate(greedy)  # config 0.0 -> greedy
    overridden = _generate(stochastic, temperature=0.0)

    assert torch.equal(config_driven, overridden)


def test_default_keeps_using_the_config_temperature():
    model = _build(proposal_temperature=1.0)

    default = _generate(model)
    explicit = _generate(model, temperature=1.0)

    assert torch.equal(default, explicit)
