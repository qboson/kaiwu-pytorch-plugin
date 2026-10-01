"""Repetition decisions depend on editable content, independently of fixed context."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel, QDiffusion, QDiffusionConfig, SequenceTokenSpec,
)


class ContextProposal(nn.Module):
    """A neural proposal refines repeated tokens once other content is visible."""

    def __init__(self, preferences, context_start=1, context_stop=None):
        super().__init__()
        self.embedding = nn.Embedding(40, 1)
        self.positions = nn.Embedding(24, 40)
        self.head = nn.Linear(1, 40, bias=False)
        self.context_start = context_start
        self.context_stop = context_stop
        self.calls = []
        choices = [10] + preferences + [11]
        choices += list(range(12, 12 + 24 - len(choices)))
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[5:] = 1
            self.positions.weight.fill_(-100)
            for index, preferred in enumerate(choices):
                self.positions.weight[index, preferred] = 80
            self.head.weight.zero_()
            self.head.weight[8, 0] = 240

    def forward(self, tokens):
        context = self.embedding(tokens[:, self.context_start:self.context_stop]).sum(1)
        logits = (self.positions(torch.arange(tokens.size(1))).unsqueeze(0)
                  + self.head(context).unsqueeze(1))
        self.calls.append((tokens.detach().clone(), logits.detach().clone()))
        return logits


class NeuralEnergy(EnergyModel):
    """Real embedding/head scoring uses the complete candidate and its PAD mask."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(40, 2)
        self.head = nn.Linear(2, 1)
        self.calls = []
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(80).reshape(40, 2) / 100)
            self.head.weight.fill_(.1)
            self.head.bias.zero_()

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        self.calls.append((candidate_tokens.detach().clone(), attention_mask.clone()))
        pooled = ((self.embedding(candidate_tokens) + self.embedding(noisy_tokens))
                  * attention_mask.unsqueeze(-1)).sum(1)
        return self.head(pooled)


def _generator(preferences, ratio=.4, disable=False, context_start=1, context_stop=None):
    proposal = ContextProposal(preferences, context_start, context_stop)
    energy = NeuralEnergy()
    model = QDiffusion(proposal, energy,
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
        QDiffusionConfig(num_candidates=1, proposal_temperature=0,
            proposal_noise_scale=0, resample_ratio=ratio, disable_resample=disable))
    return model, proposal, energy


def _public_step(model, inputs, api, protected=None):
    if api == "generate":
        return model.generate(inputs, partial_masks=protected, max_steps=1, return_state=True)
    return model.step(model.initialize_state(inputs, partial_masks=protected, max_steps=1))


@pytest.mark.parametrize("api", ["generate", "step"])
@pytest.mark.parametrize("padding", [0, 2, 8])
def test_repeated_editable_content_refines_independently_of_padding(api, padding):
    """Three copies among five editable tokens exceed .4 with any PAD suffix."""
    torch.manual_seed(41)
    model, proposal, energy = _generator([5, 5, 5, 6, 7])
    inputs = torch.tensor([[1, 5, 5, 5, 6, 7, 2] + [0] * padding])
    original = inputs.clone()

    state = _public_step(model, inputs, api)

    assert len(proposal.calls) == 2
    assert state["output_tokens"].tolist() == [[1, 8, 8, 8, 6, 7, 2] + [0] * padding]
    assert proposal.calls[1][0].tolist() == [[1, 3, 3, 3, 6, 7, 2] + [0] * padding]
    assert torch.isfinite(state["output_scores"]).all()
    torch.testing.assert_close(inputs, original)
    torch.testing.assert_close(energy.calls[0][0], original)
    torch.testing.assert_close(energy.calls[0][1], original.ne(0))


def test_neural_proposal_and_generated_prefix_are_padding_invariant():
    """The backbone prefix logits match exactly, so padding cannot alter refinement."""
    outputs = []
    initial_logits = []
    for padding in (0, 8):
        model, proposal, _ = _generator([5, 5, 5, 6, 7])
        inputs = torch.tensor([[1, 5, 5, 5, 6, 7, 2] + [0] * padding])
        state = model.generate(inputs, max_steps=1, return_state=True)
        outputs.append(state["output_tokens"][:, :7])
        initial_logits.append(proposal.calls[0][1][:, :7])
    torch.testing.assert_close(initial_logits[0], initial_logits[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[1])


@pytest.mark.parametrize("api", ["generate", "step"])
def test_protected_common_prefix_does_not_make_unique_suffix_repetitive(api):
    """Editable [5,6,7] has maximum frequency one, regardless of three fixed 5s."""
    model, proposal, _ = _generator([5, 5, 5, 5, 6, 7], context_start=4, context_stop=7)
    inputs = torch.tensor([[1, 5, 5, 5, 5, 6, 7, 2]])
    protected = torch.tensor([[False, True, True, True, False, False, False, False]])

    state = _public_step(model, inputs, api, protected)

    assert len(proposal.calls) == 1
    torch.testing.assert_close(state["output_tokens"], inputs)


@pytest.mark.parametrize("ratio", [.49, .5])
@pytest.mark.parametrize("padding", [0, 8])
def test_repetition_ratio_keeps_strict_threshold_semantics(ratio, padding):
    """Two copies among four editable tokens trigger below .5, but not at .5."""
    model, proposal, _ = _generator([5, 5, 6, 7], ratio=ratio)
    inputs = torch.tensor([[1, 5, 5, 6, 7, 2] + [0] * padding])

    state = model.generate(inputs, max_steps=1, return_state=True)

    assert len(proposal.calls) == (2 if ratio < .5 else 1)
    expected = [8, 8, 6, 7] if ratio < .5 else [5, 5, 6, 7]
    assert state["output_tokens"].tolist() == [[1] + expected + [2] + [0] * padding]


@pytest.mark.parametrize("padding", [0, 8])
def test_singleton_editable_content_is_not_repetition(padding):
    model, proposal, _ = _generator([5], ratio=.25)
    inputs = torch.tensor([[1, 5, 2] + [0] * padding])

    state = model.generate(inputs, max_steps=1, return_state=True)

    assert len(proposal.calls) == 1
    torch.testing.assert_close(state["output_tokens"], inputs)


def test_unique_editable_tokens_do_not_repeat_even_below_singleton_ratio():
    model, proposal, _ = _generator([5, 6, 7], ratio=.1)
    inputs = torch.tensor([[1, 5, 6, 7, 2]])

    state = model.generate(inputs, max_steps=1, return_state=True)

    assert len(proposal.calls) == 1
    torch.testing.assert_close(state["output_tokens"], inputs)


@pytest.mark.parametrize("ordinary_fixed", [False, True])
def test_empty_editable_region_has_no_repetition_work(ordinary_fixed):
    model, proposal, _ = _generator([5, 5], ratio=.1)
    inputs = (torch.tensor([[1, 5, 5, 2, 0, 0]]) if ordinary_fixed
              else torch.tensor([[1, 2, 0, 0, 0, 0]]))
    protected = torch.ones_like(inputs, dtype=torch.bool) if ordinary_fixed else None

    state = model.generate(inputs, partial_masks=protected, max_steps=1, return_state=True)

    assert len(proposal.calls) == 1
    torch.testing.assert_close(state["output_tokens"], inputs)


def test_multiple_repeated_types_only_mask_their_editable_occurrences():
    """Each of 5 and 6 occupies two of five editable positions, exceeding .3."""
    model, proposal, _ = _generator([5, 5, 6, 6, 7], ratio=.3)
    inputs = torch.tensor([[1, 5, 5, 6, 6, 7, 2, 0, 0, 0, 0]])

    state = model.generate(inputs, max_steps=1, return_state=True)

    assert len(proposal.calls) == 2
    assert proposal.calls[1][0].tolist() == [[1, 3, 3, 3, 3, 7, 2, 0, 0, 0, 0]]
    assert state["output_tokens"].tolist() == [[1, 8, 8, 8, 8, 7, 2, 0, 0, 0, 0]]


def test_mixed_length_batch_uses_each_samples_editable_length():
    """The short row repeats 3/5; the long row repeats only 3/13 and stays unchanged."""
    content = [5, 5, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]
    model, proposal, _ = _generator(content)
    inputs = torch.tensor([[1, 5, 5, 5, 6, 7, 2] + [0] * 8, [1] + content + [2]])

    state = model.generate(inputs, max_steps=1, return_state=True)

    assert len(proposal.calls) == 2
    assert proposal.calls[1][0].shape[0] == 1
    assert state["output_tokens"][0].tolist() == [1, 8, 8, 8, 6, 7, 2] + [0] * 8
    torch.testing.assert_close(state["output_tokens"][1], inputs[1])


@pytest.mark.parametrize("padding", [0, 8])
def test_disabling_resampling_keeps_original_proposal(padding):
    model, proposal, _ = _generator([5, 5, 5, 6, 7], disable=True)
    inputs = torch.tensor([[1, 5, 5, 5, 6, 7, 2] + [0] * padding])

    state = model.generate(inputs, max_steps=1, return_state=True)

    assert len(proposal.calls) == 1
    torch.testing.assert_close(state["output_tokens"], inputs)
