"""Exercise bounded reranking through the public training and generation paths."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import EnergyModel, QDiffusion, QDiffusionConfig, SequenceTokenSpec


class LocalProposal(nn.Module):
    def forward(self, tokens):
        logits = torch.zeros((*tokens.shape, 8), device=tokens.device)
        logits[..., 5] = 1.0
        return logits


class RecordingEnergy(EnergyModel):
    def __init__(self):
        super().__init__()
        self.calls = []

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        self.calls.append((noisy_tokens.clone(), candidate_tokens.clone(), attention_mask.clone()))
        return (candidate_tokens * attention_mask).sum(-1, keepdim=True).float()


def build_generator(limit):
    config = QDiffusionConfig(num_candidates=3, disable_resample=True)
    config.energy_batch_size = limit
    spec = SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4)
    energy = RecordingEnergy()
    return QDiffusion(LocalProposal(), energy, spec, config), energy


def test_reranking_caps_the_scorer_batch_and_preserves_order():
    tokens = torch.tensor([[1, 6, 2, 0], [1, 5, 6, 2], [1, 6, 7, 2]])
    reference, reference_energy = build_generator(None)
    chunked, chunked_energy = build_generator(4)

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(42)
        expected = reference.generate(tokens, max_steps=2)
        torch.default_generator.manual_seed(42)
        result = chunked.generate(tokens, max_steps=2)

    torch.testing.assert_close(result, expected)
    assert [len(noisy) for noisy, _, _ in chunked_energy.calls] == [4, 4, 1] * 2
    for expected_rows, chunked_rows in zip(zip(*reference_energy.calls), zip(*chunked_energy.calls)):
        torch.testing.assert_close(torch.cat(chunked_rows), torch.cat(expected_rows))


def test_training_keeps_single_energy_calls_for_its_statistics():
    generator, energy = build_generator(2)
    tokens = torch.tensor([[1, 6, 2], [1, 7, 2]])

    generator.objective({"targets": tokens})

    assert [len(noisy) for noisy, _, _ in energy.calls] == [2, 6]


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_generation_rejects_invalid_energy_batch_size(limit):
    generator, _ = build_generator(limit)
    with pytest.raises(ValueError, match="energy_batch_size"):
        generator.generate(torch.tensor([[1, 6, 2]]), max_steps=1)
