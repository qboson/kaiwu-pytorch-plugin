"""Offline contract tests for the BM-only QDiffusion-NLP example."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from example.qdiffusion.nlp.models.bm import BMEnergyModel
from example.qdiffusion.nlp.models.features import ConditionedFeatureEncoder
from example.qdiffusion.nlp.models.proposal import MDLMBackbone
from example.qdiffusion.nlp.sampling import BMGuidedSampler
from example.qdiffusion.nlp.train_bm import (
    binary_nce_loss,
    build_wrapped_blocks,
    constant_warmup_factor,
    corrupt_tokens,
    sample_antithetic_times,
)


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(character) % 7 + 3 for character in text]


class FakeMaskedLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            time_conditioning=False,
            vocab_size=6,
            hidden_size=2,
        )
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        *,
        input_ids,
        timesteps,
        output_hidden_states,
        return_dict,
    ):
        assert timesteps.shape == (input_ids.size(0),)
        assert return_dict
        logits = torch.arange(
            self.config.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        ).view(1, 1, -1).expand(*input_ids.shape, -1)
        hidden_states = (
            torch.stack(
                [input_ids.float(), input_ids.float() + 1],
                dim=-1,
            ),
        )
        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states if output_hidden_states else None,
        )


class FakeConditionedEncoder(nn.Module):
    hidden_size = 2

    def build_conditioned_output_layer(self):
        return nn.Identity()

    def encode_conditioned_tokens(
        self,
        noisy_tokens,
        candidate_tokens,
        **kwargs,
    ):
        del kwargs
        return torch.stack(
            [noisy_tokens.float(), candidate_tokens.float()],
            dim=-1,
        )


class FakeSampler:
    def solve(self, ising_matrix):
        return np.ones((2, ising_matrix.shape[0]), dtype=np.float32)


class RecordingBMEnergy(BMEnergyModel):
    def score_visible_logits(self, visible_logits):
        self.last_visible_logits = visible_logits
        return visible_logits.sum(dim=-1, keepdim=True)


class FixedProposal(nn.Module):
    def __init__(self, mask_id: int = 3, vocab_size: int = 4) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.vocab_size = vocab_size

    def forward(self, tokens):
        probabilities = torch.zeros(
            *tokens.shape,
            self.vocab_size,
            device=tokens.device,
        )
        masked = tokens.eq(self.mask_id)
        probabilities[..., 0] = torch.where(
            masked,
            torch.tensor(0.75, device=tokens.device),
            probabilities[..., 0],
        )
        probabilities[..., 1] = torch.where(
            masked,
            torch.tensor(0.25, device=tokens.device),
            probabilities[..., 1],
        )
        for token_id in range(self.mask_id):
            probabilities[..., token_id] = torch.where(
                tokens.eq(token_id),
                torch.ones_like(probabilities[..., token_id]),
                probabilities[..., token_id],
            )
        return probabilities.log()


class CandidateEnergy(nn.Module):
    def score_candidates_conditioned(
        self,
        noisy_tokens,
        candidate_tokens,
        attention_mask,
    ):
        del noisy_tokens, attention_mask
        return candidate_tokens.float().sum(dim=-1)


def test_proposal_adapter_returns_log_probabilities():
    proposal = MDLMBackbone(FakeMaskedLM(), FakeTokenizer(), mask_id=5)
    tokens = torch.tensor([[1, 5, 2]])

    log_probabilities = proposal(tokens)

    assert log_probabilities.shape == (1, 3, 6)
    assert torch.allclose(
        log_probabilities.exp().sum(dim=-1),
        torch.ones(1, 3),
    )


def test_wrapped_blocks_and_training_helpers():
    blocks = build_wrapped_blocks(
        ["ab", "cd"],
        FakeTokenizer(),
        sequence_length=4,
    )
    assert blocks
    assert all(block.shape == (4,) for block in blocks)
    assert all(block[0].item() == 1 and block[-1].item() == 2 for block in blocks)

    torch.manual_seed(7)
    timesteps = sample_antithetic_times(
        4,
        device=torch.device("cpu"),
        sampling_eps=1e-3,
    )
    assert timesteps.shape == (4,)
    assert torch.all((timesteps > 0) & (timesteps < 1))

    clean = torch.ones(4, 8, dtype=torch.long)
    noisy = corrupt_tokens(
        clean,
        timesteps,
        mask_id=5,
        noise_eps=1e-3,
    )
    assert noisy.shape == clean.shape
    assert set(noisy.unique().tolist()) <= {1, 5}
    assert binary_nce_loss(torch.zeros(2, 1), torch.zeros(2, 1)) > 0
    assert constant_warmup_factor(5, warmup_steps=10) == 0.5


def test_conditioned_features_and_continuous_bm_visible_values():
    encoder = FakeConditionedEncoder()
    feature_encoder = ConditionedFeatureEncoder(
        encoder,
        pooling_mode="mean",
    )
    noisy = torch.tensor([[1, 2, 3]])
    candidate = torch.tensor([[3, 2, 1]])

    features = feature_encoder(
        encoder,
        noisy,
        candidate,
        torch.ones_like(noisy, dtype=torch.bool),
    )
    assert features.shape == (1, 2)

    energy = RecordingBMEnergy(
        encoder,
        bm_num_visible=3,
        bm_num_hidden=2,
        sampler=FakeSampler(),
        pooling_mode="mean",
    )
    visible = torch.tensor([[0.25, -1.5, 2.0]])
    assert torch.equal(energy.discretize_visible_state(visible), visible)
    assert energy.checkpoint_metadata()["scoring_mode"] == "sampler"

    candidates = torch.tensor([[[1, 1, 1], [2, 2, 2]]])
    scores = energy.score_candidates_conditioned(
        noisy,
        candidates,
        torch.ones_like(candidates, dtype=torch.bool),
    )
    assert scores.shape == (1, 2)


def test_bm_guided_sampler_returns_fully_denoised_tokens():
    torch.manual_seed(11)
    sampler = BMGuidedSampler(
        FixedProposal(),
        energy_model=CandidateEnergy(),
        mask_id=3,
        num_candidates=2,
        remask_ratio=0.0,
    )
    masked = torch.full((2, 6), 3, dtype=torch.long)

    output = sampler.sample(masked, num_steps=4)

    assert output.shape == masked.shape
    assert not output.eq(3).any()
    assert sampler.last_stats["guided_steps"] > 0
