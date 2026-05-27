"""Offline unit tests for generic QDiffusion construction."""

import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin.qdiffusion import (
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class DummyTokenizer:
    """Minimal tokenizer stub used by direct-construction tests."""

    def batch_decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(int(token)) for token in row) for row in tokens]


class DummyProposalModel(nn.Module):
    """Tiny proposal model that returns logits over one toy vocabulary."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        del kwargs
        hidden = torch.tanh(self.hidden(self.embedding(input_ids)))
        return self.lm_head(hidden)


class DummyEnergyModel(nn.Module):
    """Tiny energy backbone whose parameters are optimized by QDiffusion."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)


class DummyEnergyAdapter:
    """Minimal generic adapter that bridges the toy energy backbone."""

    def __init__(self, energy_model: DummyEnergyModel) -> None:
        self.energy_model = energy_model

    @property
    def hidden_size(self) -> int:
        return self.energy_model.embedding.embedding_dim

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.energy_model.embedding(tokens)

    def encode_conditioned(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids, attention_mask
        return torch.tanh(self.energy_model.hidden(inputs_embeds))


class TestQDiffusionDummy(unittest.TestCase):
    """Exercises direct QDiffusion construction without DPLM dependencies."""

    def setUp(self):
        vocab_size = 8
        hidden_size = 12
        self.proposal_model = DummyProposalModel(vocab_size=vocab_size, hidden_size=hidden_size)
        self.energy_model = DummyEnergyModel(vocab_size=vocab_size, hidden_size=hidden_size)
        self.energy_adapter = DummyEnergyAdapter(self.energy_model)
        self.token_spec = SequenceTokenSpec(
            pad_id=0,
            bos_id=1,
            eos_id=2,
            mask_id=3,
            x_id=4,
            tokenizer=DummyTokenizer(),
        )
        self.config = QDiffusionConfig(
            num_diffusion_timesteps=8,
            num_candidates=2,
            proposal_temperature=0.0,
            disable_resample=True,
        )
        self.model = QDiffusion(
            proposal_model=self.proposal_model,
            energy_model=self.energy_model,
            token_spec=self.token_spec,
            energy_adapter=self.energy_adapter,
            config=self.config,
            freeze_proposal=False,
        )
        self.targets = torch.tensor(
            [
                [1, 5, 6, 2, 0],
                [1, 6, 5, 2, 0],
            ],
            dtype=torch.long,
        )

    def test_forward_shape(self):
        logits = self.model.forward(self.targets)
        self.assertEqual(logits.shape, (2, 5, 8))

    def test_energy_shape(self):
        energy = self.model.energy(self.targets, self.targets)
        self.assertEqual(energy.shape, (2, 1))

    def test_objective_fields(self):
        outputs = self.model.objective({"targets": self.targets})
        self.assertEqual(outputs["logits"].shape, (2, 5, 8))
        self.assertEqual(outputs["targets"].shape[1], 5)
        self.assertEqual(outputs["loss_mask"].dtype, torch.bool)
        self.assertEqual(outputs["weight"].shape, (2, 1))
        self.assertEqual(outputs["objective_ebm"].shape, (2, 1))

    def test_generate_one_step(self):
        generated = self.model.generate(self.targets, max_steps=1)
        self.assertEqual(generated.shape, self.targets.shape)

    def test_removed_dplm_entrypoints(self):
        self.assertFalse(hasattr(QDiffusion, "from_pretrained"))
        self.assertFalse(hasattr(QDiffusion, "build"))
        self.assertFalse(hasattr(QDiffusion, "load_backbone"))


if __name__ == "__main__":
    unittest.main()
