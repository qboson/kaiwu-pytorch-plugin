"""BM-backed conditioned rerankers for DPLM examples."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kaiwu.torch_plugin import BoltzmannMachine

from .common import DPLMFeatureEncoder, build_conditioned_features
from .sampler import build_bm_sampler


class BMConditionedEnergyModel(nn.Module):
    """Conditioned BM reranker backed by one DPLM feature encoder."""

    def __init__(
        self,
        encoder: DPLMFeatureEncoder,
        bm_num_visible: int,
        bm_num_hidden: int,
        feature_pooling: str = "mean",
        sampler: Any | None = None,
        sampler_type: str = "sa",
        sampler_kwargs: dict[str, Any] | None = None,
        visible_threshold: float = 0.5,
        use_straight_through: bool = True,
    ) -> None:
        super().__init__()
        if feature_pooling != "mean":
            raise ValueError(
                f"Unsupported BM feature_pooling for DPLM examples: {feature_pooling}"
            )

        self.encoder = encoder
        self.feature_pooling = feature_pooling
        self.visible_threshold = visible_threshold
        self.use_straight_through = use_straight_through
        self.bm_num_visible = bm_num_visible
        self.bm_num_hidden = bm_num_hidden
        self.feature_projector = nn.Linear(2 * encoder.hidden_size, bm_num_visible)
        self.energy_bm = BoltzmannMachine(num_nodes=bm_num_visible + bm_num_hidden)
        self.sampler_type = sampler_type
        self.sampler_kwargs = dict(sampler_kwargs or {})
        self.sampler = sampler or build_bm_sampler(
            sampler_type=sampler_type,
            sampler_kwargs=self.sampler_kwargs,
        )
        self._last_stats: dict[str, torch.Tensor] = {}

    @property
    def hidden_size(self) -> int:
        return self.encoder.hidden_size

    @property
    def tokenizer(self) -> Any:
        return self.encoder.tokenizer

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed_tokens(tokens)

    def encode_conditioned(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder.encode_conditioned(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

    def build_conditioned_features(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Keep this helper explicit because it is the semantic bridge between
        # sequence modeling and BM scoring: everything after this point operates
        # on BM-space states rather than protein tokens.
        return build_conditioned_features(
            self.encoder,
            noisy_tokens,
            candidate_tokens,
            attention_mask,
        )

    def build_visible_logits(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # The projector turns one concatenated sequence feature into the visible
        # part of the BM state before discretization/sampling steps.
        conditioned_features = self.build_conditioned_features(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
        return self.feature_projector(conditioned_features)

    def build_visible_state(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns visible probabilities for inspection/debugging."""
        return torch.sigmoid(
            self.build_visible_logits(
                noisy_tokens=noisy_tokens,
                candidate_tokens=candidate_tokens,
                attention_mask=attention_mask,
            )
        )

    def discretize_visible_state(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Converts visible logits into the binary state consumed by the sampler."""
        visible_probs = torch.sigmoid(visible_logits)
        hard_visible = (visible_probs >= self.visible_threshold).to(visible_probs.dtype)
        if self.use_straight_through:
            # Forward uses hard 0/1 states so the sampler sees a discrete
            # problem, while backward follows the soft probabilities.
            return hard_visible + visible_probs - visible_probs.detach()
        return hard_visible

    def sample_hidden_state(
        self,
        visible_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples one batch of BM hidden states with the configured sampler."""
        batched_states = []
        split_sizes = []
        for sample_index in range(visible_state.size(0)):
            # The sampler works one visible assignment at a time and may return
            # multiple full-state candidates for that single input example.
            sampled_states = self.energy_bm.condition_sample(
                self.sampler,
                visible_state[sample_index : sample_index + 1],
                dtype=visible_state.dtype,
            )
            batched_states.append(sampled_states)
            split_sizes.append(sampled_states.size(0))
        return torch.cat(batched_states, dim=0), torch.tensor(
            split_sizes,
            device=visible_state.device,
            dtype=torch.long,
        )

    def _set_last_stats(
        self,
        *,
        visible_state: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        self._last_stats = {
            "sampling_mode": torch.tensor(
                1.0,
                dtype=visible_state.dtype,
                device=visible_state.device,
            ),
            "visible_on_ratio": visible_state.detach().mean(),
            "hidden_on_ratio": hidden_state.detach().mean(),
        }

    def get_last_stats(self) -> dict[str, torch.Tensor]:
        """Returns lightweight sampler diagnostics from the last score call."""
        return dict(self._last_stats)

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scores candidate sequences under the conditioned BM energy model."""
        visible_logits = self.build_visible_logits(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
        visible_state = self.discretize_visible_state(visible_logits)
        # From here on we are in sampler-backed BM mode: hidden states come from
        # SA/CIM rather than from a differentiable mean-field approximation.
        full_states, split_sizes = self.sample_hidden_state(visible_state)
        hidden_state = full_states[:, self.bm_num_visible :]
        self._set_last_stats(
            visible_state=full_states[:, : self.bm_num_visible],
            hidden_state=hidden_state,
        )
        flat_energy = self.energy_bm(full_states).unsqueeze(-1)
        # A sampler may emit multiple states per input example, so we aggregate
        # them back to one scalar score per original candidate.
        split_energy = torch.split(flat_energy, split_sizes.tolist())
        return torch.stack([energy.mean(dim=0) for energy in split_energy], dim=0)


class BMConditionedEnergyAdapter:
    """Adapter exposing one DPLM-conditioned BM scorer to ``QDiffusion``."""

    def __init__(self, energy_model: BMConditionedEnergyModel) -> None:
        self.energy_model = energy_model

    @property
    def hidden_size(self) -> int:
        return self.energy_model.hidden_size

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.energy_model.embed_tokens(tokens)

    def encode_conditioned(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.energy_model.encode_conditioned(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.energy_model.score_conditioned(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )

    def get_last_stats(self) -> dict[str, torch.Tensor]:
        return self.energy_model.get_last_stats()
