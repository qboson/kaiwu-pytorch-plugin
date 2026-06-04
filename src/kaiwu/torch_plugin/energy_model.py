"""BM-backed energy models for conditioned scoring."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .full_boltzmann_machine import BoltzmannMachine


class EnergyModel(nn.Module):
    """Energy-side model interface with optional BM visible-logit scoring."""

    def __init__(
        self,
        bm_num_visible: int | None = None,
        bm_num_hidden: int | None = None,
        sampler: Any | None = None,
        visible_threshold: float = 0.5,
        use_straight_through: bool = True,
    ) -> None:
        super().__init__()
        self.visible_threshold = visible_threshold
        self.use_straight_through = use_straight_through
        self.bm_num_visible = bm_num_visible
        self.bm_num_hidden = bm_num_hidden
        self.sampler = sampler
        self._last_stats: dict[str, torch.Tensor] = {}
        if bm_num_visible is not None and bm_num_hidden is not None:
            self.energy_bm = BoltzmannMachine(num_nodes=bm_num_visible + bm_num_hidden)

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Scores candidates conditioned on noisy tokens."""
        del noisy_tokens, candidate_tokens, attention_mask
        raise NotImplementedError(
            "EnergyModel subclasses must implement score_conditioned()."
        )

    def discretize_visible_state(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Converts visible logits into the binary state consumed by the sampler."""
        visible_probs = torch.sigmoid(visible_logits)
        hard_visible = (visible_probs >= self.visible_threshold).to(visible_probs.dtype)
        if self.use_straight_through:
            return hard_visible + visible_probs - visible_probs.detach()
        return hard_visible

    def sample_hidden_state(
        self,
        visible_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples BM hidden states for each visible assignment."""
        if not hasattr(self, "energy_bm") or self.sampler is None:
            raise RuntimeError(
                "BM hidden-state sampling requires bm_num_visible, "
                "bm_num_hidden, and sampler."
            )
        batched_states = []
        split_sizes = []
        for sample_index in range(visible_state.size(0)):
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

    def score_visible_logits(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Scores visible logits under the conditioned BM energy model."""
        if self.bm_num_visible is None:
            raise RuntimeError("BM visible-logit scoring requires bm_num_visible.")
        visible_state = self.discretize_visible_state(visible_logits)
        full_states, split_sizes = self.sample_hidden_state(visible_state)
        hidden_state = full_states[:, self.bm_num_visible :]
        self._set_last_stats(
            visible_state=full_states[:, : self.bm_num_visible],
            hidden_state=hidden_state,
        )
        flat_energy = self.energy_bm(full_states).unsqueeze(-1)
        split_energy = torch.split(flat_energy, split_sizes.tolist())
        return torch.stack([energy.mean(dim=0) for energy in split_energy], dim=0)
