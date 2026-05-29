"""Energy-side encoders, rerankers, and adapters for DPLM examples."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine

from .backbone import DPLMBackbone


class DPLMFeatureEncoder(nn.Module):
    """Sequence-feature encoder backed by one DPLM backbone."""

    def __init__(self, backbone: DPLMBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.net.config.hidden_size)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone.net.get_input_embeddings()(tokens)

    def encode_tokens(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone.net(input_ids=tokens, attention_mask=attention_mask)
        return outputs["last_hidden_state"]


class RBMConditionedEnergyModel(nn.Module):
    """Conditioned RBM reranker backed by one DPLM feature encoder."""

    def __init__(
        self,
        encoder: DPLMFeatureEncoder,
        rbm_num_visible: int,
        rbm_num_hidden: int,
        feature_pooling: str = "mean",
    ) -> None:
        super().__init__()
        if feature_pooling != "mean":
            raise ValueError(
                f"Unsupported RBM feature_pooling for DPLM examples: {feature_pooling}"
            )

        self.encoder = encoder
        self.feature_pooling = feature_pooling
        self.feature_projector = nn.Linear(2 * encoder.hidden_size, rbm_num_visible)
        self.energy_rbm = RestrictedBoltzmannMachine(
            num_visible=rbm_num_visible,
            num_hidden=rbm_num_hidden,
        )

    @property
    def hidden_size(self) -> int:
        return self.encoder.hidden_size

    @property
    def tokenizer(self) -> Any:
        return self.encoder.backbone.tokenizer

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed_tokens(tokens)

    def encode_conditioned(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder.backbone.net(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs["last_hidden_state"]

    def _masked_mean_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denominator

    def build_visible_state(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        noisy_hidden = self.encoder.encode_tokens(
            noisy_tokens, attention_mask=attention_mask
        )
        candidate_hidden = self.encoder.encode_tokens(
            candidate_tokens, attention_mask=attention_mask
        )
        noisy_features = self._masked_mean_pool(noisy_hidden, attention_mask)
        candidate_features = self._masked_mean_pool(candidate_hidden, attention_mask)
        conditioned_features = torch.cat([noisy_features, candidate_features], dim=-1)
        return torch.sigmoid(self.feature_projector(conditioned_features))


class RBMConditionedEnergyAdapter:
    """Adapter exposing one DPLM-conditioned RBM scorer to ``QDiffusion``."""

    def __init__(self, energy_model: RBMConditionedEnergyModel) -> None:
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
        visible_state = self.energy_model.build_visible_state(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
        rbm_state = self.energy_model.energy_rbm.get_hidden(
            visible_state,
            requires_grad=True,
            bernoulli=False,
        )
        return self.energy_model.energy_rbm(rbm_state).unsqueeze(-1)


class BMConditionedEnergyModel(nn.Module):
    """Conditioned BM reranker backed by one DPLM feature encoder."""

    def __init__(
        self,
        encoder: DPLMFeatureEncoder,
        bm_num_visible: int,
        bm_num_hidden: int,
        feature_pooling: str = "mean",
        mean_field_steps: int = 3,
    ) -> None:
        super().__init__()
        if feature_pooling != "mean":
            raise ValueError(
                f"Unsupported BM feature_pooling for DPLM examples: {feature_pooling}"
            )

        self.encoder = encoder
        self.feature_pooling = feature_pooling
        self.mean_field_steps = mean_field_steps
        self.bm_num_visible = bm_num_visible
        self.bm_num_hidden = bm_num_hidden
        self.feature_projector = nn.Linear(2 * encoder.hidden_size, bm_num_visible)
        self.energy_bm = BoltzmannMachine(num_nodes=bm_num_visible + bm_num_hidden)
        self.hidden_init = nn.Linear(2 * encoder.hidden_size, bm_num_hidden)

    @property
    def hidden_size(self) -> int:
        return self.encoder.hidden_size

    @property
    def tokenizer(self) -> Any:
        return self.encoder.backbone.tokenizer

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed_tokens(tokens)

    def encode_conditioned(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder.backbone.net(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs["last_hidden_state"]

    def _masked_mean_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denominator

    def build_conditioned_features(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        noisy_hidden = self.encoder.encode_tokens(
            noisy_tokens, attention_mask=attention_mask
        )
        candidate_hidden = self.encoder.encode_tokens(
            candidate_tokens, attention_mask=attention_mask
        )
        noisy_features = self._masked_mean_pool(noisy_hidden, attention_mask)
        candidate_features = self._masked_mean_pool(candidate_hidden, attention_mask)
        return torch.cat([noisy_features, candidate_features], dim=-1)

    def build_visible_state(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        conditioned_features = self.build_conditioned_features(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
        return torch.sigmoid(self.feature_projector(conditioned_features))

    def infer_hidden_state(
        self, visible_state: torch.Tensor, conditioned_features: torch.Tensor
    ) -> torch.Tensor:
        hidden_state = torch.sigmoid(self.hidden_init(conditioned_features))
        q_coef = self.energy_bm.symmetrized_quadratic_coef()
        visible_to_hidden = q_coef[
            : self.bm_num_visible, self.bm_num_visible :
        ]
        hidden_to_hidden = q_coef[
            self.bm_num_visible :, self.bm_num_visible :
        ]
        hidden_bias = self.energy_bm.hidden_bias(self.bm_num_hidden)

        for _ in range(self.mean_field_steps):
            hidden_state = torch.sigmoid(
                visible_state @ visible_to_hidden
                + hidden_state @ hidden_to_hidden
                + hidden_bias
            )
        return hidden_state


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
        conditioned_features = self.energy_model.build_conditioned_features(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
        visible_state = torch.sigmoid(
            self.energy_model.feature_projector(conditioned_features)
        )
        hidden_state = self.energy_model.infer_hidden_state(
            visible_state=visible_state,
            conditioned_features=conditioned_features,
        )
        full_state = torch.cat([visible_state, hidden_state], dim=-1)
        return self.energy_model.energy_bm(full_state).unsqueeze(-1)
