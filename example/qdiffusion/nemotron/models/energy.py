"""Contextual energy model used by the Nemotron example."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import EnergyModel


SA_BACKEND = "kaiwu.classical.SimulatedAnnealingOptimizer"
CHECKPOINT_FORMAT = "nemotron-contextual-energy-v2"


class ContextualEnergyModel(EnergyModel):
    """Score same-state token candidates with a contextual encoder and KPP BM.

    The transformer encodes the frozen Nemotron hidden state, the candidate
    token embedding, and the candidate change from the current noisy state.
    Its pooled output conditions the visible units of the public KPP BM.

    Args:
        proposal_hidden_dim: Feature width of frozen Nemotron hidden states.
        candidate_feature_dim: Feature width of Nemotron token embeddings.
        bm_num_visible: Number of visible units in the KPP BM.
        bm_num_hidden: Number of hidden units in the KPP BM.
        contextual_dim: Width of the contextual encoder.
        contextual_layers: Number of contextual encoder layers.
        contextual_heads: Number of attention heads per contextual layer.
        contextual_ffn_dim: Feed-forward width of contextual layers.
        contextual_dropout: Dropout rate in contextual layers.
        max_sequence_length: Maximum noisy-block length accepted by the model.
        sa_num_solutions: Number of hidden-state samples from Kaiwu SA.
        sa_seed: Random seed for Kaiwu SA.
        energy_num_lowest: Number of lowest-energy samples to average.
        sa_kwargs: Additional Kaiwu SA keyword arguments.
        sampler: Optional sampler override used by tests or custom callers.
        device: Device on which the energy model is placed.
    """

    checkpoint_format = CHECKPOINT_FORMAT

    def __init__(
        self,
        proposal_hidden_dim: int,
        candidate_feature_dim: int,
        *,
        bm_num_visible: int = 512,
        bm_num_hidden: int = 256,
        contextual_dim: int = 1024,
        contextual_layers: int = 4,
        contextual_heads: int = 16,
        contextual_ffn_dim: int = 4096,
        contextual_dropout: float = 0.0,
        max_sequence_length: int = 512,
        sa_num_solutions: int = 8,
        sa_seed: int = 20260855,
        energy_num_lowest: int = 3,
        sa_kwargs: dict[str, Any] | None = None,
        sampler: Any | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        positive = {
            "contextual_layers": contextual_layers,
            "contextual_dim": contextual_dim,
            "contextual_heads": contextual_heads,
            "contextual_ffn_dim": contextual_ffn_dim,
            "max_sequence_length": max_sequence_length,
            "bm_num_visible": bm_num_visible,
            "bm_num_hidden": bm_num_hidden,
            "sa_num_solutions": sa_num_solutions,
            "energy_num_lowest": energy_num_lowest,
        }
        invalid = sorted(name for name, value in positive.items() if value <= 0)
        if invalid:
            raise ValueError(f"hyperparameters must be positive, got: {invalid}")
        if contextual_dim % contextual_heads:
            # MultiheadAttention only raises AssertionError here, and
            # ``python -O`` strips asserts; raising ValueError up front keeps
            # the error type consistent.
            raise ValueError(
                "contextual_dim must be divisible by contextual_heads"
            )

        self.proposal_hidden_dim = proposal_hidden_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.contextual_dim = contextual_dim
        self.contextual_layers = contextual_layers
        self.contextual_heads = contextual_heads
        self.contextual_ffn_dim = contextual_ffn_dim
        self.contextual_dropout = contextual_dropout
        self.max_sequence_length = max_sequence_length
        self.sa_seed = sa_seed
        self.energy_num_lowest = energy_num_lowest

        sampler_kwargs = {
            "size_limit": sa_num_solutions,
            "rand_seed": sa_seed,
            "process_num": min(sa_num_solutions, 3),
            **(sa_kwargs or {}),
        }
        self.sa_kwargs = sampler_kwargs
        if sampler is None:
            sampler = SimulatedAnnealingOptimizer(**sampler_kwargs)
        super().__init__(bm_num_visible, bm_num_hidden, sampler=sampler)

        self.hidden_token_projector = nn.Linear(
            self.proposal_hidden_dim, self.contextual_dim
        )
        self.candidate_token_projector = nn.Linear(
            self.candidate_feature_dim, self.contextual_dim
        )
        self.pair_projector = nn.Sequential(
            nn.Linear(3 * self.contextual_dim, self.contextual_dim),
            nn.GELU(),
            nn.LayerNorm(self.contextual_dim),
        )
        self.position_embedding = nn.Embedding(
            self.max_sequence_length, self.contextual_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.contextual_dim,
            nhead=self.contextual_heads,
            dim_feedforward=self.contextual_ffn_dim,
            dropout=self.contextual_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.contextual_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.contextual_layers,
            enable_nested_tensor=False,
        )
        self.pool_score = nn.Linear(self.contextual_dim, 1)
        self.contextual_to_visible = nn.Linear(self.contextual_dim, int(bm_num_visible))
        self.to(device)
        # TODO: remove once fix/bm-device-sync (BM device attribute sync) is
        # merged and this branch is rebased onto it. nn.Module.to() moves this
        # submodule through _apply() without invoking the BM's own .to(), so
        # the plain-attribute device needs this manual sync until then.
        self.energy_bm.device = torch.device(device)

    def discretize_visible_state(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Keep continuous visible conditions, as in the KPP NLP workflow.

        Args:
            visible_logits: Contextual BM visible conditions.

        Returns:
            Unchanged visible conditions.
        """

        return visible_logits

    def build_visible_logits(
        self,
        hidden_states: torch.Tensor,
        noisy_features: torch.Tensor,
        candidate_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Map noisy-to-candidate context into BM-visible logits.

        Args:
            hidden_states: Frozen hidden states with shape ``[batch, seq, dim]``.
            noisy_features: Token features for the noisy block.
            candidate_features: Token features for a candidate block.
            attention_mask: Positions included in contextual pooling.

        Returns:
            BM-visible logits with shape ``[batch, bm_num_visible]``.

        Raises:
            ValueError: If input sequence shapes are inconsistent or too long.
        """

        valid_mask = attention_mask.bool()
        dtype = self.hidden_token_projector.weight.dtype
        hidden = self.hidden_token_projector(hidden_states.to(dtype))
        noisy = self.candidate_token_projector(noisy_features.to(dtype))
        candidate = self.candidate_token_projector(candidate_features.to(dtype))
        paired = self.pair_projector(
            torch.cat([hidden, candidate, candidate - noisy], dim=-1)
        )
        positions = torch.arange(paired.size(1), device=paired.device).unsqueeze(0)
        encoded = self.contextual_encoder(
            paired + self.position_embedding(positions),
            src_key_padding_mask=~valid_mask,
        )
        pool_logits = self.pool_score(encoded).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~valid_mask, float("-inf"))
        pooled = (encoded * torch.softmax(pool_logits, dim=-1).unsqueeze(-1)).sum(1)
        return self.contextual_to_visible(pooled)

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        hidden_states: torch.Tensor | None = None,
        noisy_features: torch.Tensor | None = None,
        candidate_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score candidates conditioned on their current noisy block.

        Args:
            noisy_tokens: Current noisy block token ids.
            candidate_tokens: Candidate token ids for the same block.
            attention_mask: Positions used by contextual pooling.
            hidden_states: Frozen Nemotron hidden states for the noisy block.
            noisy_features: Token features for the noisy block.
            candidate_features: Token features for the candidate block.

        Returns:
            One energy per candidate with shape ``[batch, 1]``.

        Raises:
            ValueError: If contextual inputs are absent or shapes do not match.
        """

        if (
            hidden_states is None
            or noisy_features is None
            or candidate_features is None
        ):
            raise ValueError("contextual scoring requires hidden and token features")
        if (
            noisy_tokens.shape != candidate_tokens.shape
            or attention_mask.shape != candidate_tokens.shape
            or noisy_features.shape[:2] != noisy_tokens.shape
            or candidate_features.shape[:2] != candidate_tokens.shape
        ):
            raise ValueError(
                "candidates, attention_mask, and token features must share the "
                "token shape"
            )
        if candidate_tokens.size(-1) > self.max_sequence_length:
            raise ValueError("candidate sequence exceeds max_sequence_length")
        changed = noisy_tokens.ne(candidate_tokens).unsqueeze(-1)
        candidate_features = torch.where(changed, candidate_features, noisy_features)
        visible = self.build_visible_logits(
            hidden_states,
            noisy_features,
            candidate_features,
            attention_mask,
        )
        return self.score_visible_logits(
            visible,
            num_lowest=self.energy_num_lowest,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the versioned configuration needed to rebuild this model.

        Returns:
            Serializable model and sampler configuration.
        """

        return {
            "checkpoint_format": self.checkpoint_format,
            "proposal_hidden_dim": self.proposal_hidden_dim,
            "candidate_feature_dim": self.candidate_feature_dim,
            "bm_num_visible": self.bm_num_visible,
            "bm_num_hidden": self.bm_num_hidden,
            "contextual_dim": self.contextual_dim,
            "contextual_layers": self.contextual_layers,
            "contextual_heads": self.contextual_heads,
            "contextual_ffn_dim": self.contextual_ffn_dim,
            "contextual_dropout": self.contextual_dropout,
            "max_sequence_length": self.max_sequence_length,
            "sa_num_solutions": self.sa_kwargs["size_limit"],
            "sa_seed": self.sa_seed,
            "energy_num_lowest": self.energy_num_lowest,
            "sa_kwargs": dict(self.sa_kwargs),
            "sampler_backend": SA_BACKEND,
            "visible_transform": "identity",
            "feature_mode": "noisy_candidate_contextual",
        }

    def compact_state_dict(self) -> dict[str, Any]:
        """Return only model state required by the versioned checkpoint.

        Returns:
            State dictionaries for the contextual encoder and KPP BM.
        """

        names = (
            "hidden_token_projector",
            "candidate_token_projector",
            "pair_projector",
            "position_embedding",
            "contextual_encoder",
            "pool_score",
            "contextual_to_visible",
            "energy_bm",
        )
        return {name: getattr(self, name).state_dict() for name in names}

    def load_compact_state_dict(self, state: dict[str, Any]) -> None:
        """Load a state dictionary produced by ``compact_state_dict``.

        Args:
            state: Contextual encoder and KPP BM state dictionaries.

        Raises:
            ValueError: If the checkpoint state has missing or extra modules.
        """

        expected = set(self.compact_state_dict())
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise ValueError(
                f"checkpoint state mismatch: missing={missing}, extra={extra}"
            )
        for name, value in state.items():
            getattr(self, name).load_state_dict(value)


def model_from_config(
    config: dict[str, Any], *, device: str | torch.device, sampler: Any | None = None
) -> ContextualEnergyModel:
    """Rebuild a contextual energy model from checkpoint configuration.

    Args:
        config: Serialized model configuration.
        device: Target device for the rebuilt model.
        sampler: Optional Kaiwu sampler override.

    Returns:
        Rebuilt contextual energy model.

    Raises:
        ValueError: If checkpoint metadata names another model or sampler.
    """

    if config.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format: {config.get('checkpoint_format')}"
        )
    if config.get("sampler_backend") != SA_BACKEND:
        raise ValueError("checkpoint was not trained with the required SA backend")
    keys = {
        "proposal_hidden_dim",
        "candidate_feature_dim",
        "bm_num_visible",
        "bm_num_hidden",
        "contextual_dim",
        "contextual_layers",
        "contextual_heads",
        "contextual_ffn_dim",
        "contextual_dropout",
        "max_sequence_length",
        "sa_num_solutions",
        "sa_seed",
        "energy_num_lowest",
        "sa_kwargs",
    }
    kwargs = {key: config[key] for key in keys}
    return ContextualEnergyModel(**kwargs, sampler=sampler, device=device)
