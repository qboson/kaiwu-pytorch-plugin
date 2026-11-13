"""Internal adapter for the frozen text-diffusion proposal checkpoint."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec


def _extract_logits(outputs: Any) -> torch.Tensor:
    """Extract logits from a supported model-output container.

    Args:
        outputs: Tensor, tuple, or Hugging Face model output.

    Returns:
        The model logits tensor.

    Raises:
        TypeError: If ``outputs`` does not expose logits.
    """

    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise TypeError("MDLM output must expose a logits tensor.")


class MDLMBackbone(nn.Module):
    """Wrap an official MDLM checkpoint behind the proposal-model API.

    Args:
        model: Loaded MDLM masked-language model.
        tokenizer: Tokenizer whose vocabulary matches ``model``.
        mask_id: Vocabulary ID reserved for the diffusion mask token.

    Attributes:
        model: Underlying MDLM model.
        tokenizer: Matching tokenizer.
        mask_id: Diffusion mask-token ID.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, mask_id: int) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = int(mask_id)

        if getattr(self.model.config, "time_conditioning", False):
            raise ValueError(
                "The QDiffusion MDLM adapter currently supports checkpoints "
                "with time_conditioning=False only."
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "kuleshov-group/mdlm-owt",
        *,
        tokenizer_name_or_path: str = "gpt2",
        trust_remote_code: bool = True,
        **model_kwargs: Any,
    ) -> MDLMBackbone:
        """Load an MDLM model and reconstruct its GPT-2 tokenizer.

        Args:
            model_name_or_path: Hugging Face model ID or local checkpoint path.
            tokenizer_name_or_path: Hugging Face tokenizer ID or local path.
            trust_remote_code: Whether Transformers may load checkpoint code.
            **model_kwargs: Extra arguments forwarded to
                ``AutoModelForMaskedLM.from_pretrained``.

        Returns:
            A validated MDLM proposal adapter.

        Raises:
            ValueError: If the tokenizer mask ID differs from the checkpoint.
        """

        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

        # Released MDLM checkpoints append the diffusion mask to the base
        # tokenizer vocabulary, so the final vocabulary entry must be <|mask|>.
        mask_id = int(model.config.vocab_size) - 1
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        if tokenizer.mask_token_id != mask_id:
            raise ValueError(
                "MDLM tokenizer mask id does not match the checkpoint vocabulary: "
                f"tokenizer={tokenizer.mask_token_id}, checkpoint={mask_id}."
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer, mask_id=mask_id)

    def _timesteps(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build zero time conditioning expected by the released checkpoint.

        Args:
            input_ids: Token IDs with shape ``(batch, sequence)``.

        Returns:
            A float32 zero tensor with one value per sequence.
        """

        return torch.zeros(
            input_ids.size(0),
            device=input_ids.device,
            dtype=torch.float32,
        )

    @property
    def hidden_size(self) -> int:
        """Return the final token-representation width.

        Returns:
            Hidden width reported by the checkpoint configuration.

        Raises:
            AttributeError: If no recognized hidden-width field exists.
        """

        config = self.model.config
        for name in ("hidden_size", "hidden_dim", "n_embd"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        raise AttributeError("MDLM config does not expose a hidden size.")

    def _raw_forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> Any:
        """Run the checkpoint without applying SUBS parameterization.

        Args:
            input_ids: Token IDs with shape ``(batch, sequence)``.
            output_hidden_states: Whether to return transformer hidden states.

        Returns:
            The native Hugging Face model output.
        """
        return self.model(
            input_ids=input_ids,
            timesteps=self._timesteps(input_ids),
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def _subs_parameterization(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the official MDLM substitution parameterization.

        Args:
            logits: Raw vocabulary logits.
            input_ids: Current noisy token IDs.

        Returns:
            Log probabilities respecting the absorbing-mask transition.
        """

        logits = logits.clone()
        # The reverse process cannot predict another mask as the clean token.
        logits[..., self.mask_id] = -torch.inf
        log_probs = torch.log_softmax(logits, dim=-1)

        # SUBS keeps every already revealed token fixed as a delta
        # distribution, allowing the sampler to update masked positions only.
        unmasked = input_ids.ne(self.mask_id)
        log_probs[unmasked] = -torch.inf
        log_probs[unmasked, input_ids[unmasked]] = 0.0
        return log_probs

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return SUBS-parameterized proposal log probabilities.

        Args:
            input_ids: Current noisy token IDs.
            **kwargs: Reserved for the generic proposal interface.

        Returns:
            Log probabilities with shape ``(batch, sequence, vocabulary)``.

        Raises:
            TypeError: If unsupported proposal arguments are supplied.
        """

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected MDLM proposal arguments: {unexpected}")
        outputs = self._raw_forward(input_ids)
        return self._subs_parameterization(_extract_logits(outputs), input_ids)

    def encode_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return final MDLM hidden states for token sequences.

        Args:
            input_ids: Token IDs with shape ``(batch, sequence)``.
            attention_mask: Accepted for interface compatibility and ignored.

        Returns:
            Final hidden states with shape ``(batch, sequence, hidden)``.

        Raises:
            RuntimeError: If the checkpoint does not return hidden states.
        """

        del attention_mask
        outputs = self._raw_forward(input_ids, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("MDLM checkpoint did not return hidden states.")
        return hidden_states[-1]

    def build_conditioned_output_layer(self) -> nn.Module:
        """Build a hidden-width output layer for conditioned BM features.

        Returns:
            A new output layer matching the proposal backbone architecture.

        Raises:
            AttributeError: If the checkpoint lacks the expected DiT output
                layer.
        """

        backbone = getattr(self.model, "backbone", None)
        output_layer = getattr(backbone, "output_layer", None)
        modulation = getattr(output_layer, "adaLN_modulation", None)
        if output_layer is None or modulation is None:
            raise AttributeError(
                "MDLM backbone does not expose the DiT output-layer interface "
                "required by the conditioned BM encoder."
            )
        return type(output_layer)(
            self.hidden_size,
            self.hidden_size,
            int(modulation.in_features),
        )

    def encode_conditioned_tokens(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        *,
        input_projection: nn.Module,
        output_layer: nn.Module,
    ) -> torch.Tensor:
        """Encode a noisy sequence jointly with a clean-token candidate.

        Args:
            noisy_tokens: Diffusion state ``x_t``.
            candidate_tokens: Candidate clean sequence ``x_0``.
            input_projection: Trainable projection from concatenated proposal
                embeddings to the MDLM hidden width.
            output_layer: Trainable hidden-width output layer.

        Returns:
            Conditioned token features with shape
            ``(batch, sequence, hidden)``.

        Raises:
            AttributeError: If the proposal lacks conditioned-encoder
                components.
        """

        backbone = getattr(self.model, "backbone", None)
        required = ("vocab_embed", "sigma_map", "rotary_emb", "blocks")
        missing = [
            name
            for name in required
            if not hasattr(backbone, name)
        ]
        if missing:
            raise AttributeError(
                "Proposal backbone is missing conditioned-encoder components: "
                + ", ".join(missing)
            )
        noisy_embeddings = backbone.vocab_embed(noisy_tokens)
        candidate_embeddings = backbone.vocab_embed(candidate_tokens)
        # Concatenating x_t and x_0 preserves candidate-specific information
        # before the shared proposal blocks contextualize the sequence.
        hidden_states = input_projection(
            torch.cat([noisy_embeddings, candidate_embeddings], dim=-1)
        )
        conditioning = F.silu(
            backbone.sigma_map(self._timesteps(noisy_tokens))
        )
        rotary_cos_sin = backbone.rotary_emb(hidden_states)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            for block in backbone.blocks:
                hidden_states = block(
                    hidden_states,
                    rotary_cos_sin,
                    conditioning,
                    seqlens=None,
                )
            hidden_states = output_layer(hidden_states, conditioning)
        return hidden_states

    def train_last_blocks(self, num_blocks: int) -> list[str]:
        """Freeze the proposal except for its final transformer blocks.

        Args:
            num_blocks: Number of trailing blocks to make trainable.

        Returns:
            Fully qualified names of trainable proposal parameters.

        Raises:
            AttributeError: If the model does not expose transformer blocks.
            ValueError: If ``num_blocks`` is outside the available range.
        """

        blocks = getattr(
            getattr(self.model, "backbone", None),
            "blocks",
            None,
        )
        if blocks is None:
            raise AttributeError(
                "MDLM model does not expose model.backbone.blocks."
            )
        if num_blocks < 0 or num_blocks > len(blocks):
            raise ValueError(
                f"num_blocks must be between 0 and {len(blocks)}."
            )
        # Start from a fully frozen proposal so the requested trainable suffix
        # is explicit and checkpoint metadata remains auditable.
        for parameter in self.parameters():
            parameter.requires_grad = False
        if num_blocks:
            for block in blocks[-num_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]


def build_mdlm_token_spec(backbone: MDLMBackbone) -> SequenceTokenSpec:
    """Build generic QDiffusion token metadata for an MDLM backbone.

    Args:
        backbone: Validated MDLM proposal adapter.

    Returns:
        Token IDs and tokenizer required by sequence QDiffusion.

    Raises:
        ValueError: If the tokenizer lacks a required special-token ID.
    """

    tokenizer = backbone.tokenizer
    missing = [
        name
        for name, value in (
            ("pad_token_id", tokenizer.pad_token_id),
            ("bos_token_id", tokenizer.bos_token_id),
            ("eos_token_id", tokenizer.eos_token_id),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"MDLM tokenizer is missing required ids: {', '.join(missing)}"
        )

    return SequenceTokenSpec(
        mask_id=backbone.mask_id,
        pad_id=int(tokenizer.pad_token_id),
        bos_id=int(tokenizer.bos_token_id),
        eos_id=int(tokenizer.eos_token_id),
        x_id=None,
        tokenizer=tokenizer,
    )
