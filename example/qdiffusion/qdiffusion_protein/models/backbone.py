"""Backbone and token-spec helpers for the DPLM example stack."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForMaskedLM

from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec

from .esm_patch import _EsmForDPLM


@dataclass
class DPLMNetConfig:
    """Network-level DPLM runtime configuration."""

    arch_type: str = "esm"
    name: str = "esm2_t33_650M_UR50D"
    dropout: float = 0.1
    pretrain: bool = False
    pretrained_model_name_or_path: str = ""


@dataclass
class DPLMLoRAConfig:
    """LoRA configuration preserved for compatibility with old checkpoints."""

    enable: bool = field(default=False)
    lora_rank: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    lora_target_module: str = field(default="")
    modules_to_save: str = field(default="")


@dataclass
class DPLMConfig:
    """Wrapper configuration for the example DPLM backbone."""

    num_diffusion_timesteps: int = field(default=500)
    lora: DPLMLoRAConfig = field(default_factory=DPLMLoRAConfig)
    net: DPLMNetConfig = field(default_factory=DPLMNetConfig)
    gradient_ckpt: bool = field(default=False)
    use_coupled_sampling: bool = field(default=False)


def load_yaml_config(fpath: str):
    """Loads one OmegaConf YAML file and resolves interpolations."""
    cfg = OmegaConf.load(fpath)
    OmegaConf.resolve(cfg)
    return cfg


def get_net_class(dplm_type: str):
    """Returns the supported masked-LM class for one DPLM type."""
    if dplm_type != "dplm_esm":
        raise ValueError(f"Unsupported dplm_type for DPLM examples: {dplm_type}")
    return _EsmForDPLM


def get_net(cfg):
    """Builds one underlying masked-LM network from the example config."""
    if cfg.net.arch_type != "esm":
        raise NotImplementedError(
            f"Unsupported arch_type for DPLM examples: {cfg.net.arch_type}"
        )

    # The example keeps one patched ESM implementation as its single DPLM
    # backbone family; higher-level code should not depend on patch details.
    config = AutoConfig.from_pretrained(cfg.net.name)
    net = _EsmForDPLM(config, dropout=cfg.net.dropout)

    if cfg.net.pretrain:
        pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        is_local = os.path.exists(pretrained_model_name_or_path)
        if is_local:
            # Local training artifacts come from the original DPLM codebase, so
            # we strip their checkpoint prefixes before loading into the example
            # wrapper.
            pretrained_state_dict = torch.load(
                pretrained_model_name_or_path, map_location="cpu"
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()
            for key, value in pretrained_state_dict.items():
                new_pretrained_state_dict[key[10:]] = value
            net.load_state_dict(new_pretrained_state_dict, strict=True)
        else:
            pretrained_net = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path
            )
            net.load_state_dict(pretrained_net.state_dict(), strict=True)
            del pretrained_net

    return net


class DPLMBackbone(nn.Module):
    """Minimal DPLM backbone wrapper used by the examples."""

    _default_cfg = DPLMConfig()

    def __init__(self, cfg: Any | None = None, net: nn.Module | None = None):
        super().__init__()
        self._update_cfg(cfg or {})

        # ``self.net`` is the only heavy model object here; the wrapper mainly
        # exposes token ids and a stable forward interface to the example code.
        self.net = get_net(self.cfg) if net is None else net
        self.tokenizer = self.net.tokenizer

        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.x_id = self.net.x_id

        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(
        cls,
        net_name,
        cfg_override=None,
        net_override=None,
        from_huggingface=True,
    ):
        """Loads one DPLM backbone wrapper from a checkpoint or model id."""
        cfg_override = cfg_override or {}
        net_override = net_override or {}

        if not from_huggingface:
            # This branch restores one local DPLM training artifact rather than
            # a Hub model id.
            cfg_path = Path(net_name).parents[1]
            cfg_path = Path(cfg_path, ".hydra", "config.yaml")
            cfg = load_yaml_config(str(cfg_path)).model
            cfg.net.pretrain = False
            cfg.pop("_target_")
            model = cls(cfg)

            pretrained_state_dict = torch.load(
                net_name, map_location=torch.device("cpu")
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()
            for key, value in pretrained_state_dict.items():
                new_pretrained_state_dict[key[6:]] = value

            model.load_state_dict(new_pretrained_state_dict, strict=False)
            return model

        dplm_type = AutoConfig.from_pretrained(net_name).dplm_type
        net_class = get_net_class(dplm_type)
        net = net_class.from_pretrained(net_name, **net_override)
        return cls(cfg=cfg_override, net=net)

    def _update_cfg(self, cfg):
        """Merges runtime config overrides onto the default DPLM config."""
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)

    def forward(self, input_ids, return_last_hidden_state=False):
        """Runs the wrapped DPLM network."""
        outputs = self.net(input_ids=input_ids)
        logits = outputs["logits"]
        if return_last_hidden_state:
            return logits, outputs["last_hidden_state"]
        return logits


def build_dplm_token_spec(backbone: DPLMBackbone) -> SequenceTokenSpec:
    """Builds one generic token spec from a DPLM backbone wrapper."""
    return SequenceTokenSpec(
        mask_id=backbone.mask_id,
        pad_id=backbone.pad_id,
        bos_id=backbone.bos_id,
        eos_id=backbone.eos_id,
        x_id=backbone.x_id,
        tokenizer=backbone.tokenizer,
    )
