"""Default-config contract for the DPLM example dataclasses."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion"))

from dplm.models.backbone import (  # noqa: E402
    DPLMConfig,
    DPLMLoRAConfig,
    DPLMNetConfig,
)


def test_network_defaults_match_the_public_checkpoint():
    """The default net config points at the public ESM2 checkpoint, untrained."""
    net = DPLMNetConfig()
    assert net.arch_type == "esm"
    assert net.name == "esm2_t33_650M_UR50D"
    assert net.dropout == 0.1
    assert net.pretrain is False
    assert net.pretrained_model_name_or_path == ""


def test_lora_defaults_are_disabled_but_preserved():
    """LoRA stays disabled with the legacy defaults kept for old checkpoints."""
    lora = DPLMLoRAConfig()
    assert lora.enable is False
    assert lora.lora_rank == 16
    assert lora.lora_dropout == 0.1
    assert lora.lora_target_module == ""
    assert lora.modules_to_save == ""


def test_wrapper_defaults():
    """The wrapper keeps 500 diffusion steps and disables both extras."""
    config = DPLMConfig()
    assert config.num_diffusion_timesteps == 500
    assert config.gradient_ckpt is False
    assert config.use_coupled_sampling is False
    assert isinstance(config.net, DPLMNetConfig)
    assert isinstance(config.lora, DPLMLoRAConfig)
