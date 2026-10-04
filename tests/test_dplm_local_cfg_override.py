"""Regression test: local DPLM checkpoints must honor ``cfg_override``.

``DPLMBackbone.from_pretrained(..., from_huggingface=False)`` accepted a
``cfg_override`` mapping but constructed the model only from the YAML config,
so overrides such as ``{"gradient_ckpt": True}`` were silently discarded on
the local-checkpoint branch while the Hugging Face branch honored them.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("omegaconf")

import torch  # noqa: E402  pylint: disable=wrong-import-position
from omegaconf import OmegaConf  # noqa: E402  pylint: disable=wrong-import-position

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "example" / "qdiffusion" / "dplm" / "models"


def _install_transformers_stubs(monkeypatch):
    """Provide import-time stand-ins for the heavy transformers dependency."""
    esm_names = (
        "EsmAttention",
        "EsmEncoder",
        "EsmLayer",
        "EsmLMHead",
        "EsmPreTrainedModel",
        "EsmSelfAttention",
    )
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = object
    transformers.AutoModelForMaskedLM = object
    transformers.AutoTokenizer = object
    modeling_outputs = types.ModuleType("transformers.modeling_outputs")
    modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions = object
    esm_modeling = types.ModuleType("transformers.models.esm.modeling_esm")
    for name in esm_names:
        setattr(esm_modeling, name, object)
    models_pkg = types.ModuleType("transformers.models")
    models_pkg.esm = types.ModuleType("transformers.models.esm")
    for module in (
        transformers,
        models_pkg,
        models_pkg.esm,
        esm_modeling,
        modeling_outputs,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    sys.modules["transformers"].modeling_outputs = modeling_outputs
    sys.modules["transformers"].models = models_pkg


class _TinyNet(torch.nn.Module):
    """Minimal net exposing the attribute surface DPLMBackbone expects."""

    def __init__(self):
        super().__init__()
        self.tokenizer = object()
        self.mask_id = 0
        self.pad_id = 0
        self.bos_id = 0
        self.eos_id = 0
        self.x_id = 0
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True

    def forward(self, input_ids):  # pragma: no cover - never executed here
        return {"logits": input_ids}


@pytest.fixture(name="backbone_module")
def load_backbone(monkeypatch):
    """Loads the real backbone module with heavy imports stubbed."""
    _install_transformers_stubs(monkeypatch)
    package = types.ModuleType("dplm_backbone_under_test")
    package.__path__ = [str(MODELS_DIR)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.backbone", MODELS_DIR / "backbone.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_local_checkpoint(monkeypatch, backbone_module, yaml_model_cfg):
    """Points the local branch at an in-memory config and empty checkpoint."""
    yaml_cfg = OmegaConf.create({"model": yaml_model_cfg})

    monkeypatch.setattr(
        backbone_module, "load_yaml_config", lambda path: yaml_cfg, raising=True
    )
    monkeypatch.setattr(
        backbone_module.torch,
        "load",
        lambda path, map_location=None: {"state_dict": {}},
        raising=True,
    )

    def _fake_get_net(cfg):
        return _TinyNet()

    monkeypatch.setattr(backbone_module, "get_net", _fake_get_net, raising=True)


_LOCAL_YAML_MODEL = {
    "_target_": "unit_test.Model",
    "num_diffusion_timesteps": 12,
    "gradient_ckpt": False,
    "net": {
        "arch_type": "esm",
        "name": "unit_test_net",
        "dropout": 0.1,
        "pretrain": True,
        "pretrained_model_name_or_path": "unused",
    },
}


def test_local_checkpoint_applies_cfg_override(monkeypatch, backbone_module):
    _fake_local_checkpoint(monkeypatch, backbone_module, _LOCAL_YAML_MODEL)

    model = backbone_module.DPLMBackbone.from_pretrained(
        "fake/ckpt.pth",
        cfg_override={"gradient_ckpt": True, "num_diffusion_timesteps": 7},
        from_huggingface=False,
    )

    assert model.cfg.gradient_ckpt is True
    assert model.cfg.num_diffusion_timesteps == 7
    # The local branch must keep disabling pretraining on the restored net.
    assert model.cfg.net.pretrain is False


def test_local_checkpoint_without_override_keeps_yaml_values(
    monkeypatch, backbone_module
):
    _fake_local_checkpoint(monkeypatch, backbone_module, _LOCAL_YAML_MODEL)

    model = backbone_module.DPLMBackbone.from_pretrained(
        "fake/ckpt.pth", from_huggingface=False
    )

    assert model.cfg.gradient_ckpt is False
    assert model.cfg.num_diffusion_timesteps == 12


def test_local_checkpoint_override_can_enable_gradient_checkpointing(
    monkeypatch, backbone_module
):
    net = _TinyNet()

    def _fake_get_net(cfg):
        return net

    _fake_local_checkpoint(monkeypatch, backbone_module, _LOCAL_YAML_MODEL)
    monkeypatch.setattr(backbone_module, "get_net", _fake_get_net, raising=True)

    backbone_module.DPLMBackbone.from_pretrained(
        "fake/ckpt.pth", cfg_override={"gradient_ckpt": True}, from_huggingface=False
    )

    assert net.checkpointing_enabled is True
