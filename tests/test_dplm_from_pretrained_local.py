"""Local-checkpoint contract for the DPLM backbone wrapper."""

import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

import models.backbone as backbone  # noqa: E402


class FakeNet:
    """Stand-in network carrying the attributes the wrapper copies."""

    def __init__(self):
        self.config = SimpleNamespace(hidden_size=8)
        self.tokenizer = "TOKENIZER"
        self.mask_id = 1
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
        self.x_id = 4
        self.loaded = []

    def load_state_dict(self, state, strict=True):
        self.loaded.append((dict(state), strict))


def build_run_dir(tmp_path):
    """Creates the run/checkpoints/.hydra layout the local branch expects."""
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text(
        "model:\n"
        "  _target_: example.Model\n"
        "  num_diffusion_timesteps: 12\n"
        "  gradient_ckpt: false\n"
        "  net:\n"
        "    arch_type: esm\n"
        "    name: from-yaml\n"
        "    dropout: 0.3\n"
        "    pretrain: true\n"
        "    pretrained_model_name_or_path: original-ckpt\n",
        encoding="utf-8",
    )
    checkpoint = checkpoint_dir / "artifact.pt"
    checkpoint.write_bytes(b"stub")
    return checkpoint


def test_local_branch_reads_the_hydra_config_and_strips_the_prefix(tmp_path, monkeypatch):
    """The local branch reads run/.hydra/config.yaml, disables pretrain and strips keys."""
    checkpoint = build_run_dir(tmp_path)
    net = FakeNet()
    seen_cfg = []

    def fake_get_net(cfg):
        seen_cfg.append(cfg)
        return net

    def fake_load(path, map_location="cpu"):
        assert str(path) == str(checkpoint)
        assert str(map_location) == "cpu"
        return {"state_dict": {"PREFIXabc": 5, "PREFIXxyz": 6}}

    loaded = []

    def fake_load_state_dict(self, state, strict=True):
        loaded.append((dict(state), strict))

    monkeypatch.setattr(backbone, "get_net", fake_get_net)
    monkeypatch.setattr(backbone.torch, "load", fake_load)
    monkeypatch.setattr(backbone.DPLMBackbone, "load_state_dict", fake_load_state_dict)
    wrapper = backbone.DPLMBackbone.from_pretrained(
        str(checkpoint), from_huggingface=False
    )
    assert isinstance(wrapper.net, FakeNet)
    assert len(seen_cfg) == 1
    assert seen_cfg[0].num_diffusion_timesteps == 12
    assert seen_cfg[0].net.name == "from-yaml"
    assert seen_cfg[0].net.dropout == 0.3
    assert seen_cfg[0].net.pretrain is False
    assert not hasattr(seen_cfg[0], "_target_")
    assert loaded == [({"abc": 5, "xyz": 6}, False)]
