"""YAML-loading contract for the DPLM example backbone."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from omegaconf import OmegaConf

from models.backbone import load_yaml_config  # noqa: E402


def test_interpolations_are_resolved(tmp_path):
    """Values referencing another key are resolved to the referenced value."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "seed: 7\npaths:\n  output: results\n"
        "resolved_seed: ${seed}\nresolved_new: ${paths.output}/run\n",
        encoding="utf-8",
    )
    cfg = load_yaml_config(str(path))
    assert cfg.seed == 7
    assert cfg.resolved_seed == 7
    assert cfg.resolved_new == "results/run"
    unresolved = OmegaConf.to_container(cfg, resolve=False)
    assert unresolved["resolved_seed"] == 7
    assert unresolved["resolved_new"] == "results/run"


def test_nested_sections_are_loaded_with_their_types(tmp_path):
    """Nested mappings and lists keep their structure."""
    path = tmp_path / "nested.yaml"
    path.write_text(
        "net:\n  name: esm2\n  dropout: 0.25\nnodes: [512, 256]\n",
        encoding="utf-8",
    )
    cfg = load_yaml_config(str(path))
    assert cfg.net.name == "esm2"
    assert cfg.net.dropout == 0.25
    assert list(cfg.nodes) == [512, 256]
