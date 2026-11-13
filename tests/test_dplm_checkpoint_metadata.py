"""Regression test for safe loading of DPLM energy checkpoints."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

_RUNTIME_PATH = (
    Path(__file__).resolve().parents[1]
    / "example/qdiffusion/dplm/utils/runtime.py"
)
_SPEC = importlib.util.spec_from_file_location("_dplm_runtime_metadata", _RUNTIME_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load the DPLM runtime utility")
runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runtime)


class _EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(2, 2)
        self.feature_projector = nn.Linear(2, 1)
        self.energy_bm = nn.Linear(1, 1)
        self.sampler_type = "cim"
        self.sampler_kwargs = {
            "tmp_dir": Path("/tmp/cim-cache"),
            "nested": {"checkpoint": Path("/tmp/cim.lic")},
            "paths": [Path("/tmp/cim-a"), {"nested": Path("/tmp/cim-b")}],
            "reads": 10,
        }


class _Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.energy_model = _EnergyModel()


class TestCheckpointMetadata(unittest.TestCase):
    def test_path_metadata_survives_safe_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = _Generator()
            checkpoint_path = runtime.save_checkpoint(
                root / "checkpoints",
                "best.pt",
                generator=generator,
                epoch=3,
                metric=0.5,
            )

            loaded = torch.load(checkpoint_path, map_location="cpu")
            sampler_kwargs = loaded["metadata"]["bm_sampler_kwargs"]
            self.assertEqual(sampler_kwargs["tmp_dir"], "/tmp/cim-cache")
            self.assertEqual(
                sampler_kwargs["nested"]["checkpoint"], "/tmp/cim.lic"
            )
            self.assertEqual(
                sampler_kwargs["paths"],
                ["/tmp/cim-a", {"nested": "/tmp/cim-b"}],
            )
            self.assertEqual(sampler_kwargs["reads"], 10)

            restored = _Generator()
            metadata = runtime.load_trained_energy_weights(
                restored, str(checkpoint_path), "cpu"
            )
            self.assertEqual(metadata["bm_sampler_type"], "cim")
            for original, copied in zip(
                generator.energy_model.parameters(),
                restored.energy_model.parameters(),
            ):
                self.assertTrue(torch.equal(original, copied))


if __name__ == "__main__":
    unittest.main()
