"""Real-file CPU restoration and a requested-CUDA staging-policy probe."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn


_RUNTIME_PATH = Path(__file__).resolve().parents[1] / "example/qdiffusion/dplm/utils/runtime.py"
_SPEC = importlib.util.spec_from_file_location("_checkpoint_runtime_tests", _RUNTIME_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load the complete DPLM runtime utility")
runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runtime)

_DTYPE = torch.float64
_EXPECTED = {
    "energy_encoder": {
        "0.weight": [[1.0, 0.5, -0.25], [-0.5, 1.0, 0.75], [0.25, -0.5, 1.5]],
        "0.bias": [0.1, -0.2, 0.3],
        "1.weight": [1.2, 0.8, 1.1],
        "1.bias": [0.05, -0.1, 0.2],
        "1.running_mean": [0.2, -0.3, 0.4],
        "1.running_var": [1.3, 2.3, 0.7],
        "1.num_batches_tracked": 23,
    },
    "feature_projector": {"weight": [[0.3, -0.7, 1.2], [-0.4, 0.8, 0.5]], "bias": [0.2, -0.1]},
    "energy_bm": {"weight": [[0.6, -1.1]], "bias": [0.25]},
}
_INPUT = [[0.2, -0.5, 1.0], [-0.7, 0.3, 0.8]]


def energy_modules(generator):
    energy = generator.energy_model
    return {"energy_encoder": energy.encoder.backbone,
            "feature_projector": energy.feature_projector, "energy_bm": energy.energy_bm}


def state_tensors(values):
    return {name: torch.tensor(value, dtype=torch.int64 if name.endswith("num_batches_tracked") else _DTYPE,
                               device="cpu") for name, value in values.items()}


def make_components(value):
    """Construct independent genuine Torch components at the checkpoint interfaces."""
    with torch.random.fork_rng(devices=[]):
        generator = nn.Module()
        generator.energy_model = nn.Module()
        generator.energy_model.encoder = nn.Module()
        generator.energy_model.encoder.backbone = nn.Sequential(
            nn.Linear(3, 3, dtype=_DTYPE, device="cpu"),
            nn.BatchNorm1d(3, dtype=_DTYPE, device="cpu"),
        )
        generator.energy_model.feature_projector = nn.Linear(3, 2, dtype=_DTYPE, device="cpu")
        generator.energy_model.energy_bm = nn.Linear(2, 1, dtype=_DTYPE, device="cpu")
        generator.proposal = nn.Linear(3, 3, dtype=_DTYPE, device="cpu")
    with torch.no_grad():
        for parameter in generator.parameters():
            parameter.fill_(value)
        bn = generator.energy_model.encoder.backbone[1]
        bn.running_mean.fill_(-1)
        bn.running_var.fill_(2)
        bn.num_batches_tracked.fill_(9)
    generator.eval()
    generator.proposal.train()
    generator.energy_model.sampler_type = "fixture"
    generator.energy_model.sampler_kwargs = {
        "steps": 7, "temperature": 0.75,
        "calibration": torch.tensor([2.0, 3.0], dtype=_DTYPE, device="cpu"),
    }
    return generator


def score(generator):
    with torch.no_grad():
        energy = generator.energy_model
        features = energy.encoder.backbone(torch.tensor(_INPUT, dtype=_DTYPE, device="cpu"))
        return energy.energy_bm(energy.feature_projector(features)).flatten()


def expected_scores():
    """Compute the fixed scorer independently with scalar arithmetic."""
    encoder = _EXPECTED["energy_encoder"]
    projector = _EXPECTED["feature_projector"]
    bm = _EXPECTED["energy_bm"]
    result = []
    for row in _INPUT:
        linear = [sum(w * x for w, x in zip(weights, row)) + bias
                  for weights, bias in zip(encoder["0.weight"], encoder["0.bias"])]
        normalized = [(x - mean) / math.sqrt(variance + 1e-5) * weight + bias
                      for x, mean, variance, weight, bias in zip(linear, encoder["1.running_mean"],
                      encoder["1.running_var"], encoder["1.weight"], encoder["1.bias"])]
        projected = [sum(w * x for w, x in zip(weights, normalized)) + bias
                     for weights, bias in zip(projector["weight"], projector["bias"])]
        result.append(sum(w * x for w, x in zip(bm["weight"][0], projected)) + bm["bias"][0])
    return torch.tensor(result, dtype=_DTYPE, device="cpu")


class CheckpointCPUStagingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = make_components(0.9)
        for name, module in energy_modules(self.source).items():
            module.load_state_dict(state_tensors(_EXPECTED[name]))
        self.target = make_components(-3.0)
        self.modules = dict(self.target.named_modules())
        self.parameters = dict(self.target.named_parameters())
        self.buffers = dict(self.target.named_buffers())
        self.devices = {name: tensor.device for name, tensor in list(self.parameters.items()) + list(self.buffers.items())}
        self.modes = {name: module.training for name, module in self.modules.items()}
        self.proposal_state = {name: tensor.clone() for name, tensor in self.target.proposal.state_dict().items()}
        self.assertFalse(torch.allclose(score(self.target), expected_scores()))
        self.path = runtime.save_checkpoint(Path(self.directory.name), "energy.pt",
                                            generator=self.source, epoch=4, metric=0.125)
        self.assertTrue(self.path.is_file())

    def assert_restored(self, metadata):
        for name, module in energy_modules(self.target).items():
            actual = module.state_dict()
            expected = state_tensors(_EXPECTED[name])
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(set(dict(self.target.named_modules())), set(self.modules))
        self.assertEqual(set(dict(self.target.named_parameters())), set(self.parameters))
        self.assertEqual(set(dict(self.target.named_buffers())), set(self.buffers))
        for name, module in self.target.named_modules():
            self.assertIs(module, self.modules[name])
            self.assertEqual(module.training, self.modes[name])
        for name, tensor in self.target.named_parameters():
            self.assertIs(tensor, self.parameters[name])
            self.assertEqual(tensor.device, self.devices[name])
        for name, tensor in self.target.named_buffers():
            self.assertIs(tensor, self.buffers[name])
            self.assertEqual(tensor.device, self.devices[name])
        proposal_state = self.target.proposal.state_dict()
        self.assertEqual(set(proposal_state), set(self.proposal_state))
        for name, tensor in proposal_state.items():
            torch.testing.assert_close(tensor, self.proposal_state[name], rtol=0, atol=0)
        torch.testing.assert_close(score(self.target), expected_scores(), rtol=1e-12, atol=1e-12)
        self.assertEqual(metadata["bm_sampler_type"], "fixture")
        self.assertEqual(metadata["bm_sampler_kwargs"]["steps"], 7)
        self.assertEqual(metadata["bm_sampler_kwargs"]["temperature"], 0.75)
        torch.testing.assert_close(metadata["bm_sampler_kwargs"]["calibration"],
                                   torch.tensor([2.0, 3.0], dtype=_DTYPE, device="cpu"), rtol=0, atol=0)

    def test_unwrapped_cpu_file_roundtrip(self):
        payload = torch.load(self.path, map_location="cpu")
        self.assertEqual(payload["epoch"], 4)
        self.assertEqual(payload["metric"], 0.125)
        self.assertNotIn("proposal", payload["state_dict"])
        metadata = runtime.load_trained_energy_weights(self.target, str(self.path), "cpu")
        self.assert_restored(metadata)

    def test_requested_cuda_stages_on_cpu_policy(self):
        """Policy-only probe delegating actual file deserialization to CPU."""
        actual_load = torch.load
        observed_locations = []

        def load_on_cpu(*args, **kwargs):
            observed_locations.append(str(kwargs.get("map_location")))
            kwargs["map_location"] = "cpu"
            return actual_load(*args, **kwargs)

        with mock.patch.object(runtime.torch, "load", side_effect=load_on_cpu):
            metadata = runtime.load_trained_energy_weights(self.target, str(self.path), "cuda:0")
        self.assert_restored(metadata)
        self.assertEqual(observed_locations, ["cpu"])


if __name__ == "__main__":
    unittest.main()
