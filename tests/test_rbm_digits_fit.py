"""Contract test for the ``RBMRunner.fit`` training loop."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
RBM_DIGITS = REPO / "example" / "rbm_digits" / "rbm_digits.py"


def _load_runner():
    # Only orchestration is under test: no real SDK constructor or solver import.
    names = (
        "kaiwu", "kaiwu.torch_plugin", "kaiwu.classical", "kaiwu.cim",
        "kaiwu.torch_plugin._kaiwu_compat",
    )
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in names}
    modules = {name: types.ModuleType(name) for name in names}
    modules["kaiwu"].__path__ = []
    modules["kaiwu.torch_plugin"].__path__ = []
    sampler_calls = []

    def _forbidden_sdk_action(*_args, **_kwargs):
        raise AssertionError("Real SDK models/CIM/reducers are not this fixture")

    def _inert_sampler(**kwargs):
        sampler_calls.append(kwargs)
        return object()

    modules["kaiwu.classical"].SimulatedAnnealingOptimizer = _inert_sampler
    modules["kaiwu.torch_plugin"].RestrictedBoltzmannMachine = _forbidden_sdk_action
    modules["kaiwu.cim"].CIMOptimizer = _forbidden_sdk_action
    modules["kaiwu.cim"].PrecisionReducer = _forbidden_sdk_action
    modules["kaiwu.torch_plugin._kaiwu_compat"].PrecisionReducer = _forbidden_sdk_action
    modules["kaiwu"].classical = modules["kaiwu.classical"]
    modules["kaiwu"].cim = modules["kaiwu.cim"]
    modules["kaiwu"].torch_plugin = modules["kaiwu.torch_plugin"]
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("rbm_digits_fit_uut", RBM_DIGITS)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module, sampler_calls
    finally:
        for name, value in saved.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _FakeRBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.quadratic_coef = nn.Parameter(torch.ones(1))
        self.linear_bias = nn.Parameter(torch.ones(1))
        self.events = []
        self.device = None
        self.trace = []

    def to(self, device):
        """Record the requested device without moving or training a real model."""
        self.device = device
        return self

    def parameters(self):
        """Expose two synthetic scalar parameters to the optimizer spy."""
        return iter([self.quadratic_coef, self.linear_bias])

    def get_hidden(self, inputs):
        """Record each synthetic input batch without applying a real model."""
        self.events.append(("hidden", tuple(inputs.shape)))
        self.trace.append(("hidden", inputs.detach().clone()))
        return inputs

    def sample(self, sampler):
        """Record the opaque sampler without invoking any SDK solver."""
        self.events.append(("sample", sampler))
        self.trace.append(("sample", sampler))
        return torch.zeros(1)

    def objective(self, _hidden, _sampled):
        """Return a scalar autograd fixture and observe backward traversal."""
        self.events.append(("objective",))
        self.trace.append(("objective",))
        loss = self.quadratic_coef.sum() + self.linear_bias.sum()

        def observe_backward(gradient):
            """Record one actual scalar autograd traversal."""
            self.trace.append(("backward", gradient.detach().clone()))

        loss.register_hook(observe_backward)
        return loss


class _FakeOptimizer:
    instances: list["_FakeOptimizer"] = []
    trace = []

    def __init__(self, parameters, lr=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.zero_grad_calls = 0
        self.step_calls = 0
        _FakeOptimizer.instances.append(self)

    def zero_grad(self):
        """Clear synthetic gradients and record the batch reset boundary."""
        self.zero_grad_calls += 1
        self.trace.append(("zero_grad",))
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        """Record gradients without applying any parameter update."""
        self.step_calls += 1
        gradients = [None if p.grad is None else p.grad.detach().clone() for p in self.parameters]
        self.trace.append(("step", gradients))


def test_trains_the_configured_number_of_iterations(monkeypatch):
    """Pin each batch protocol, backward traversal, and loop count."""
    module, sampler_calls = _load_runner()
    created: list[_FakeRBM] = []
    trace = []
    monkeypatch.setattr(_FakeOptimizer, "trace", trace)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    def fake_rbm(num_visible, num_hidden):
        """Construct the synthetic scalar fixture and share its event trace."""
        model = _FakeRBM(num_visible, num_hidden)
        created.append(model)
        model.trace = trace
        return model

    monkeypatch.setattr(module, "RestrictedBoltzmannMachine", fake_rbm)
    monkeypatch.setattr(module, "SGD", _FakeOptimizer)
    _FakeOptimizer.instances.clear()
    runner = module.RBMRunner(
        n_components=3,
        learning_rate=0.05,
        batch_size=2,
        n_iter=2,
        verbose=False,
    )
    data = np.arange(10, dtype=np.float32).reshape(5, 2)

    result = runner.fit(data)

    assert result is runner
    assert len(created) == 1
    model = created[0]
    assert (model.num_visible, model.num_hidden) == (2, 3)
    assert model.device == runner.device
    assert len(model.events) == 18  # six batches x (hidden, sample, objective)
    assert [event[0] for event in model.events[:3]] == ["hidden", "sample", "objective"]
    assert model.events[0][1] == (2, 2)
    optimizer = _FakeOptimizer.instances[0]
    assert optimizer.lr == 0.05
    assert optimizer.zero_grad_calls == 6
    assert optimizer.step_calls == 6
    assert runner.rbm is model


    assert sampler_calls == [{"alpha": 0.999, "size_limit": 100}]
    assert [event[0] for event in trace] == [
        "hidden", "sample", "zero_grad", "objective", "backward", "step"
    ] * 6
    expected_batches = [
        torch.from_numpy(data[start:stop])
        for start, stop in [(0, 2), (2, 4), (4, 5)]
    ] * 2
    actual_batches = [event[1] for event in trace if event[0] == "hidden"]
    assert len(actual_batches) == 6
    for actual, expected in zip(actual_batches, expected_batches):
        torch.testing.assert_close(actual, expected)
    for event in trace:
        if event[0] == "backward":
            torch.testing.assert_close(event[1], torch.tensor(1.0))
        elif event[0] == "step":
            assert len(event[1]) == 2
            assert all(gradient is not None for gradient in event[1])
            assert all(torch.isfinite(gradient).all() for gradient in event[1])
