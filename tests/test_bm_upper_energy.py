"""Full BM energy values, derivatives, fallback and resource regressions."""
import itertools
from decimal import Decimal
import types

import pytest
import torch

from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine


def legacy(model, states):
    return -states @ model.linear_bias - 0.5 * torch.sum(
        states.matmul(model.symmetrized_quadratic_coef()) * states, dim=-1,
    )


def make_model(size, dtype=torch.float64):
    return BoltzmannMachine(
        size, quadratic_coef=torch.randn(size, size, dtype=dtype),
        linear_bias=torch.randn(size, dtype=dtype), device="cpu",
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [7, 17, 27])
@pytest.mark.parametrize("size", [1, 2, 4, 7])
@pytest.mark.parametrize("binary", [False, True])
def test_values_and_parameter_input_gradients(dtype, seed, size, binary):
    torch.manual_seed(seed)
    model = make_model(size, dtype)
    states = (torch.randint(0, 2, (5, size)).to(dtype) if binary
              else torch.randn(5, size, dtype=dtype)).requires_grad_()
    saved = {name: parameter.clone() for name, parameter in model.named_parameters()}
    output, reference = model(states), legacy(model, states)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(output, reference, atol=tolerance, rtol=tolerance)
    parameters = (states, model.quadratic_coef, model.linear_bias)
    actual_grad = torch.autograd.grad(output.sum(), parameters)
    expected_grad = torch.autograd.grad(reference.sum(), parameters)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(actual_grad[1].tril()) == 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, saved[name], rtol=0, atol=0)
        assert parameter.grad is None


@pytest.mark.parametrize("size", [1, 2, 4, 7])
def test_independent_enumerated_pair_energies(size):
    torch.manual_seed(17)
    model = make_model(size)
    for bits in itertools.product((0., 1.), repeat=size):
        state = torch.tensor([bits], dtype=torch.float64)
        expected = -sum(float(model.linear_bias[i]) * bits[i] for i in range(size))
        expected -= sum(float(model.quadratic_coef[i, j]) * bits[i] * bits[j]
                        for i in range(size) for j in range(i + 1, size))
        torch.testing.assert_close(model(state), torch.tensor([expected], dtype=state.dtype))


@pytest.mark.parametrize("layout", ["slice", "transpose", "negative_values", "empty", "one_dimensional"])
def test_shapes_and_layouts(layout):
    model = make_model(4)
    if layout == "slice":
        states = torch.randn(8, 8, dtype=torch.float64)[::2, ::2]
    elif layout == "transpose":
        states = torch.randn(4, 7, dtype=torch.float64).T
    elif layout == "negative_values":
        states = -torch.rand(3, 4, dtype=torch.float64)
    elif layout == "empty":
        states = torch.empty(0, 4, dtype=torch.float64)
    else:
        states = torch.rand(4, dtype=torch.float64)
    torch.testing.assert_close(model(states), legacy(model, states))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduced_parameter_dtype_uses_exact_legacy_path(dtype):
    model = make_model(4, dtype)
    states = torch.rand(3, 4, dtype=dtype)
    torch.testing.assert_close(model(states), legacy(model, states), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_autocast_values_and_gradients_preserve_legacy(dtype):
    model = make_model(4, torch.float32)
    states = torch.rand(3, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=dtype):
        output, reference = model(states), legacy(model, states)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    actual = torch.autograd.grad(output.sum(), (states, *model.parameters()))
    expected = torch.autograd.grad(reference.sum(), (states, *model.parameters()))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("customization", ["subclass", "instance", "class"])
def test_custom_symmetry_still_called(customization, monkeypatch):
    calls = []
    def custom(self):
        calls.append(True)
        return self.quadratic_coef + 2 * torch.eye(self.num_nodes, dtype=self.quadratic_coef.dtype)
    if customization == "subclass":
        class Custom(BoltzmannMachine):
            symmetrized_quadratic_coef = custom
        model = Custom(3, device="cpu").double()
    else:
        model = make_model(3)
        if customization == "instance":
            monkeypatch.setattr(model, "symmetrized_quadratic_coef", types.MethodType(custom, model))
        else:
            monkeypatch.setattr(BoltzmannMachine, "symmetrized_quadratic_coef", custom)
    states = torch.rand(2, 3, dtype=torch.float64)
    actual = model(states)
    assert len(calls) == 1
    torch.testing.assert_close(actual, legacy(model, states), rtol=0, atol=0)


def test_default_float_forward_does_not_build_symmetry(monkeypatch):
    model = make_model(8)
    original = torch.Tensor.__add__
    square_additions = []
    def record(self, other):
        if self.shape == (8, 8) and isinstance(other, torch.Tensor) and other.shape == self.shape:
            square_additions.append(True)
        return original(self, other)
    monkeypatch.setattr(torch.Tensor, "__add__", record)
    output = model(torch.ones(2, 8, dtype=torch.float64))
    assert output.shape == (2,)
    assert not square_additions


def test_gradcheck_and_gradgradcheck():
    model = make_model(3)
    states = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
    quadratic = model.quadratic_coef.detach().clone().requires_grad_()
    bias = model.linear_bias.detach().clone().requires_grad_()
    def energy(x, q, b):
        return torch.func.functional_call(model, {"quadratic_coef": q, "linear_bias": b}, (x,))
    assert torch.autograd.gradcheck(energy, (states, quadratic, bias))
    assert torch.autograd.gradgradcheck(energy, (states, quadratic, bias))


def test_module_hooks_and_objective_gradients():
    model = make_model(3)
    seen = []
    handle = model.register_forward_hook(lambda *args: seen.append(True))
    positive = torch.rand(4, 3, dtype=torch.float64)
    negative = torch.rand(5, 3, dtype=torch.float64)
    objective = model.objective(positive, negative)
    assert len(seen) == 2
    handle.remove()
    expected = legacy(model, positive).mean() - legacy(model, negative).mean()
    torch.testing.assert_close(objective, expected)
    a = torch.autograd.grad(objective, tuple(model.parameters()))
    b = torch.autograd.grad(expected, tuple(model.parameters()))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def test_extreme_single_pair_does_not_double_before_halving():
    model = make_model(2)
    with torch.no_grad():
        model.quadratic_coef.zero_(); model.linear_bias.zero_()
        model.quadratic_coef[0, 1] = 1e308
    state = torch.ones(1, 2, dtype=torch.float64)
    assert torch.isinf(legacy(model, state)).all()
    assert model(state).item() == -float(Decimal('1e308'))


def test_cancellation_against_independent_decimal():
    model = make_model(3)
    with torch.no_grad():
        model.quadratic_coef.zero_(); model.linear_bias.zero_()
        model.quadratic_coef[0, 1] = 2. ** 40
        model.quadratic_coef[0, 2] = -2. ** 40
        model.quadratic_coef[1, 2] = 0.5
    state = torch.ones(1, 3, dtype=torch.float64)
    expected = -(Decimal(2) ** 40 - Decimal(2) ** 40 + Decimal('0.5'))
    assert model(state).item() == float(expected)


def test_meta_shapes_do_not_require_autocast_device_support():
    model = BoltzmannMachine(3, quadratic_coef=torch.empty(3, 3, device="meta"),
                             linear_bias=torch.empty(3, device="meta"), device="meta")
    assert model(torch.empty(2, 3, device="meta")).shape == (2,)


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_gradient_context_remains_disabled(context):
    model = make_model(3)
    with context():
        assert not model(torch.ones(2, 3, dtype=torch.float64)).requires_grad


def test_complex_dtype_preserves_legacy_path():
    model = make_model(3, torch.complex128)
    states = torch.randn(2, 3, dtype=torch.complex128)
    torch.testing.assert_close(model(states), legacy(model, states), rtol=0, atol=0)


def test_existing_gradients_and_training_flags_unchanged():
    model = make_model(3).eval()
    before = []
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
        before.append(parameter.grad.clone())
    result = model(torch.rand(2, 3, dtype=torch.float64))
    assert result.requires_grad
    assert not model.training
    for parameter, saved in zip(model.parameters(), before):
        torch.testing.assert_close(parameter.grad, saved, rtol=0, atol=0)
