"""Regressions for preserving explicitly supplied GBRBM parameters."""

from pathlib import Path
import sys

import pytest
import torch

src_root = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_root))

import kaiwu

kaiwu.__path__ = [str(src_root / "kaiwu")] + list(kaiwu.__path__)
for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith("kaiwu.torch_plugin."):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def supplied_parameters(visible_gaussian, dtype=torch.float32):
    """Return recognizable weights outside the default initialization range."""
    num_gaussian, num_bernoulli = (3, 2) if visible_gaussian else (2, 3)
    weights = torch.arange(1, 7, dtype=dtype).reshape(num_gaussian, num_bernoulli)
    bias = torch.arange(7, 7 + num_bernoulli, dtype=dtype)
    return weights, bias


@pytest.mark.parametrize("visible_gaussian", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constructor_preserves_supplied_values_and_caller_tensors(visible_gaussian, dtype):
    """Construction must preserve pretrained values and their input tensors."""
    weights, bias = supplied_parameters(visible_gaussian, dtype)
    expected_weights, expected_bias = weights.clone(), bias.clone()
    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        3, 2, is_visible_gaussian=visible_gaussian, dtype=dtype,
        quadratic_coef=weights, linear_bias=bias, device=torch.device("cpu"),
    )

    torch.testing.assert_close(bm.quadratic_coef, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(bm.linear_bias, expected_bias, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(bias, expected_bias, rtol=0, atol=0)
    assert bm.quadratic_coef.requires_grad
    assert bm.linear_bias.requires_grad
    observed = torch.ones(2, bm.num_bernoulli, dtype=dtype)
    inferred = bm.infer_from_bernoulli(observed, no_random=True)
    expected_gaussian = observed @ expected_weights.t() + bm.mu
    torch.testing.assert_close(inferred[:, :bm.num_gaussian], expected_gaussian)


@pytest.mark.parametrize("visible_gaussian", [True, False])
@pytest.mark.parametrize("supplied_name", ["quadratic_coef", "linear_bias"])
def test_partial_inputs_preserve_values_and_initialize_missing_parameters(
    visible_gaussian, supplied_name
):
    """Only absent parameters should receive the default random initialization."""
    weights, bias = supplied_parameters(visible_gaussian)
    supplied = weights if supplied_name == "quadratic_coef" else bias
    expected = supplied.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        bm = GaussianBernoulliRestrictedBoltzmannMachine(
            3, 2, is_visible_gaussian=visible_gaussian,
            device=torch.device("cpu"), **{supplied_name: supplied},
        )

    torch.testing.assert_close(getattr(bm, supplied_name), expected, rtol=0, atol=0)
    torch.testing.assert_close(supplied, expected, rtol=0, atol=0)
    missing = bm.linear_bias if supplied_name == "quadratic_coef" else bm.quadratic_coef
    assert torch.isfinite(missing).all()
    assert (missing.abs() <= 0.16).all()
    assert torch.count_nonzero(missing) > 0
    assert (bm.mu.abs() <= 0.16).all()
    torch.testing.assert_close(bm.var, torch.ones(bm.num_gaussian))


@pytest.mark.parametrize("visible_gaussian", [True, False])
def test_default_parameters_remain_randomly_initialized(visible_gaussian):
    """Default means, weights and biases keep the bounded normal initialization."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        bm = GaussianBernoulliRestrictedBoltzmannMachine(
            3, 2, is_visible_gaussian=visible_gaussian, device=torch.device("cpu"),
        )

    for parameter in (bm.mu, bm.quadratic_coef, bm.linear_bias):
        assert torch.isfinite(parameter).all()
        assert (parameter.abs() <= 0.16).all()
        assert torch.count_nonzero(parameter) > 0
    torch.testing.assert_close(bm.log_var, torch.zeros(bm.num_gaussian))


@pytest.mark.parametrize("visible_gaussian", [True, False])
def test_explicit_init_parameter_still_resets_supplied_parameters(visible_gaussian):
    """The public reset method should continue reinitializing every parameter."""
    weights, bias = supplied_parameters(visible_gaussian)
    bm = GaussianBernoulliRestrictedBoltzmannMachine(
        3, 2, is_visible_gaussian=visible_gaussian,
        quadratic_coef=weights, linear_bias=bias, device=torch.device("cpu"),
    )
    bm.init_parameter(init_var=4, std=0)

    for parameter in (bm.mu, bm.quadratic_coef, bm.linear_bias):
        torch.testing.assert_close(parameter, torch.zeros_like(parameter))
    torch.testing.assert_close(bm.var, torch.full((bm.num_gaussian,), 4.0))
