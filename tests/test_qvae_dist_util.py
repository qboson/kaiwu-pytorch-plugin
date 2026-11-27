"""Unit tests for the QVAE distribution utility classes and functions."""

import math
import warnings

import pytest
import torch
from torch import distributions
from torch.nn import functional

from kaiwu.torch_plugin.qvae_dist_util import (
    DistUtil,
    Exponential,
    FactorialBernoulliUtil,
    MixtureGeneric,
    SmoothingDist,
    sigmoid_cross_entropy_with_logits,
)


def _truncated_exp_mean(beta: float) -> float:
    """Return the exact mean of the truncated exponential law on the unit interval."""
    return 1.0 / beta - 1.0 / math.expm1(beta)


# Group A: smoothing distribution interface contract.


@pytest.mark.parametrize(
    "method_name, argument",
    [
        ("pdf", torch.tensor(0.5)),
        ("cdf", torch.tensor(0.5)),
        ("sample", (2, 3)),
        ("log_pdf", torch.tensor(0.5)),
    ],
)
def test_smoothing_dist_base_methods_raise_not_implemented(method_name, argument):
    """Every interface method of the smoothing base class must raise."""
    base_dist = SmoothingDist()
    with pytest.raises(NotImplementedError):
        getattr(base_dist, method_name)(argument)


# Group B: truncated exponential smoothing distribution.


def test_exponential_init_stores_float32_cpu_beta():
    """Beta is stored as a CPU float32 tensor whatever the input type is."""
    numeric_cases = ((2.5, 2.5), (3, 3.0))
    for beta_value, expected in numeric_cases:
        dist = Exponential(beta_value)
        assert isinstance(dist.beta, torch.Tensor)
        assert dist.beta.dtype == torch.float32
        assert dist.beta.device.type == "cpu"
        torch.testing.assert_close(dist.beta, torch.tensor(expected, dtype=torch.float32))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensor_dist = Exponential(torch.tensor(1.5, dtype=torch.float64))
    assert tensor_dist.beta.dtype == torch.float32
    torch.testing.assert_close(tensor_dist.beta, torch.tensor(1.5, dtype=torch.float32))


@pytest.mark.parametrize("beta", [0.1, 1.0, 5.0])
def test_exponential_pdf_closed_form_and_properties(beta):
    """The pdf matches its closed form, stays positive, and decreases strictly."""
    dist = Exponential(beta)
    zeta = torch.linspace(0.0, 1.0, 11)
    denominator = 1.0 - math.exp(-beta)
    expected = torch.tensor(
        [beta * math.exp(-beta * value) / denominator for value in zeta.tolist()],
        dtype=torch.float32,
    )
    pdf_values = dist.pdf(zeta)
    torch.testing.assert_close(pdf_values, expected, rtol=1e-5, atol=1e-6)
    assert torch.all(pdf_values > 0.0)
    assert torch.all(torch.diff(pdf_values) < 0.0)
    ratio = pdf_values[0] / pdf_values[-1]
    torch.testing.assert_close(ratio, torch.tensor(math.exp(beta)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("beta", [0.2, 1.0, 3.0, 10.0])
def test_exponential_pdf_integrates_to_one(beta):
    """The pdf integrates to one over the unit interval."""
    dist = Exponential(beta)
    grid = torch.linspace(0.0, 1.0, 2001)
    area = torch.trapezoid(dist.pdf(grid), grid)
    torch.testing.assert_close(area, torch.tensor(1.0), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("beta", [0.2, 1.0, 4.0])
def test_exponential_cdf_boundaries_monotonicity_and_derivative(beta):
    """The cdf hits both boundaries, increases strictly, and differentiates to the pdf."""
    dist = Exponential(beta)
    grid = torch.linspace(0.0, 1.0, 21, requires_grad=True)
    cdf_values = dist.cdf(grid)
    torch.testing.assert_close(cdf_values[0], torch.tensor(0.0), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(cdf_values[-1], torch.tensor(1.0), rtol=0.0, atol=1e-6)
    assert torch.all(torch.diff(cdf_values) > 0.0)
    assert torch.all(cdf_values >= 0.0)
    assert torch.all(cdf_values <= 1.0)
    (grad_zeta,) = torch.autograd.grad(cdf_values.sum(), grid)
    torch.testing.assert_close(grad_zeta, dist.pdf(grid.detach()), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("beta", [0.1, 1.0, 8.0])
def test_exponential_log_pdf_matches_log_of_pdf_and_grad(beta):
    """The log pdf equals the log of the pdf and its slope equals minus beta."""
    dist = Exponential(beta)
    zeta = torch.linspace(0.0, 1.0, 9, requires_grad=True)
    log_values = dist.log_pdf(zeta)
    torch.testing.assert_close(log_values, torch.log(dist.pdf(zeta)), rtol=1e-5, atol=1e-6)
    (grad_zeta,) = torch.autograd.grad(log_values.sum(), zeta)
    expected_grad = torch.full_like(grad_zeta, -beta)
    torch.testing.assert_close(grad_zeta, expected_grad, rtol=1e-6, atol=1e-6)


def test_exponential_log_pdf_stable_when_pdf_underflows():
    """The log pdf stays finite where the plain pdf underflows to zero."""
    dist = Exponential(120.0)
    zeta = torch.tensor([0.0, 0.5, 1.0])
    pdf_values = dist.pdf(zeta)
    assert pdf_values[-1].item() == 0.0
    assert torch.isneginf(torch.log(pdf_values[-1]))
    log_values = dist.log_pdf(zeta)
    assert torch.isfinite(log_values).all()
    expected = torch.tensor(math.log(120.0) - 120.0)
    torch.testing.assert_close(log_values[-1], expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("shape", [(), (5,), (3, 4), (2, 3, 4), (0, 4)])
def test_exponential_sample_shapes_bounds_and_determinism(shape):
    """Sampling keeps the requested shape, stays in the unit interval, and repeats."""
    dist = Exponential(2.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = dist.sample(shape)
        torch.manual_seed(123)
        second = dist.sample(shape)
    assert first.shape == torch.Size(shape)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.all(first >= 0.0)
    assert torch.all(first <= 1.0)


@pytest.mark.parametrize("beta", [0.5, 2.0, 8.0])
def test_exponential_sample_matches_theoretical_mean_and_cdf(beta):
    """Samples reproduce the analytic mean and the analytic cdf."""
    dist = Exponential(beta)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        samples = dist.sample((20_000,))
    expected_mean = torch.tensor(_truncated_exp_mean(beta))
    torch.testing.assert_close(samples.mean(), expected_mean, rtol=0.0, atol=0.015)
    points = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    empirical_cdf = (samples.unsqueeze(1) <= points).float().mean(dim=0)
    deviation = torch.max(torch.abs(empirical_cdf - dist.cdf(points))).item()
    assert deviation < 0.02
    assert _truncated_exp_mean(8.0) < _truncated_exp_mean(0.5)


def test_exponential_preserves_input_shapes_and_float64_promotion():
    """Element-wise methods keep the input shape and promote float64 inputs."""
    dist = Exponential(1.5)
    shape_cases = (
        ("scalar", torch.tensor(0.4)),
        ("matrix", torch.zeros(2, 3)),
        ("cube", torch.zeros(2, 3, 4)),
        ("empty", torch.empty(0, 3)),
    )
    for case_name, inputs in shape_cases:
        for method_name in ("pdf", "cdf", "log_pdf"):
            outputs = getattr(dist, method_name)(inputs)
            assert outputs.shape == inputs.shape, case_name
            assert outputs.dtype == torch.float32, case_name
            assert torch.isfinite(outputs).all(), case_name

    double_inputs = torch.tensor([0.2, 0.8], dtype=torch.float64)
    for method_name in ("pdf", "cdf", "log_pdf"):
        outputs = getattr(dist, method_name)(double_inputs)
        assert outputs.dtype == torch.float64
        assert torch.isfinite(outputs).all()


# Group C: distribution utility interface contract.


def test_dist_util_base_methods_raise_not_implemented():
    """Every interface method of the utility base class must raise."""
    base_util = DistUtil()
    with pytest.raises(NotImplementedError):
        base_util.reparameterize(True)
    with pytest.raises(NotImplementedError):
        base_util.reparameterize(False)
    with pytest.raises(NotImplementedError):
        base_util.entropy()


# Group D: sigmoid cross entropy with logits.


def test_sigmoid_cross_entropy_matches_pytorch_bce():
    """The custom loss matches the numerically stable PyTorch implementation."""
    logits = torch.tensor([[-2.0, -0.5, 0.0, 1.5], [3.0, 0.0, -1.0, 0.25]])
    hard_labels = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    soft_labels = torch.tensor([[0.0, 0.25, 0.5, 0.75], [1.0, 0.6, 0.4, 0.1]])
    for labels in (hard_labels, soft_labels):
        actual = sigmoid_cross_entropy_with_logits(logits, labels)
        expected = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    column_logits = torch.tensor([[-1.0], [0.0], [2.5]])
    row_labels = torch.tensor([[0.0, 0.25, 0.5, 1.0]])
    broadcast = sigmoid_cross_entropy_with_logits(column_logits, row_labels)
    assert broadcast.shape == (3, 4)
    expanded_expected = functional.binary_cross_entropy_with_logits(
        column_logits.expand(3, 4), row_labels.expand(3, 4), reduction="none"
    )
    torch.testing.assert_close(broadcast, expanded_expected, rtol=1e-6, atol=1e-6)


def test_sigmoid_cross_entropy_special_values_and_symmetry():
    """Zero logits always yield ln 2 and flipping both arguments keeps the loss."""
    zero_logits = torch.zeros(4)
    labels = torch.tensor([0.0, 0.2, 0.7, 1.0])
    zero_values = sigmoid_cross_entropy_with_logits(zero_logits, labels)
    torch.testing.assert_close(zero_values, torch.full((4,), math.log(2.0)), rtol=1e-6, atol=1e-6)

    logits = torch.tensor([-3.0, -0.5, 1.2, 4.0])
    soft_labels = torch.tensor([0.1, 0.3, 0.8, 1.0])
    forward = sigmoid_cross_entropy_with_logits(logits, soft_labels)
    flipped = sigmoid_cross_entropy_with_logits(-logits, 1.0 - soft_labels)
    torch.testing.assert_close(forward, flipped, rtol=1e-6, atol=1e-6)

    hard_labels = (soft_labels > 0.5).float()
    hard_values = sigmoid_cross_entropy_with_logits(logits, hard_labels)
    assert torch.all(hard_values >= 0.0)


def test_sigmoid_cross_entropy_soft_label_minimum_equals_entropy():
    """Soft labels minimise at the matching logit with the binary entropy as value."""
    probs = torch.tensor([0.1, 0.3, 0.5, 0.8, 0.95])
    optimal_logits = torch.logit(probs)
    minimum = sigmoid_cross_entropy_with_logits(optimal_logits, probs)
    expected = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
    torch.testing.assert_close(minimum, expected, rtol=1e-5, atol=1e-5)
    for shift in (0.5, -0.5):
        perturbed = sigmoid_cross_entropy_with_logits(optimal_logits + shift, probs)
        assert torch.all(perturbed > minimum)


def test_sigmoid_cross_entropy_extreme_logits_numerical_stability():
    """Extreme logits stay finite and fall back to the linear softplus regime."""
    logits = torch.tensor([-100.0, -50.0, 50.0, 100.0])
    zero_values = sigmoid_cross_entropy_with_logits(logits, torch.zeros(4))
    one_values = sigmoid_cross_entropy_with_logits(logits, torch.ones(4))
    assert torch.isfinite(zero_values).all()
    assert torch.isfinite(one_values).all()
    expected_zero = torch.tensor([0.0, 0.0, 50.0, 100.0])
    expected_one = torch.tensor([100.0, 50.0, 0.0, 0.0])
    torch.testing.assert_close(zero_values, expected_zero, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(one_values, expected_one, rtol=0.0, atol=1e-6)


def test_sigmoid_cross_entropy_gradient_and_empty_tensor():
    """The gradient equals sigmoid minus label and empty inputs stay empty."""
    logits = torch.tensor([[-2.0, 0.0, 1.5]], requires_grad=True)
    labels = torch.tensor([[0.0, 0.4, 1.0]])
    sigmoid_cross_entropy_with_logits(logits, labels).sum().backward()
    expected = torch.sigmoid(logits.detach()) - labels
    torch.testing.assert_close(logits.grad, expected, rtol=1e-6, atol=1e-6)

    empty_logits = torch.empty(0, 3)
    empty_values = sigmoid_cross_entropy_with_logits(empty_logits, empty_logits)
    assert empty_values.shape == (0, 3)


# Group E: factorial Bernoulli utility.


def test_factorial_bernoulli_reparameterize_raises_during_training():
    """Training mode is rejected because Bernoulli sampling is not differentiable."""
    dist = FactorialBernoulliUtil(torch.zeros(2, 3))
    with pytest.raises(NotImplementedError, match="not differentiable during training"):
        dist.reparameterize(is_training=True)


def test_factorial_bernoulli_reparameterize_eval_binary_dtype_and_seed():
    """Evaluation samples are reproducible binary tensors of float32 dtype."""
    logits = torch.tensor([[0.2, -0.8, 1.5]], dtype=torch.float64, requires_grad=True)
    dist = FactorialBernoulliUtil(logits)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        first = dist.reparameterize(is_training=False)
        torch.manual_seed(7)
        second = dist.reparameterize(is_training=False)
    assert first.shape == logits.shape
    assert first.dtype == torch.float32
    assert first.requires_grad is False
    assert torch.equal(first, second)
    assert torch.all((first == 0.0) | (first == 1.0))


def test_factorial_bernoulli_reparameterize_extreme_logits_and_mean():
    """Saturated logits are deterministic and moderate logits match their probability."""
    saturated_high = FactorialBernoulliUtil(torch.full((4, 5), 100.0))
    saturated_low = FactorialBernoulliUtil(torch.full((4, 5), -100.0))
    assert torch.all(saturated_high.reparameterize(is_training=False) == 1.0)
    assert torch.all(saturated_low.reparameterize(is_training=False) == 0.0)

    base_logits = torch.tensor([-2.0, 0.0, 1.5])
    batched = FactorialBernoulliUtil(base_logits.expand(20_000, 3))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        samples = batched.reparameterize(is_training=False)
    torch.testing.assert_close(
        samples.mean(dim=0), torch.sigmoid(base_logits), rtol=0.0, atol=0.015
    )


def test_factorial_bernoulli_entropy_exact_symmetry_and_extremes():
    """Entropy is bounded by ln 2, even in the logits, and vanishes when saturated."""
    logits = torch.tensor([-50.0, -2.0, 0.0, 2.0, 50.0])
    entropy_values = FactorialBernoulliUtil(logits).entropy()
    assert entropy_values.shape == logits.shape
    assert torch.isfinite(entropy_values).all()
    assert torch.all(entropy_values >= 0.0)
    assert torch.all(entropy_values <= math.log(2.0) + 1e-6)

    maximum = torch.tensor(math.log(2.0))
    torch.testing.assert_close(entropy_values[2], maximum, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(entropy_values[1], entropy_values[3], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(entropy_values[0], entropy_values[4], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(entropy_values[0], torch.tensor(0.0), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(entropy_values[4], torch.tensor(0.0), rtol=0.0, atol=1e-6)

    probs = torch.sigmoid(logits[1:4])
    expected = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
    torch.testing.assert_close(entropy_values[1:4], expected, rtol=1e-5, atol=1e-6)


def test_factorial_bernoulli_entropy_gradient():
    """The entropy gradient equals minus the logit times the Bernoulli variance."""
    logits = torch.tensor([-1.5, 0.0, 1.5], requires_grad=True)
    FactorialBernoulliUtil(logits).entropy().sum().backward()
    probs = torch.sigmoid(logits.detach())
    expected = -logits.detach() * probs * (1.0 - probs)
    torch.testing.assert_close(logits.grad, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(logits.grad[1], torch.tensor(0.0), rtol=0.0, atol=1e-6)


def test_factorial_bernoulli_log_prob_per_var_exact_normalization_and_grad():
    """Log probabilities match PyTorch Bernoulli, normalise, mirror, and differentiate."""
    logits = torch.tensor([[-2.0, 0.0, 2.5], [1.0, -1.0, 0.5]], requires_grad=True)
    dist = FactorialBernoulliUtil(logits)
    samples = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    log_probs = dist.log_prob_per_var(samples)
    reference = distributions.Bernoulli(logits=logits.detach()).log_prob(samples)
    torch.testing.assert_close(log_probs, reference, rtol=1e-6, atol=1e-6)
    assert torch.all(log_probs <= 0.0)

    zeros = torch.zeros_like(logits)
    ones = torch.ones_like(logits)
    total = torch.exp(dist.log_prob_per_var(zeros)) + torch.exp(dist.log_prob_per_var(ones))
    torch.testing.assert_close(total, torch.ones_like(total), rtol=1e-6, atol=1e-6)

    mirrored = FactorialBernoulliUtil(-logits).log_prob_per_var(ones)
    torch.testing.assert_close(dist.log_prob_per_var(zeros), mirrored, rtol=1e-6, atol=1e-6)

    dist.log_prob_per_var(samples).sum().backward()
    expected_grad = samples - torch.sigmoid(logits.detach())
    torch.testing.assert_close(logits.grad, expected_grad, rtol=1e-6, atol=1e-6)


def test_factorial_bernoulli_empty_tensor_handling():
    """Empty inputs propagate through sampling, entropy, and log probability."""
    empty = torch.empty(0, 4)
    dist = FactorialBernoulliUtil(empty)
    assert dist.reparameterize(is_training=False).shape == (0, 4)
    assert dist.entropy().shape == (0, 4)
    assert dist.log_prob_per_var(empty).shape == (0, 4)


# Group F: mixture of two overlapping smoothing distributions.


def test_mixture_generic_init_and_inherited_entropy():
    """Construction wires the smoothing distribution and keeps the parent entropy."""
    logits = torch.tensor([[0.3, -1.2], [2.0, 0.0]])
    low_beta = MixtureGeneric(logits, 0.5)
    high_beta = MixtureGeneric(logits, 5.0)
    reference = FactorialBernoulliUtil(logits)

    assert MixtureGeneric.num_param == 1
    assert low_beta.num_param == 1
    assert isinstance(low_beta.smoothing_dist, Exponential)
    torch.testing.assert_close(low_beta.smoothing_dist.beta, torch.tensor(0.5))
    torch.testing.assert_close(high_beta.smoothing_dist.beta, torch.tensor(5.0))
    torch.testing.assert_close(low_beta.entropy(), reference.entropy())
    torch.testing.assert_close(high_beta.entropy(), reference.entropy())


def test_mixture_generic_reparameterize_bounds_seed_and_mode_parity():
    """Both training flags yield identical bounded samples and identical gradients."""
    logits = torch.tensor([[0.5, -1.0, 0.0], [2.0, -2.5, 1.0]], requires_grad=True)
    mix = MixtureGeneric(logits, 2.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        training_samples = mix.reparameterize(is_training=True)
        training_samples.sum().backward()
        training_grad = logits.grad.detach().clone()
        logits.grad = None
        torch.manual_seed(99)
        evaluation_samples = mix.reparameterize(is_training=False)
        evaluation_samples.sum().backward()
        evaluation_grad = logits.grad.detach().clone()

    assert training_samples.shape == logits.shape
    assert torch.equal(training_samples, evaluation_samples)
    assert torch.isfinite(training_samples).all()
    assert torch.all(training_samples >= 0.0)
    assert torch.all(training_samples <= 1.0)
    assert torch.equal(training_grad, evaluation_grad)


@pytest.mark.parametrize("beta", [0.5, 2.0, 6.0])
def test_mixture_generic_reparameterize_exact_implicit_gradient(beta):
    """The implicit reparameterization gradient matches the closed form."""
    logits = torch.tensor([[-1.5, -0.3, 0.0, 0.7, 2.0]], requires_grad=True)
    mix = MixtureGeneric(logits, beta)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2024)
        samples = mix.reparameterize(is_training=True)
    weights = torch.tensor([[0.5, 1.0, 2.0, 3.0, 4.0]])
    (samples * weights).sum().backward()

    frozen = samples.detach()
    probs = torch.sigmoid(logits.detach())
    smoothing = mix.smoothing_dist
    pdf_0 = smoothing.pdf(frozen)
    pdf_1 = smoothing.pdf(1.0 - frozen)
    cdf_0 = smoothing.cdf(frozen)
    cdf_1 = 1.0 - smoothing.cdf(1.0 - frozen)
    grad_q = (cdf_0 - cdf_1) / (probs * pdf_1 + (1.0 - probs) * pdf_0)
    expected_grad = weights * grad_q * probs * (1.0 - probs)

    torch.testing.assert_close(logits.grad, expected_grad, rtol=1e-5, atol=1e-6)
    assert torch.all(logits.grad >= 0.0)
    assert torch.isfinite(logits.grad).all()
    assert mix.smoothing_dist.beta.grad is None


def test_mixture_generic_reparameterize_no_grad_input_and_float64():
    """Sampling keeps detached inputs detached and promotes float64 logits."""
    frozen_logits = torch.tensor([[0.2, -0.4]])
    frozen_mix = MixtureGeneric(frozen_logits, 1.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        frozen_samples = frozen_mix.reparameterize(is_training=True)
    assert frozen_samples.requires_grad is False
    assert torch.all(frozen_samples >= 0.0)
    assert torch.all(frozen_samples <= 1.0)

    double_logits = torch.tensor([[0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    double_mix = MixtureGeneric(double_logits, 1.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        double_samples = double_mix.reparameterize(is_training=False)
    assert double_samples.dtype == torch.float64
    double_samples.sum().backward()
    assert double_logits.grad is not None
    assert double_logits.grad.dtype == torch.float64
    assert torch.isfinite(double_logits.grad).all()


def test_mixture_generic_reparameterize_statistical_mean_and_extremes():
    """Sample means follow the two-component mixture expectation."""
    beta = 3.0
    base_logits = torch.tensor([-2.0, 0.0, 2.0])
    batched = MixtureGeneric(base_logits.expand(20_000, 3), beta)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(808)
        samples = batched.reparameterize(is_training=False)

    probs = torch.sigmoid(base_logits)
    component_mean = _truncated_exp_mean(beta)
    expected_mean = (1.0 - probs) * component_mean + probs * (1.0 - component_mean)
    torch.testing.assert_close(samples.mean(dim=0), expected_mean, rtol=0.0, atol=0.015)

    saturated_high = MixtureGeneric(torch.full((4_000,), 50.0), 10.0)
    saturated_low = MixtureGeneric(torch.full((4_000,), -50.0), 10.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(15)
        high_samples = saturated_high.reparameterize(is_training=False)
        low_samples = saturated_low.reparameterize(is_training=False)
    high_mean = high_samples.mean().item()
    low_mean = low_samples.mean().item()
    assert high_mean > 0.85
    assert low_mean < 0.15
    assert high_mean > low_mean


@pytest.mark.parametrize("logit_value, beta", [(-1.5, 1.0), (0.0, 2.0), (1.8, 4.0)])
def test_mixture_generic_log_prob_per_var_formula_and_normalization(logit_value, beta):
    """The mixture log density matches its closed form and integrates to one."""
    mix = MixtureGeneric(torch.tensor(logit_value), beta)
    grid = torch.linspace(0.0, 1.0, 2001)
    log_probs = mix.log_prob_per_var(grid)

    probs = torch.sigmoid(torch.tensor(logit_value))
    smoothing = mix.smoothing_dist
    density = probs * smoothing.pdf(1.0 - grid) + (1.0 - probs) * smoothing.pdf(grid)
    torch.testing.assert_close(log_probs, torch.log(density), rtol=1e-6, atol=1e-6)

    area = torch.trapezoid(torch.exp(log_probs), grid)
    torch.testing.assert_close(area, torch.tensor(1.0), rtol=1e-4, atol=1e-4)
    if logit_value > 0.0:
        assert log_probs[-1] > 0.0


def test_mixture_generic_log_prob_per_var_symmetry_extremes_and_grad():
    """The mixture log density is symmetric, saturates correctly, and differentiates."""
    logits = torch.tensor([[1.2, -0.7]])
    samples = torch.tensor([[0.2, 0.65]])
    forward = MixtureGeneric(logits, 2.0).log_prob_per_var(samples)
    flipped = MixtureGeneric(-logits, 2.0).log_prob_per_var(1.0 - samples)
    torch.testing.assert_close(forward, flipped, rtol=1e-6, atol=1e-6)

    balanced = MixtureGeneric(torch.zeros(2), 2.0)
    torch.testing.assert_close(
        balanced.log_prob_per_var(samples), balanced.log_prob_per_var(1.0 - samples)
    )

    saturated_high = MixtureGeneric(torch.full((1, 2), 50.0), 2.0)
    saturated_low = MixtureGeneric(torch.full((1, 2), -50.0), 2.0)
    high_expected = saturated_high.smoothing_dist.log_pdf(1.0 - samples)
    low_expected = saturated_low.smoothing_dist.log_pdf(samples)
    torch.testing.assert_close(
        saturated_high.log_prob_per_var(samples), high_expected, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        saturated_low.log_prob_per_var(samples), low_expected, rtol=1e-6, atol=1e-6
    )

    grad_logits = torch.tensor([[0.4, -1.1]], requires_grad=True)
    grad_mix = MixtureGeneric(grad_logits, 2.0)
    grad_mix.log_prob_per_var(samples).sum().backward()
    probs = torch.sigmoid(grad_logits.detach())
    smoothing = grad_mix.smoothing_dist
    pdf_0 = smoothing.pdf(samples)
    pdf_1 = smoothing.pdf(1.0 - samples)
    expected_grad = (pdf_1 - pdf_0) / (probs * pdf_1 + (1.0 - probs) * pdf_0)
    expected_grad = expected_grad * probs * (1.0 - probs)
    torch.testing.assert_close(grad_logits.grad, expected_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("beta", [0.5, 2.0, 10.0])
def test_mixture_generic_log_ratio_closed_form_and_antisymmetry(beta):
    """The log ratio equals beta times (2 zeta - 1) and is antisymmetric about one half."""
    mix = MixtureGeneric(torch.zeros(1), beta)
    zeta = torch.linspace(0.0, 1.0, 11)
    values = mix.log_ratio(zeta)
    expected = beta * (2.0 * zeta - 1.0)
    torch.testing.assert_close(values, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mix.log_ratio(1.0 - zeta), -values, rtol=0.0, atol=1e-6)

    torch.testing.assert_close(
        mix.log_ratio(torch.tensor(0.5)), torch.tensor(0.0), rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        mix.log_ratio(torch.tensor(0.0)), torch.tensor(-beta), rtol=0.0, atol=1e-5
    )
    torch.testing.assert_close(
        mix.log_ratio(torch.tensor(1.0)), torch.tensor(beta), rtol=0.0, atol=1e-5
    )


def test_mixture_generic_log_ratio_logit_independence_stability_and_grad():
    """The log ratio ignores the logits, stays finite for large beta, and has slope 2 beta."""
    positive = MixtureGeneric(torch.full((2,), 5.0), 3.0)
    negative = MixtureGeneric(torch.full((2,), -5.0), 3.0)
    zeta = torch.linspace(0.1, 0.9, 5)
    assert torch.equal(positive.log_ratio(zeta), negative.log_ratio(zeta))

    large_beta = MixtureGeneric(torch.zeros(1), 120.0)
    large_values = large_beta.log_ratio(torch.tensor([0.0, 0.5, 1.0]))
    assert torch.isfinite(large_values).all()
    expected = torch.tensor([-120.0, 0.0, 120.0])
    torch.testing.assert_close(large_values, expected, rtol=0.0, atol=1e-4)

    zeta_grad = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
    moderate = MixtureGeneric(torch.zeros(1), 2.5)
    (grad_value,) = torch.autograd.grad(moderate.log_ratio(zeta_grad).sum(), zeta_grad)
    expected_grad = torch.full_like(grad_value, 5.0)
    torch.testing.assert_close(grad_value, expected_grad, rtol=0.0, atol=1e-6)


def test_mixture_generic_empty_tensor_handling():
    """Empty inputs propagate through every public operation of the mixture."""
    empty = torch.empty(0, 4)
    mix = MixtureGeneric(empty, 1.5)
    assert mix.reparameterize(is_training=True).shape == (0, 4)
    assert mix.reparameterize(is_training=False).shape == (0, 4)
    assert mix.log_prob_per_var(empty).shape == (0, 4)
    assert mix.log_ratio(empty).shape == (0, 4)
    assert mix.entropy().shape == (0, 4)
