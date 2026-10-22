"""Check GBRBM module autograd behavior against PyTorch gradient contexts."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
import kaiwu

kaiwu.__path__.insert(0, str(SOURCE / "kaiwu"))
from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.torch_plugin import gbrbm

assert Path(gbrbm.__file__).resolve() == (
    SOURCE / "kaiwu" / "torch_plugin" / "gbrbm.py"
).resolve()

CONTEXTS = [
    "grad",
    "no_grad",
    "inference",
    "no_grad_inside_enable",
    "enable_inside_no_grad",
    "enable_inside_inference",
]


@contextmanager
def gradient_context(name):
    """Select the caller's gradient mode, including nested overriding contexts."""
    factories = {
        "grad": [torch.enable_grad],
        "no_grad": [torch.no_grad],
        "inference": [torch.inference_mode],
        "no_grad_inside_enable": [torch.enable_grad, torch.no_grad],
        "enable_inside_no_grad": [torch.no_grad, torch.enable_grad],
        "enable_inside_inference": [torch.inference_mode, torch.enable_grad],
    }
    with ExitStack() as contexts:
        for factory in factories[name]:
            contexts.enter_context(factory())
        yield


def make_gaussian_model(gaussian_visible):
    """Build a real two-Gaussian/one-Bernoulli module with fixed parameters."""
    model = gbrbm.GaussianBernoulliRestrictedBoltzmannMachine(
        2 if gaussian_visible else 1,
        1 if gaussian_visible else 2,
        is_visible_gaussian=gaussian_visible,
        device="cpu",
    ).double()
    with torch.no_grad():
        model.mu.copy_(torch.tensor([0.25, -0.6], dtype=torch.float64))
        model.log_var.copy_(torch.log(torch.tensor([0.75, 1.5], dtype=torch.float64)))
        model.quadratic_coef.copy_(torch.tensor([[0.8], [-0.55]], dtype=torch.float64))
        model.linear_bias.copy_(torch.tensor([0.4], dtype=torch.float64))
    return model


def make_model(kind):
    """Provide BM and RBM controls for the normal nn.Module forward contract."""
    if kind == "bm":
        return BoltzmannMachine(3, device="cpu").double()
    if kind == "rbm":
        return RestrictedBoltzmannMachine(2, 1, device="cpu").double()
    return make_gaussian_model(kind == "gaussian_visible")


def states(requires_grad=True):
    """Use nontrivial states away from the clipped-variance boundary."""
    return torch.tensor(
        [[0.2, -1.1, 1.0], [1.0, 0.3, 0.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )


def analytic_energy_and_gradients(model, state):
    """Differentiate the Gaussian-Bernoulli energy algebraically."""
    with torch.no_grad():
        gaussian = state[:, :2]
        hidden = state[:, 2:]
        variance = model.log_var.exp()
        displacement = gaussian - model.mu
        interaction = hidden @ model.quadratic_coef.t()
        energy = (
            0.5 * (displacement.square() / variance).sum(dim=-1)
            - ((gaussian / variance) * interaction).sum(dim=-1)
            - (hidden * model.linear_bias).sum(dim=-1)
        )
        state_gradient = torch.cat(
            (
                (displacement - interaction) / variance,
                -(gaussian / variance) @ model.quadratic_coef - model.linear_bias,
            ),
            dim=-1,
        )
        parameter_gradients = (
            (-displacement / variance).sum(dim=0),
            (-0.5 * displacement.square() / variance
             + (gaussian / variance) * interaction).sum(dim=0),
            -(gaussian / variance).t() @ hidden,
            -hidden.sum(dim=0),
        )
    return energy, state_gradient, parameter_gradients


@pytest.mark.parametrize("kind", ["bm", "rbm", "gaussian_visible", "bernoulli_visible"])
@pytest.mark.parametrize("mode", CONTEXTS)
def test_forward_follows_the_callers_gradient_context(kind, mode):
    """GBRBM forward follows the same context rules as real BM/RBM modules."""
    model = make_model(kind)
    state = states()
    outer_grad_mode = torch.is_grad_enabled()
    with gradient_context(mode):
        ambient_grad_mode = torch.is_grad_enabled()
        expect_graph = ambient_grad_mode and not torch.is_inference_mode_enabled()
        actual = model(state)
        assert actual.requires_grad == expect_graph
        assert (actual.grad_fn is not None) == expect_graph
        assert torch.is_grad_enabled() == ambient_grad_mode
    assert torch.is_grad_enabled() == outer_grad_mode


@pytest.mark.parametrize("gaussian_visible", [True, False])
@pytest.mark.parametrize("input_requires_grad", [True, False])
@pytest.mark.parametrize("use_objective", [True, False])
def test_no_grad_does_not_save_backward_tensors(
    gaussian_visible, input_requires_grad, use_objective
):
    """Inference must avoid building hidden backward graphs, including in objectives."""
    model = make_gaussian_model(gaussian_visible)
    state = states(input_requires_grad)
    saved_tensors = []

    def pack(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        with torch.no_grad():
            if use_objective:
                output = model.objective(state, state + 0.3)
            else:
                output = model(state)

    assert not saved_tensors
    assert not output.requires_grad


@pytest.mark.parametrize("gaussian_visible", [True, False])
@pytest.mark.parametrize("training", [True, False])
def test_forward_preserves_analytic_energy_and_training_gradients(
    gaussian_visible, training
):
    """Training and eval modes both allow gradients when the caller enables them."""
    model = make_gaussian_model(gaussian_visible)
    model.train(training)
    state = states()
    expected_energy, expected_state_gradient, expected_parameter_gradients = (
        analytic_energy_and_gradients(model, state)
    )
    with torch.no_grad():
        with torch.enable_grad():
            output = model(state)
            output.sum().backward()

    torch.testing.assert_close(output, expected_energy)
    torch.testing.assert_close(state.grad, expected_state_gradient)
    for parameter, expected in zip(model.parameters(), expected_parameter_gradients):
        torch.testing.assert_close(parameter.grad, expected)


@pytest.mark.parametrize("gaussian_visible", [True, False])
def test_objective_preserves_analytic_parameter_gradients(gaussian_visible):
    """The inherited contrastive objective remains trainable through forward."""
    model = make_gaussian_model(gaussian_visible)
    positive = states(requires_grad=False)
    negative = positive + 0.3
    positive_energy, _, positive_gradients = analytic_energy_and_gradients(model, positive)
    negative_energy, _, negative_gradients = analytic_energy_and_gradients(model, negative)

    objective = model.objective(positive, negative)
    objective.backward()

    torch.testing.assert_close(objective, positive_energy.mean() - negative_energy.mean())
    for parameter, pos, neg in zip(model.parameters(), positive_gradients, negative_gradients):
        torch.testing.assert_close(parameter.grad, (pos - neg) / len(positive))


@pytest.mark.parametrize("gaussian_visible", [True, False])
@pytest.mark.parametrize("enable_grad", [True, False])
@pytest.mark.parametrize("mode", CONTEXTS)
def test_explicit_energy_gradient_flag_keeps_its_override_semantics(
    gaussian_visible, enable_grad, mode
):
    """Explicit energy flags still override no_grad, while inference_mode stays in force."""
    model = make_gaussian_model(gaussian_visible)
    state = states()
    expected_energy, _, _ = analytic_energy_and_gradients(model, state)
    with gradient_context(mode):
        ambient_grad_mode = torch.is_grad_enabled()
        expect_graph = enable_grad and not torch.is_inference_mode_enabled()
        output = model.energy(state, enable_grad=enable_grad)
        assert output.requires_grad == expect_graph
        assert torch.is_grad_enabled() == ambient_grad_mode
    torch.testing.assert_close(output, expected_energy)


@pytest.mark.parametrize("gaussian_visible", [True, False])
def test_energy_keeps_its_default_disabled_gradient_mode(gaussian_visible):
    """The explicit energy API keeps its existing no-gradient default."""
    model = make_gaussian_model(gaussian_visible)
    with torch.enable_grad():
        output = model.energy(states())
        assert not output.requires_grad
        assert torch.is_grad_enabled()
