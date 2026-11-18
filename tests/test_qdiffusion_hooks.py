"""QDiffusion must honor the module call contract of both backbones."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel, QDiffusion, QDiffusionConfig, SequenceTokenSpec,
)


class Proposal(nn.Module):
    """Small trainable backbone with a forwarded keyword argument."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.output = nn.Linear(4, 8)

    def forward(self, tokens, scale=1.0):
        return self.output(self.embedding(tokens)) * scale


class Scorer(EnergyModel):
    """Conditioned energy with an observable, differentiable score."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.125))

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        difference = (candidate_tokens - noisy_tokens).float() * attention_mask
        return difference.sum(dim=1, keepdim=True) * self.weight


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(7)
        return QDiffusion(
            Proposal(), Scorer(),
            SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
            QDiffusionConfig(num_diffusion_timesteps=8, num_candidates=2,
                             disable_resample=True),
            freeze_proposal=False,
        )


@pytest.mark.parametrize("entrypoint", ["forward", "proposal", "__call__"])
def test_proposal_pre_and_post_hooks_transform_inputs_and_logits(model, entrypoint):
    tokens = torch.tensor([[1, 5, 2, 0]])
    changed_tokens = tokens.clone()
    changed_tokens[:, 1] = 6
    expected = model.proposal_model(changed_tokens, scale=2.0) + 7
    calls = []

    def before(module, args, kwargs):
        assert module is model.proposal_model
        assert kwargs == {"scale": 2.0}
        calls.append("before")
        return (changed_tokens,), kwargs

    def after(module, args, output):
        assert module is model.proposal_model
        assert torch.equal(args[0], changed_tokens)
        calls.append("after")
        return output + 7

    with model.proposal_model.register_forward_pre_hook(before, with_kwargs=True), \
            model.proposal_model.register_forward_hook(after):
        result = getattr(model, entrypoint)(tokens, scale=2.0)
    torch.testing.assert_close(result, expected)
    assert calls == ["before", "after"]
    result.sum().backward()
    assert model.proposal_model.embedding.weight.grad is not None


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_energy_hooks_receive_keywords_and_can_replace_score(model, explicit_mask):
    noisy = torch.tensor([[1, 3, 2, 0]])
    candidates = torch.tensor([[1, 5, 2, 0]])
    mask = torch.tensor([[False, True, False, False]]) if explicit_mask else None
    expected_mask = candidates.ne(0) if mask is None else mask
    replacement = torch.tensor([[1, 6, 2, 0]])
    expected = model.energy_model(noisy, replacement, expected_mask) + 4
    calls = []

    def before(module, args, kwargs):
        assert module is model.energy_model
        assert args == ()
        assert torch.equal(kwargs["noisy_tokens"], noisy)
        assert torch.equal(kwargs["attention_mask"], expected_mask)
        calls.append("before")
        return args, dict(kwargs, candidate_tokens=replacement)

    def after(module, args, kwargs, output):
        assert module is model.energy_model
        assert torch.equal(kwargs["candidate_tokens"], replacement)
        calls.append("after")
        return output + 4

    with model.energy_model.register_forward_pre_hook(before, with_kwargs=True), \
            model.energy_model.register_forward_hook(after, with_kwargs=True):
        actual = model.energy(noisy, candidates, mask)
    torch.testing.assert_close(actual, expected)
    assert calls == ["before", "after"]
    actual.sum().backward()
    torch.testing.assert_close(model.energy_model.weight.grad, torch.tensor(3.0))


def test_objective_uses_hook_outputs_for_both_energy_terms_and_backward(model):
    energies = []
    proposals = []

    def energy_hook(module, args, output):
        del module, args
        transformed = output + 2
        energies.append(transformed)
        return transformed

    def proposal_hook(module, args, output):
        del module, args
        proposals.append(output)

    targets = torch.tensor([[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]])
    with model.energy_model.register_forward_hook(energy_hook), \
            model.proposal_model.register_forward_hook(proposal_hook):
        result = model.objective({"targets": targets})
    assert len(proposals) == 1
    assert len(energies) == 2
    negative = energies[1].reshape(2, 2).mean(dim=1, keepdim=True)
    expected = nn.functional.softplus(energies[0]) + nn.functional.softplus(-negative)
    torch.testing.assert_close(result["energy_objective"], expected)
    expected_grad = torch.autograd.grad(expected.sum(), model.energy_model.weight,
                                        retain_graph=True)[0]
    result["energy_objective"].sum().backward()
    torch.testing.assert_close(model.energy_model.weight.grad, expected_grad)
    assert model.proposal_model.embedding.weight.grad is None


def test_generation_calls_backbone_hooks(model):
    calls = []

    def record(module, args, output):
        del args, output
        calls.append(module)

    tokens = torch.tensor([[1, 3, 3, 2, 0]])
    with model.proposal_model.register_forward_hook(record), \
            model.energy_model.register_forward_hook(record):
        generated = model.generate(tokens, max_steps=1)
    assert generated.shape == tokens.shape
    assert model.proposal_model in calls
    assert model.energy_model in calls
