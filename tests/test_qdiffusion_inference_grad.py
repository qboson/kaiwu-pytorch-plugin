"""Generation should not record autograd, even with trainable backbones."""
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel, QDiffusion, QDiffusionConfig, SequenceTokenSpec,
)


class Proposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 5)
        self.projection = nn.Linear(5, 8)
        self.grad_modes = []

    def forward(self, tokens):
        self.grad_modes.append(torch.is_grad_enabled())
        return self.projection(self.embedding(tokens))


class Energy(EnergyModel):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 3)
        self.projection = nn.Linear(3, 1)
        self.grad_modes = []
        self.fail = False

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        self.grad_modes.append(torch.is_grad_enabled())
        if self.fail:
            raise RuntimeError('controlled energy failure')
        return self.projection(self.embedding(candidate_tokens).mean(dim=1))


def make_model(freeze=False, resample=False):
    torch.manual_seed(123)
    return QDiffusion(
        Proposal(), Energy(), SequenceTokenSpec(3, 0, 1, 2),
        QDiffusionConfig(num_diffusion_timesteps=8, num_candidates=2,
                         disable_resample=not resample, resample_ratio=0.1),
        freeze_proposal=freeze,
    )


def tokens():
    return torch.tensor([[1, 5, 6, 5, 2], [1, 6, 5, 2, 0]])


@pytest.mark.parametrize('freeze', [False, True])
@pytest.mark.parametrize('steps', [0, 2, 8])
@pytest.mark.parametrize('return_state', [False, True])
def test_generate_disables_grad_for_both_models(freeze, steps, return_state):
    model = make_model(freeze)
    saved = []
    def pack(tensor):
        saved.append(tensor)
        return tensor
    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        output = model.generate(tokens(), max_steps=steps, return_state=return_state)
        assert torch.is_grad_enabled()
    assert not saved
    assert model.proposal_model.grad_modes == [False] * steps
    assert model.energy_model.grad_modes == [False] * steps
    if return_state:
        assert not output['output_scores'].requires_grad
        assert output['output_scores'].grad_fn is None
        assert not output['output_tokens'].requires_grad
    else:
        assert output.dtype == torch.long and output.shape == tokens().shape


@pytest.mark.parametrize('resample', [False, True])
def test_generation_matches_manual_inference_loop(resample):
    model = make_model(resample=resample).eval()
    fixed = torch.tensor([[False, True, False, False, False], [False] * 5])
    torch.manual_seed(789)
    actual = model.generate(tokens(), max_steps=4, partial_masks=fixed, return_state=True)
    torch.manual_seed(789)
    with torch.no_grad():
        expected = model.initialize_state(tokens(), max_steps=4, partial_masks=fixed)
        for _ in range(4):
            expected = model.step(expected, partial_masks=fixed)
    assert torch.equal(actual['output_tokens'], expected['output_tokens'])
    assert torch.equal(actual['output_scores'], expected['output_scores'])
    assert actual['output_tokens'][0, 1] == tokens()[0, 1]


@pytest.mark.parametrize('training', [False, True])
def test_generate_preserves_mode_parameters_and_existing_gradients(training):
    model = make_model().train(training)
    # Mixed child modes and requires_grad settings must not be normalized.
    model.energy_model.eval()
    list(model.proposal_model.parameters())[0].requires_grad_(False)
    params = list(model.parameters())
    for p in params:
        p.grad = torch.ones_like(p)
    flags = [p.requires_grad for p in params]
    modes = [m.training for m in model.modules()]
    weights = [p.detach().clone() for p in params]
    model.generate(tokens(), max_steps=3)
    assert [p.requires_grad for p in params] == flags
    assert [m.training for m in model.modules()] == modes
    for p, value in zip(params, weights):
        assert torch.equal(p, value)
        assert torch.equal(p.grad, torch.ones_like(p))


def test_generate_restores_grad_mode_on_failure():
    model = make_model()
    model.energy_model.fail = True
    with torch.enable_grad():
        with pytest.raises(RuntimeError, match='controlled energy failure'):
            model.generate(tokens(), max_steps=2)
        assert torch.is_grad_enabled()
    assert model.energy_model.grad_modes == [False]


def test_training_and_single_step_remain_differentiable_after_generation():
    model = make_model()
    model.generate(tokens(), max_steps=2)
    assert torch.is_grad_enabled()
    model.proposal_model.grad_modes.clear()
    model.energy_model.grad_modes.clear()
    model(tokens()).sum().backward()
    assert all(p.grad is not None for p in model.proposal_model.parameters())
    model.zero_grad(set_to_none=True)
    output = model.objective({'targets': tokens()})
    output['energy_objective'].mean().backward()
    assert all(p.grad is not None for p in model.energy_model.parameters())
    assert model.energy_model.grad_modes and all(model.energy_model.grad_modes)
    state = model.initialize_state(tokens(), max_steps=2)
    state = model.step(state)
    assert state['output_scores'].requires_grad


def test_nested_no_grad_context_is_preserved():
    model = make_model()
    with torch.no_grad():
        model.generate(tokens(), max_steps=2)
        assert not torch.is_grad_enabled()


def test_resampling_forwards_do_not_save_backward_tensors():
    model = make_model(resample=True)
    saved = []
    def pack(tensor):
        saved.append(tensor)
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        output = model.generate(tokens(), max_steps=3, return_state=True)
    # The low repetition threshold forces an additional proposal per step.
    assert len(model.proposal_model.grad_modes) > 3
    assert not any(model.proposal_model.grad_modes)
    assert not any(model.energy_model.grad_modes)
    assert not saved and output['output_scores'].grad_fn is None
