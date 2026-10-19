"""Reject invalid allocation counts before entering QDiffusion tensor paths."""

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


@pytest.mark.parametrize("name", ["num_diffusion_timesteps", "num_candidates"])
@pytest.mark.parametrize("value", [0, -1, 1.5, 2.0, True, False, "2", None, float("nan")])
def test_config_rejects_invalid_counts(name, value):
    """Invalid counts identify their field instead of failing inside Torch later."""
    with pytest.raises(ValueError, match=rf"{name} must be a positive integer"):
        QDiffusionConfig(**{name: value})


class TokenSumEnergy(EnergyModel):
    """A differentiable offline scorer for exercising the real objective."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Return one learned token-sum score per candidate sequence."""
        del noisy_tokens
        return self.scale * (candidate_tokens * attention_mask).sum(1, keepdim=True)


@pytest.mark.parametrize("count_type", [int, np.int32, np.int64])
@pytest.mark.parametrize("coupled", [False, True])
def test_positive_integer_counts_run_objective(count_type, coupled):
    """The minimum timestep count and multiple candidates retain both objectives."""
    config = QDiffusionConfig(
        num_diffusion_timesteps=count_type(1),
        num_candidates=count_type(2),
        use_coupled_sampling=coupled,
        disable_resample=True,
    )
    # An embedding maps token ids directly to proposal logits, with no downloads.
    model = QDiffusion(
        nn.Embedding(7, 7), TokenSumEnergy(),
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2),
        config=config, device="cpu",
    )
    targets = torch.tensor([[1, 4, 5, 2], [1, 6, 4, 2]])
    result = model.objective({"targets": targets})
    expected_rows = 4 if coupled else 2
    assert result["energy_objective"].shape == (expected_rows, 1)
    assert torch.isfinite(result["energy_objective"]).all()
    assert torch.equal(result["weight"], torch.ones(expected_rows, 1))
    result["energy_objective"].mean().backward()
    assert torch.isfinite(model.energy_model.scale.grad)


def test_default_counts_remain_unchanged():
    """Default experiment settings are unaffected by validation."""
    config = QDiffusionConfig()
    assert config.num_diffusion_timesteps == 500
    assert config.num_candidates == 1
