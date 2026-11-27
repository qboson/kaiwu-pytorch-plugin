"""Per-candidate diagnostic weighting regressions for EnergyModel and QDiffusion."""

import torch
from torch import nn

import kaiwu as kw
from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


def _configure_degenerate_visible_bm(model: EnergyModel) -> None:
    """Set BM weights so v=[0,0] has 4 degenerate hidden ground states and v=[1,1] has 1."""
    with torch.no_grad():
        model.energy_bm.quadratic_coef.zero_()
        model.energy_bm.quadratic_coef[0, 2] = 1.0
        model.energy_bm.quadratic_coef[1, 2] = 1.0
        model.energy_bm.quadratic_coef[0, 3] = 1.0
        model.energy_bm.quadratic_coef[1, 3] = 1.0
        model.energy_bm.linear_bias.zero_()


class ConstantTokenProposal(nn.Module):
    """Deterministic proposal preferring ordinary token 5."""

    def __init__(self) -> None:
        super().__init__()
        logits = torch.full((8,), -100.0)
        logits[5] = 100.0
        self.register_buffer("logits", logits)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Broadcasts token logits across batch and sequence axes."""
        return self.logits.expand(*tokens.shape, -1)


class TokenThresholdEnergyModel(EnergyModel):
    """Maps sequence content into degenerate vs unique BM visible conditions."""

    def __init__(self) -> None:
        super().__init__(
            bm_num_visible=2,
            bm_num_hidden=2,
            sampler=kw.classical.BruteForceOptimizer(),
        )
        _configure_degenerate_visible_bm(self)
        self.scale = nn.Parameter(torch.tensor(100.0))

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Maps token 6 rows to +scale logits and other rows to -scale logits."""
        del noisy_tokens
        has_six = ((candidate_tokens == 6) & attention_mask).any(dim=-1, keepdim=True)
        signs = torch.where(has_six, 1.0, -1.0).to(self.scale.dtype)
        visible_logits = signs.expand(-1, 2) * self.scale
        return self.score_visible_logits(visible_logits)


def test_score_visible_logits_stats_are_unbiased_under_uneven_sdk_degeneracy() -> None:
    """Uneven solver solution counts per row must not skew visible or hidden on-ratios."""
    model = EnergyModel(
        bm_num_visible=2,
        bm_num_hidden=2,
        sampler=kw.classical.BruteForceOptimizer(),
    )
    _configure_degenerate_visible_bm(model)
    visible_logits = torch.tensor(
        [[-100.0, -100.0], [100.0, 100.0]], requires_grad=True
    )

    _, split_sizes = model.sample_hidden_state(
        model.discretize_visible_state(visible_logits.detach())
    )
    assert split_sizes.tolist() == [8, 2]

    scores = model.score_visible_logits(visible_logits)
    stats = model.get_last_stats()

    torch.testing.assert_close(scores, torch.tensor([[0.0], [-4.0]]))
    torch.testing.assert_close(stats["visible_on_ratio"], torch.tensor(0.5))
    torch.testing.assert_close(stats["hidden_on_ratio"], torch.tensor(0.75))
    for stat_tensor in stats.values():
        assert not stat_tensor.requires_grad
        assert stat_tensor.grad_fn is None

    scores.sum().backward()
    assert visible_logits.grad is not None
    assert torch.isfinite(visible_logits.grad).all()


def test_qdiffusion_objective_on_ratios_weight_each_sequence_and_candidate_equally() -> None:
    """Objective positive/negative BM on-ratios match equal per-row weighting."""
    energy_model = TokenThresholdEnergyModel()
    generator = QDiffusion(
        proposal_model=ConstantTokenProposal(),
        energy_model=energy_model,
        token_spec=SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
        config=QDiffusionConfig(
            num_candidates=2,
            proposal_temperature=0.0,
            proposal_noise_scale=0.0,
            disable_resample=True,
        ),
    )
    # Row 0 has only token 7 -> visible [0, 0] (8 solver rows, hidden mean 0.5).
    # Rows 1 and 2 have token 6 -> visible [1, 1] (2 solver rows each, hidden mean 1.0).
    targets = torch.tensor(
        [
            [1, 7, 2, 0],
            [1, 6, 2, 0],
            [1, 6, 2, 0],
        ]
    )
    torch.manual_seed(0)
    outputs = generator.objective({"targets": targets})

    torch.testing.assert_close(
        outputs["positive_visible_on_ratio"], torch.tensor(2.0 / 3.0)
    )
    torch.testing.assert_close(
        outputs["positive_hidden_on_ratio"], torch.tensor((0.5 + 1.0 + 1.0) / 3.0)
    )
    # Every editable position in negative candidates proposes token 5 -> all rows [0, 0].
    torch.testing.assert_close(
        outputs["negative_visible_on_ratio"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        outputs["negative_hidden_on_ratio"], torch.tensor(0.5)
    )
