"""num_output=0 must keep all visible columns in both BM conditioning paths."""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "bm_generation_trainer", ROOT / "example" / "bm_generation" / "trainer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RecordingBM:
    def __init__(self):
        self.visible = []

    def condition_sample(self, sampler, visible):
        del sampler
        self.visible.append(visible.clone())
        return visible


def test_zero_output_conditions_on_every_visible_column():
    module = load_trainer()
    bm = _RecordingBM()
    chunk = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    state_v, state_vi = module.process_solve_graph_worker((bm, None, chunk, 0))

    assert bm.visible[1].shape == chunk.shape
    torch.testing.assert_close(state_vi, chunk)
    torch.testing.assert_close(state_v, chunk)


def test_positive_output_uses_the_non_output_prefix():
    module = load_trainer()
    bm = _RecordingBM()
    chunk = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    _, state_vi = module.process_solve_graph_worker((bm, None, chunk, 3))

    assert bm.visible[1].shape == (2, 9)
    torch.testing.assert_close(state_vi, chunk[:, :9])


def test_zero_output_columns_leaves_states_untouched():
    module = load_trainer()
    states = torch.ones(2, 5)

    assert torch.equal(module._zero_output_columns(states.clone(), 0), states)
    masked = module._zero_output_columns(states.clone(), 2)
    assert torch.equal(masked[:, :3], torch.ones(2, 3))
    assert torch.equal(masked[:, 3:], torch.zeros(2, 2))
