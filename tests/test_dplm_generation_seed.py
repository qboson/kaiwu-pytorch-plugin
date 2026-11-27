"""The DPLM comparison should seed decoding independently of diagnostics."""

import pytest
import torch

pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from example.qdiffusion.dplm.workflows.workflow_helpers import run_generation_over_records


class ToyTokenizer:
    def __call__(self, sequence, **kwargs):
        return {"input_ids": torch.tensor([[0, *[ord(letter) for letter in sequence], 1]])}

    def batch_decode(self, tokens, **kwargs):
        return ["".join(chr(token) for token in row[1:-1].tolist()) for row in tokens]


class LocalGenerator:
    device = "cpu"
    tokenizer = ToyTokenizer()

    def __init__(self, diagnostic_draws):
        self.diagnostic_draws = diagnostic_draws
        self.decode_entry_states = []

    def objective(self, batch):
        return {"energy_objective": torch.rand(self.diagnostic_draws)}

    def generate(self, tokens, **kwargs):
        self.decode_entry_states.append(torch.random.get_rng_state().clone())
        generated = tokens.clone()
        generated[:, 1:-1] = torch.randint(ord("A"), ord("Z") + 1, tokens[:, 1:-1].shape)
        return generated


def test_objective_draw_count_cannot_change_seeded_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    records = [("r1", "ACDEFGHIK"), ("r2", "LMNPQRSTV")]
    baseline = LocalGenerator(diagnostic_draws=3)
    guided = LocalGenerator(diagnostic_draws=17)
    kwargs = {"max_steps": 2, "seed_base": 45, "output_dir": tmp_path}

    with torch.random.fork_rng(devices=[]):
        first, first_rows = run_generation_over_records(baseline, records, label="first", **kwargs)
        second, second_rows = run_generation_over_records(guided, records, label="second", **kwargs)

    assert first == second
    assert first[0][1] != first[1][1]
    assert first_rows[0]["energy_objective_mean"] != second_rows[0]["energy_objective_mean"]
    for baseline_state, guided_state in zip(baseline.decode_entry_states, guided.decode_entry_states):
        torch.testing.assert_close(baseline_state, guided_state)
