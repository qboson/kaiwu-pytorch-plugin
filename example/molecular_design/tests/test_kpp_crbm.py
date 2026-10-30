"""Offline KPP integration contracts; run each source tree in a fresh process.

QC_MODEL_SOURCE=sqa (default) or cim_sa selects the example package. Requires
the real kaiwu.torch_plugin implementation, PyTorch, dimod, and dwave-neal.
No license initialization, CIM submission, checkpoint, or dataset is needed.
"""
import itertools
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("KPP_STATS_ENABLED", "false")
ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("QC_MODEL_SOURCE", "sqa")
if SOURCE not in ("cim_sa", "sqa"):
    raise ValueError("QC_MODEL_SOURCE must be cim_sa or sqa")
sys.path.insert(0, str(ROOT / SOURCE))

import dimod
import numpy as np
import torch
from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
from qc_mol_design.config import CFG, N_BINS_PER_PROP
from qc_mol_design.models.crbm import CRBM
from qc_mol_design.training.cd_trainer import _cd_step
from qc_mol_design.training.qc_trainer import _NealIsingSampler, _qc_step, _sample_neg_phase_sa


def binary_states(n):
    return torch.tensor(list(itertools.product((0., 1.), repeat=n)))


def categorical_states():
    indices = torch.tensor(list(itertools.product(range(2), repeat=3)))
    return torch.nn.functional.one_hot(indices, 2).reshape(-1, 6).float()


class KPPContracts(unittest.TestCase):
    def setUp(self):
        for context in (
            patch.dict(N_BINS_PER_PROP, {"qed": 2, "logp": 2, "sas": 2}),
            patch.object(CFG, "n_hidden", 2),
            patch.object(CFG, "fp_dim", 3),
        ):
            context.start()
            self.addCleanup(context.stop)
        torch.manual_seed(23)
        self.model = CRBM()
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.normal_(0, 0.4)

    def test_real_rbm_energy_and_gradients(self):
        model = self.model.double()
        self.assertIsInstance(model.rbm, RestrictedBoltzmannMachine)
        self.assertEqual(model.rbm.device, model.W.device)
        f = torch.randn(5, 3, dtype=torch.float64)
        y = torch.randint(2, (5, 6)).double()
        h = torch.randint(2, (5, 2)).double()
        old = {name: getattr(model, name).detach().clone().requires_grad_()
               for name in ("b", "c", "W", "W1", "W2")}
        expected = (-(y * (old["b"] + f @ old["W1"])).sum(-1)
                    -(h * (old["c"] + f @ old["W2"])).sum(-1)
                    -(y @ old["W"] * h).sum(-1))
        calls = []
        hook = model.rbm.register_forward_hook(lambda *_: calls.append(True))
        try:
            actual = model.energy(y, h, f)
        finally:
            hook.remove()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.sum().backward()
        expected.sum().backward()
        for name, grad in (
            ("b", model.rbm.linear_bias.grad[:6]),
            ("c", model.rbm.linear_bias.grad[6:]),
            ("W", model.W.grad), ("W1", model.W1.grad), ("W2", model.W2.grad),
        ):
            torch.testing.assert_close(grad, old[name].grad)

    def test_legacy_and_native_checkpoints_including_nested_module(self):
        old = {name: getattr(self.model, name).detach().clone()
               for name in ("b", "c", "W", "W1", "W2")}
        restored = CRBM()
        restored.load_state_dict(old, strict=True)
        self.assertEqual(set(old), {"b", "c", "W", "W1", "W2"})
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)
        native = CRBM()
        native.load_state_dict(restored.state_dict(), strict=True)
        self.assertEqual(set(native.state_dict()),
                         {"W1", "W2", "rbm.quadratic_coef", "rbm.linear_bias"})
        wrapper = torch.nn.ModuleDict({"conditional": CRBM()})
        wrapper.load_state_dict({"conditional." + k: v for k, v in old.items()})
        torch.testing.assert_close(wrapper["conditional"].W, self.model.W)
        with self.assertRaisesRegex(RuntimeError, "incomplete legacy"):
            restored.load_state_dict({k: v for k, v in old.items() if k != "b"})
        with self.assertRaisesRegex(RuntimeError, "mixed legacy"):
            restored.load_state_dict({**old, **native.state_dict()})
        with self.assertRaisesRegex(RuntimeError, "shapes do not match"):
            restored.load_state_dict({**old, "b": old["b"][:-1]})

    def test_categorical_conditionals_and_exact_marginals(self):
        model = self.model
        f = torch.randn(1, 3)
        labels, hidden = categorical_states(), binary_states(2)
        joint_energy = model.energy(
            labels.repeat_interleave(4, dim=0), hidden.repeat(8, 1), f.expand(32, -1),
        ).reshape(8, 4)
        expected = -torch.logsumexp(-joint_energy, dim=1)
        torch.testing.assert_close(model.free_energy(labels, f.expand(8, -1)), expected)
        torch.testing.assert_close(model.infer_y_map(f), labels[expected.argmin()].unsqueeze(0))
        target = -torch.logsumexp(-expected[labels[:, 1] == 1], dim=0)
        torch.testing.assert_close(model.target_free_energy("qed", 1, f)[0], target)
        hidden_mean = torch.softmax(-joint_energy, dim=1) @ hidden
        torch.testing.assert_close(model.p_h_given_yf(labels, f.expand(8, -1)), hidden_mean)
        logits = model.p_y_given_hf_logits(hidden, f.expand(4, -1)).double()
        samples = model.sample_y_from_logits(logits)
        self.assertEqual(samples.dtype, torch.float64)
        torch.testing.assert_close(samples.reshape(4, 3, 2).sum(-1), torch.ones(4, 3).double())

    def test_bm_qubo_and_ising_energies_exhaustively(self):
        f = torch.randn(3)
        states = binary_states(8)
        b, c = self.model.effective_biases(f.unsqueeze(0))
        unpenalized = torch.diag(-torch.cat([b[0], c[0]]))
        unpenalized[:6, 6:] = -self.model.W
        auto_rho = max(10., 2 * unpenalized.abs().sum(1).max().item() + 1.)
        for supplied, rho in ((None, auto_rho), (0., auto_rho), (7., 7.)):
            bm = self.model.conditioned_bm(f, supplied)
            self.assertIsInstance(bm, BoltzmannMachine)
            self.assertTrue(all(not p.requires_grad for p in bm.parameters()))
            q = self.model.build_neg_phase_qubo(f, supplied)
            penalties = (states[:, :6].reshape(-1, 3, 2).sum(-1) - 1).square().sum(-1)
            expected = (self.model.energy(states[:, :6], states[:, 6:], f.expand(256, -1))
                        + rho * (penalties - 3))
            torch.testing.assert_close(bm(states), expected, rtol=2e-5, atol=1e-5)
            torch.testing.assert_close((states @ q * states).sum(-1), expected, rtol=2e-5, atol=1e-5)
            matrix = bm.get_ising_matrix().astype(np.float64)
            logical = 2 * states.numpy().astype(np.float64) - 1
            for auxiliary in (-1, 1):
                physical = np.column_stack([logical * auxiliary, np.full(256, auxiliary)])
                ising = -np.einsum("bi,ij,bj->b", physical, matrix, physical)
                difference = ising - expected.detach().numpy()
                np.testing.assert_allclose(difference, difference[0], rtol=0, atol=2e-5)
        with self.assertRaisesRegex(ValueError, "one fingerprint"):
            self.model.conditioned_bm(f.unsqueeze(0))

    def test_neal_adapter_sign_variable_order_and_multiplicity(self):
        matrix = np.array([[0., 2., -3.], [2., 0., 1.], [-3., 1., 0.]])
        # Explicit non-sorted variable order and an aggregated row with 3 reads.
        response = dimod.SampleSet.from_samples(
            (np.array([[1, -1, 1], [-1, 1, 1]]), [2, 0, 1]),
            vartype="SPIN", energy=[-2., 2.], num_occurrences=[3, 1],
        )
        with patch("neal.SimulatedAnnealingSampler.sample_ising", return_value=response) as solve:
            spins = _NealIsingSampler(4, 10).solve(matrix)
        h, j = solve.call_args.args
        for state in itertools.product((-1, 1), repeat=3):
            energy = sum(h[i] * state[i] for i in h) + sum(
                value * state[i] * state[k] for (i, k), value in j.items()
            )
            self.assertEqual(energy, -np.asarray(state) @ matrix @ np.asarray(state))
        np.testing.assert_array_equal(spins, [[-1, 1, 1]] * 3 + [[1, 1, -1]])

    def test_kpp_decoding_conditions_and_no_onehot_projection(self):
        # Deliberately invalid visible groups: retain exactly what was sampled.
        logical = torch.tensor([[1., 1., 0., 0., 1., 0., 1., 0.]])
        physical = np.r_[2 * logical.numpy()[0] - 1, 1].astype(np.int8)
        rows = np.stack([physical, -physical, physical])
        with patch.object(_NealIsingSampler, "solve", return_value=rows) as solve:
            y, h, conditions = _sample_neg_phase_sa(
                self.model, torch.randn(3, 3), n_reads=3, n_sweeps=10, max_per_sample=2,
            )
        self.assertEqual(solve.call_count, 2)
        torch.testing.assert_close(torch.cat([y, h], -1), logical.expand(6, -1))
        torch.testing.assert_close(conditions, torch.tensor([0, 0, 0, 1, 1, 1]))
        with self.assertRaisesRegex(ValueError, "batch-mean"):
            _sample_neg_phase_sa(self.model, torch.randn(1, 3), per_sample=False)

    def test_local_neal_and_training_update_kpp_parameters(self):
        f = torch.randn(3, 3)
        y = categorical_states()[:3]
        y_neg, h_neg, indices = _sample_neg_phase_sa(
            self.model, f, n_reads=4, n_sweeps=20, max_per_sample=0,
        )
        self.assertEqual(y_neg.shape, (12, 6))
        self.assertEqual(h_neg.shape, (12, 2))
        torch.testing.assert_close(indices, torch.arange(3).repeat_interleave(4))
        self.assertTrue(torch.isin(torch.cat([y_neg, h_neg], -1), torch.tensor([0., 1.])).all())
        before = self.model.rbm.linear_bias.detach().clone()
        result = _qc_step(self.model, y, f, lr=0.03, n_reads=4, n_sweeps=20)
        self.assertTrue(math.isfinite(result))
        self.assertFalse(torch.equal(before, self.model.rbm.linear_bias))
        before_cd = self.model.rbm.linear_bias.detach().clone()
        result = _cd_step(self.model, y, f, lr=0.03, k=2)
        self.assertTrue(math.isfinite(result))
        self.assertFalse(torch.equal(before_cd, self.model.rbm.linear_bias))
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_nested_kpp_device_move(self):
        model = self.model.cuda()
        self.assertEqual(model.rbm.device, model.W.device)
        energy = model.energy(torch.ones(1, 6, device="cuda"),
                              torch.ones(1, 2, device="cuda"), torch.zeros(1, 3, device="cuda"))
        self.assertTrue(torch.isfinite(energy).all())
        bm = model.conditioned_bm(torch.zeros(3, device="cuda"))
        self.assertEqual(bm.linear_bias.device.type, "cpu")
        model.cpu()
        self.assertEqual(model.rbm.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
