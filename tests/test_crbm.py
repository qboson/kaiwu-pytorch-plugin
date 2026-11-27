"""Unit tests for ConditionalRestrictedBoltzmannMachine (CRBM)."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import torch
from kaiwu.classical import SimulatedAnnealingOptimizer

import kaiwu.torch_plugin as plugin
from kaiwu.torch_plugin import (
    ConditionalRestrictedBoltzmannMachine,
    RestrictedBoltzmannMachine,
)
from kaiwu.torch_plugin.abstract_boltzmann_machine import AbstractBoltzmannMachine


def _all_binary_states(num_bits: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Enumerate all 2**num_bits binary vectors."""
    return torch.tensor(list(itertools.product((0.0, 1.0), repeat=num_bits)), dtype=dtype)


class LocalSimulatedAnnealingSampler:
    """Deterministic license-free Ising SA sampler maximizing ``s^T J s``.

    Matches the Kaiwu optimizer interface ``solve(ising_mat) -> spins`` where
    each row is a spin configuration in ``{-1, +1}^{N+1}`` minimizing
    ``H(s) = -s^T J s``.
    """

    def __init__(
        self,
        num_reads: int = 8,
        num_sweeps: int = 40,
        t_start: float = 5.0,
        t_end: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.t_start = t_start
        self.t_end = t_end
        self.rng = np.random.default_rng(seed)

    def solve(self, ising_mat: np.ndarray) -> np.ndarray:
        """Return ``num_reads`` annealed spin rows in ``{-1, +1}``."""
        coupling = (ising_mat + ising_mat.T) * 0.5
        np.fill_diagonal(coupling, 0.0)
        num_spins = coupling.shape[0]
        temps = np.geomspace(self.t_start, self.t_end, self.num_sweeps)
        rows = []
        for _ in range(self.num_reads):
            spins = self.rng.choice([-1.0, 1.0], size=num_spins)
            field = coupling @ spins
            for temp in temps:
                for idx in self.rng.permutation(num_spins):
                    # Flipping s_i changes H = -s^T J s by delta_e = 4 s_i (J s)_i
                    delta_e = 4.0 * spins[idx] * field[idx]
                    if delta_e <= 0.0 or self.rng.random() < np.exp(-delta_e / temp):
                        spins[idx] = -spins[idx]
                        field += 2.0 * spins[idx] * coupling[:, idx]
            rows.append(spins.copy())
        return np.stack(rows, axis=0)


class _BadShapeSampler:
    """Sampler returning malformed spin arrays for contract validation."""

    def __init__(self, output):
        self.output = output

    def solve(self, _ising_mat):
        return self.output


class TestCRBMConstructionAndValidation(unittest.TestCase):
    """Exports, inheritance, parameter initialization, and input validation."""

    def test_exported_and_subclass(self):
        self.assertIn("ConditionalRestrictedBoltzmannMachine", plugin.__all__)
        model = ConditionalRestrictedBoltzmannMachine(4, 3, 2, device="cpu")
        self.assertIsInstance(model, AbstractBoltzmannMachine)
        self.assertEqual(model.num_visible, 4)
        self.assertEqual(model.num_condition, 3)
        self.assertEqual(model.num_hidden, 2)
        self.assertEqual(model.num_nodes, 6)
        self.assertEqual(model.dtype, torch.float32)
        self.assertEqual(model.device, torch.device("cpu"))
        self.assertIsNotNone(model.cond_visible_coef)

    def test_use_cond_visible_false(self):
        model = ConditionalRestrictedBoltzmannMachine(
            4, 3, 2, use_cond_visible=False, device="cpu"
        )
        self.assertFalse(model.use_cond_visible)
        self.assertIsNone(model.cond_visible_coef)
        param_names = {name for name, _ in model.named_parameters()}
        self.assertEqual(
            param_names, {"quadratic_coef", "linear_bias", "cond_hidden_coef"}
        )

    def test_invalid_constructor_arguments(self):
        for bad in (0, -2, True, 2.5, "4"):
            with self.assertRaises(ValueError):
                ConditionalRestrictedBoltzmannMachine(bad, 2, 2, device="cpu")
            with self.assertRaises(ValueError):
                ConditionalRestrictedBoltzmannMachine(2, bad, 2, device="cpu")
            with self.assertRaises(ValueError):
                ConditionalRestrictedBoltzmannMachine(2, 2, bad, device="cpu")

        with self.assertRaises(TypeError):
            ConditionalRestrictedBoltzmannMachine(
                2, 2, 2, use_cond_visible=1, device="cpu"
            )
        with self.assertRaises(ValueError):
            ConditionalRestrictedBoltzmannMachine(
                2,
                3,
                2,
                use_cond_visible=False,
                cond_visible_coef=torch.zeros(3, 2),
                device="cpu",
            )
        with self.assertRaises(ValueError):
            ConditionalRestrictedBoltzmannMachine(
                2, 2, 2, dtype=torch.int64, device="cpu"
            )
        with self.assertRaises(ValueError):
            ConditionalRestrictedBoltzmannMachine(
                2, 2, 2, h_range=(1.0, -1.0), device="cpu"
            )
        with self.assertRaises(ValueError):
            ConditionalRestrictedBoltzmannMachine(
                2, 2, 2, j_range=(-float("inf"), 1.0), device="cpu"
            )
        with self.assertRaises(ValueError):
            ConditionalRestrictedBoltzmannMachine(
                2, 2, 2, h_range=(False, True), device="cpu"
            )
        with self.assertRaises(ValueError):
            ConditionalRestrictedBoltzmannMachine(
                2, 2, 2, h_range=(1.0,), device="cpu"
            )
        with self.assertRaises(ValueError):
            ConditionalRestrictedBoltzmannMachine(
                2, 3, 4, quadratic_coef=torch.zeros(2, 5), device="cpu"
            )
        with self.assertRaises(TypeError):
            ConditionalRestrictedBoltzmannMachine(
                2, 3, 4, linear_bias=[0.0] * 6, device="cpu"
            )

    def test_invalid_runtime_inputs(self):
        model = ConditionalRestrictedBoltzmannMachine(3, 2, 2, device="cpu")
        with self.assertRaises(ValueError):
            model.forward(torch.zeros(4, 4))
        with self.assertRaises(TypeError):
            model.forward([[0.0] * 5])
        with self.assertRaises(ValueError):
            model.energy(torch.zeros(2, 3), torch.zeros(3, 2))
        with self.assertRaises(ValueError):
            model.free_energy(torch.zeros(3, 3), condition=torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            model.free_energy(torch.zeros(2, 3), condition=torch.zeros(2, 5))
        with self.assertRaises(TypeError):
            model.free_energy(torch.zeros(2, 3), condition=[1.0, 0.0])
        with self.assertRaises(ValueError):
            model.condition_ising_matrix(torch.zeros(0, 2))
        with self.assertRaises(ValueError):
            model.condition_sample(_BadShapeSampler(np.ones((1, 6))), torch.zeros(0, 2))
        with self.assertRaises(RuntimeError):
            model.condition_sample(_BadShapeSampler(np.ones((2, 4))), torch.zeros(2))
        with self.assertRaises(ValueError):
            model.gibbs_sample(num_steps=0, num_samples=2)
        with self.assertRaises(ValueError):
            model.gibbs_sample()
        with self.assertRaises(ValueError):
            model.gibbs_sample(
                condition=torch.zeros(3, 2), s_visible=torch.zeros(3, 3), num_samples=2
            )
        with self.assertRaises(ValueError):
            model.gibbs_sample(condition=torch.zeros(3, 2), num_samples=2)


class TestCRBMEnergyAndDistributions(unittest.TestCase):
    """Exact energy, free energy, conditionals, and gradient flow checks."""

    def _make_model(self, use_cond_visible: bool, dtype: torch.dtype = torch.float64):
        gen = torch.Generator().manual_seed(42)
        w = torch.randn(3, 2, generator=gen, dtype=dtype)
        b = torch.randn(5, generator=gen, dtype=dtype)
        u = torch.randn(4, 2, generator=gen, dtype=dtype)
        a = torch.randn(4, 3, generator=gen, dtype=dtype) if use_cond_visible else None
        return ConditionalRestrictedBoltzmannMachine(
            num_visible=3,
            num_condition=4,
            num_hidden=2,
            use_cond_visible=use_cond_visible,
            quadratic_coef=w,
            linear_bias=b,
            cond_hidden_coef=u,
            cond_visible_coef=a,
            dtype=dtype,
            device="cpu",
        )

    def test_energy_matches_analytic_formula_and_rbm_reduction(self):
        for use_cond_visible in (True, False):
            with self.subTest(use_cond_visible=use_cond_visible):
                model = self._make_model(use_cond_visible=use_cond_visible)
                gen = torch.Generator().manual_seed(7)
                v = torch.randint(0, 2, (5, 3), generator=gen).double()
                h = torch.randint(0, 2, (5, 2), generator=gen).double()
                c = torch.randn(5, 4, generator=gen, dtype=torch.float64)

                expected = (
                    -torch.sum((v @ model.quadratic_coef) * h, dim=-1)
                    - torch.sum((c @ model.cond_hidden_coef) * h, dim=-1)
                    - v @ model.visible_bias
                    - h @ model.hidden_bias
                )
                if use_cond_visible:
                    expected = expected - torch.sum(
                        (c @ model.cond_visible_coef) * v, dim=-1
                    )

                s_all = torch.cat([v, h], dim=-1)
                torch.testing.assert_close(model.energy(v, h, c), expected)
                torch.testing.assert_close(model(s_all, c), expected)

                # Effective biases for None, 1D, and 2D conditions
                bv_none, bh_none = model.effective_biases(None)
                torch.testing.assert_close(bv_none, model.visible_bias)
                torch.testing.assert_close(bh_none, model.hidden_bias)
                bv_1d, bh_1d = model.effective_biases(c[0])
                self.assertEqual(bv_1d.shape, (3,))
                self.assertEqual(bh_1d.shape, (2,))

                # Broadcast single condition (1D and (1, C))
                c_single = c[0]
                expected_single = model.energy(v, h, c_single.unsqueeze(0).expand(5, -1))
                torch.testing.assert_close(model.energy(v, h, c_single), expected_single)
                torch.testing.assert_close(
                    model(s_all, c_single.unsqueeze(0)), expected_single
                )

                # Unconditional (c=None or c=0) reduces identically to RBM
                rbm = RestrictedBoltzmannMachine(
                    3,
                    2,
                    quadratic_coef=model.quadratic_coef.detach().clone(),
                    linear_bias=model.linear_bias.detach().clone(),
                    device="cpu",
                )
                torch.testing.assert_close(model(s_all), rbm(s_all))
                torch.testing.assert_close(
                    model(s_all, torch.zeros_like(c)), rbm(s_all)
                )

    def test_free_energy_and_conditionals_match_exact_marginals(self):
        for use_cond_visible in (True, False):
            with self.subTest(use_cond_visible=use_cond_visible):
                model = self._make_model(use_cond_visible=use_cond_visible)
                c = torch.tensor([0.5, -0.3, 1.2, -0.7], dtype=torch.float64)
                all_v = _all_binary_states(3, dtype=torch.float64)
                all_h = _all_binary_states(2, dtype=torch.float64)

                # Joint energy table E(v, h | c) of shape (8, 4)
                v_rep = all_v.repeat_interleave(all_h.shape[0], dim=0)
                h_rep = all_h.repeat(all_v.shape[0], 1)
                joint_e = model.energy(v_rep, h_rep, c).reshape(
                    all_v.shape[0], all_h.shape[0]
                )

                # Exact F(v | c) = -log sum_h exp(-E(v, h | c))
                expected_fe = -torch.logsumexp(-joint_e, dim=1)
                actual_fe = model.free_energy(all_v, c)
                torch.testing.assert_close(actual_fe, expected_fe, rtol=1e-12, atol=1e-12)

                # Exact P(h_j = 1 | v, c)
                expected_h_prob = torch.softmax(-joint_e, dim=1) @ all_h
                s_hid = model.get_hidden(all_v, c, bernoulli=False)
                torch.testing.assert_close(s_hid[:, :3], all_v)
                torch.testing.assert_close(
                    s_hid[:, 3:], expected_h_prob, rtol=1e-12, atol=1e-12
                )

                # Exact P(v_i = 1 | h, c)
                expected_v_prob = torch.softmax(-joint_e.t(), dim=1) @ all_v
                s_vis = model.get_visible(all_h, c, bernoulli=False)
                torch.testing.assert_close(s_vis[:, 3:], all_h)
                torch.testing.assert_close(
                    s_vis[:, :3], expected_v_prob, rtol=1e-12, atol=1e-12
                )

                # Bernoulli outputs are strictly binary {0, 1}
                bin_vals = torch.tensor([0.0, 1.0], dtype=torch.float64)
                s_hid_bin = model.get_hidden(all_v, c, bernoulli=True)
                s_vis_bin = model.get_visible(all_h, c, bernoulli=True)
                self.assertTrue(torch.isin(s_hid_bin[:, 3:], bin_vals).all())
                self.assertTrue(torch.isin(s_vis_bin[:, :3], bin_vals).all())

    def test_gradient_flow_and_objective(self):
        for use_cond_visible in (True, False):
            with self.subTest(use_cond_visible=use_cond_visible):
                model = self._make_model(use_cond_visible=use_cond_visible)
                gen = torch.Generator().manual_seed(11)
                v_pos = torch.randint(0, 2, (4, 3), generator=gen).double()
                v_neg = torch.randint(0, 2, (6, 3), generator=gen).double()
                c_pos = torch.randn(4, 4, generator=gen, dtype=torch.float64, requires_grad=True)
                c_neg = torch.randn(6, 4, generator=gen, dtype=torch.float64)

                s_pos = model.get_hidden(v_pos, c_pos, requires_grad=True)
                self.assertTrue(s_pos.requires_grad)
                s_neg = model.get_hidden(v_neg, c_neg, requires_grad=False)
                self.assertFalse(s_neg.requires_grad)

                loss = model.objective(
                    s_pos, s_neg, condition=c_pos, condition_negative=c_neg
                )
                loss.backward()

                for name, param in model.named_parameters():
                    self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                    self.assertTrue(torch.isfinite(param.grad).all())
                    self.assertGreater(param.grad.abs().sum().item(), 0.0)
                self.assertIsNotNone(c_pos.grad)
                self.assertTrue(torch.isfinite(c_pos.grad).all())

                # Unconditional objective (condition=None)
                uncond_loss = model.objective(s_pos.detach(), s_neg)
                self.assertTrue(torch.isfinite(uncond_loss))

                # Free energy gradient check
                model.zero_grad()
                fe_loss = model.free_energy(v_pos, c_pos.detach()).mean()
                fe_loss.backward()
                for param in model.parameters():
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())

                # get_visible gradient flow
                h_in = torch.rand(3, 2, dtype=torch.float64, requires_grad=True)
                s_vis = model.get_visible(h_in, c_pos[:3].detach(), requires_grad=True)
                self.assertTrue(s_vis.requires_grad)
                s_vis.sum().backward()
                self.assertIsNotNone(h_in.grad)


class TestCRBMIsingEquivalenceAndSampling(unittest.TestCase):
    """Exact PyTorch-vs-Ising energy equivalence and SA sampling/training."""

    def _assert_ising_matches_energy(
        self,
        model: ConditionalRestrictedBoltzmannMachine,
        ising_mat: np.ndarray,
        condition: torch.Tensor | None,
        atol: float,
    ) -> None:
        self.assertEqual(ising_mat.shape, (model.num_nodes + 1, model.num_nodes + 1))
        np.testing.assert_allclose(ising_mat, ising_mat.T, rtol=0.0, atol=atol)
        np.testing.assert_allclose(np.diag(ising_mat), 0.0, rtol=0.0, atol=atol)

        states = _all_binary_states(model.num_nodes, dtype=model.dtype)
        with torch.no_grad():
            pt_energy = model(states, condition=condition).double().numpy()

        logical = 2.0 * states.double().numpy() - 1.0
        j_mat = ising_mat.astype(np.float64)
        for aux in (-1.0, 1.0):
            physical = np.column_stack(
                [logical * aux, np.full(states.shape[0], aux, dtype=np.float64)]
            )
            ising_energy = -np.einsum("bi,ij,bj->b", physical, j_mat, physical)
            diffs = ising_energy - pt_energy
            max_dev = float(np.max(np.abs(diffs - diffs[0])))
            self.assertLess(max_dev, atol)

    def test_ising_energy_equivalence_exhaustive(self):
        for use_cond_visible in (True, False):
            for dtype, atol in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
                with self.subTest(use_cond_visible=use_cond_visible, dtype=dtype):
                    torch.manual_seed(123)
                    model = ConditionalRestrictedBoltzmannMachine(
                        num_visible=3,
                        num_condition=4,
                        num_hidden=3,
                        use_cond_visible=use_cond_visible,
                        dtype=dtype,
                        device="cpu",
                    )
                    with torch.no_grad():
                        for param in model.parameters():
                            param.normal_(0.0, 0.8)

                    # Unconditional (c = None)
                    mat_none = model.condition_ising_matrix(None)
                    self._assert_ising_matches_energy(model, mat_none, None, atol)
                    np.testing.assert_allclose(model.get_ising_matrix(), mat_none, atol=0.0)

                    # Single 1D condition
                    c_single = torch.randn(4, dtype=dtype)
                    mat_single = model.condition_ising_matrix(c_single)
                    self._assert_ising_matches_energy(model, mat_single, c_single, atol)
                    np.testing.assert_allclose(
                        model.get_ising_matrix(c_single), mat_single, atol=0.0
                    )

                    # Batch of conditions
                    c_batch = torch.randn(3, 4, dtype=dtype)
                    mats_batch = model.condition_ising_matrix(c_batch)
                    self.assertEqual(mats_batch.shape, (3, 7, 7))
                    for i in range(3):
                        self._assert_ising_matches_energy(
                            model, mats_batch[i], c_batch[i], atol
                        )

    def test_bfloat16_ising_export(self):
        model = ConditionalRestrictedBoltzmannMachine(
            2, 2, 2, dtype=torch.bfloat16, device="cpu"
        )
        mat = model.condition_ising_matrix(torch.ones(2, dtype=torch.bfloat16))
        self.assertEqual(mat.shape, (5, 5))
        self.assertEqual(mat.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(mat)))

    def test_condition_switches_sampled_distribution(self):
        # Construct a model where condition [1, 0] favors v=[1, 1, 0, 0]
        # and condition [0, 1] favors v=[0, 0, 1, 1].
        w = torch.tensor(
            [[3.0, -3.0], [3.0, -3.0], [-3.0, 3.0], [-3.0, 3.0]],
            dtype=torch.float32,
        )
        b = torch.zeros(6, dtype=torch.float32)
        u = torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32)
        a = torch.tensor(
            [[3.0, 3.0, -3.0, -3.0], [-3.0, -3.0, 3.0, 3.0]],
            dtype=torch.float32,
        )
        model = ConditionalRestrictedBoltzmannMachine(
            num_visible=4,
            num_condition=2,
            num_hidden=2,
            use_cond_visible=True,
            quadratic_coef=w,
            linear_bias=b,
            cond_hidden_coef=u,
            cond_visible_coef=a,
            device="cpu",
        )
        c0 = torch.tensor([1.0, 0.0])
        c1 = torch.tensor([0.0, 1.0])
        target0 = torch.tensor([1.0, 1.0, 0.0, 0.0])
        target1 = torch.tensor([0.0, 0.0, 1.0, 1.0])

        # Gibbs sampling reflects condition switch
        torch.manual_seed(0)
        g0 = model.gibbs_sample(condition=c0, num_steps=20, num_samples=16)
        g1 = model.gibbs_sample(condition=c1, num_steps=20, num_samples=16)
        self.assertGreater((g0[:, :4] == target0).all(dim=-1).float().mean().item(), 0.9)
        self.assertGreater((g1[:, :4] == target1).all(dim=-1).float().mean().item(), 0.9)

        # Ising SA sampling reflects condition switch
        sampler = LocalSimulatedAnnealingSampler(num_reads=12, num_sweeps=40, seed=7)
        s0 = model.sample(sampler, condition=c0)
        s1 = model.sample(sampler, condition=c1)
        self.assertGreater((s0[:, :4] == target0).all(dim=-1).float().mean().item(), 0.9)
        self.assertGreater((s1[:, :4] == target1).all(dim=-1).float().mean().item(), 0.9)

        # Single condition with return_conditions=True
        s0_pair, c0_matched = model.condition_sample(sampler, c0, return_conditions=True)
        self.assertEqual(s0_pair.shape, (12, model.num_nodes))
        torch.testing.assert_close(c0_matched, c0.unsqueeze(0).expand(12, -1))

        # Batch condition_sample with return_conditions=True and False
        c_both = torch.stack([c0, c1])
        samples, matched_c = model.condition_sample(
            sampler, c_both, return_conditions=True
        )
        self.assertEqual(samples.shape, (24, model.num_nodes))
        self.assertEqual(matched_c.shape, (24, model.num_condition))
        samples_only = model.condition_sample(sampler, c_both, return_conditions=False)
        self.assertEqual(samples_only.shape, (24, model.num_nodes))

        # Unconditional sample also works
        uncond = model.sample(sampler)
        self.assertEqual(uncond.shape, (12, model.num_nodes))

    def test_kaiwu_sa_optimizer_contract(self):
        """Verify Kaiwu SimulatedAnnealingOptimizer receives exact Ising matrices."""
        model = ConditionalRestrictedBoltzmannMachine(3, 2, 2, device="cpu")
        sa_opt = SimulatedAnnealingOptimizer(
            initial_temperature=10.0,
            alpha=0.95,
            cutoff_temperature=0.1,
            iterations_per_t=10,
            size_limit=4,
        )
        local_sa = LocalSimulatedAnnealingSampler(num_reads=4, num_sweeps=20, seed=1)
        seen_matrices = []

        def fake_solve(ising_mat):
            seen_matrices.append(ising_mat.copy())
            return local_sa.solve(ising_mat)

        conds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        with patch.object(sa_opt, "solve", side_effect=fake_solve):
            samples, matched = model.condition_sample(
                sa_opt, conds, return_conditions=True
            )
            uncond = model.sample(sa_opt)

        self.assertEqual(len(seen_matrices), 3)
        np.testing.assert_allclose(seen_matrices[0], model.condition_ising_matrix(conds[0]))
        np.testing.assert_allclose(seen_matrices[1], model.condition_ising_matrix(conds[1]))
        np.testing.assert_allclose(seen_matrices[2], model.get_ising_matrix())
        self.assertEqual(samples.shape, (8, 5))
        self.assertEqual(matched.shape, (8, 2))
        self.assertEqual(uncond.shape, (4, 5))

    def test_end_to_end_sa_conditional_learning_convergence(self):
        torch.manual_seed(2026)
        np.random.seed(2026)

        prototypes = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], dtype=torch.float32
        )
        conds = torch.eye(2, dtype=torch.float32)

        model = ConditionalRestrictedBoltzmannMachine(
            num_visible=4,
            num_condition=2,
            num_hidden=4,
            use_cond_visible=True,
            h_range=(-4.0, 4.0),
            j_range=(-4.0, 4.0),
            device="cpu",
        )
        sampler = LocalSimulatedAnnealingSampler(
            num_reads=8, num_sweeps=30, t_start=5.0, t_end=1.0, seed=2026
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)

        def free_energy_margin() -> float:
            with torch.no_grad():
                fe_true = model.free_energy(prototypes, conds).mean()
                fe_wrong = model.free_energy(prototypes.flip(0), conds).mean()
                return float((fe_wrong - fe_true).item())

        margin_before = free_energy_margin()
        for _ in range(30):
            s_pos = model.get_hidden(prototypes, conds, bernoulli=False)
            s_neg, c_neg = model.condition_sample(
                sampler, conds, return_conditions=True
            )
            optimizer.zero_grad()
            loss = model.objective(
                s_pos, s_neg, condition=conds, condition_negative=c_neg
            )
            loss.backward()
            optimizer.step()
            model.clip_parameters()

        margin_after = free_energy_margin()
        self.assertGreater(margin_after, margin_before + 2.0)

        # Conditional generation accuracy after training
        eval_sampler = LocalSimulatedAnnealingSampler(
            num_reads=10, num_sweeps=40, t_start=5.0, t_end=0.05, seed=99
        )
        correct = 0
        total = 0
        for idx in range(2):
            gen_samples = model.condition_sample(eval_sampler, conds[idx])
            matches = (gen_samples[:, :4] == prototypes[idx]).all(dim=-1)
            correct += int(matches.sum().item())
            total += int(matches.numel())
        self.assertGreaterEqual(correct / total, 0.9)


class TestCRBMClipSerializationAndDevice(unittest.TestCase):
    """Parameter clipping, state_dict round-trip, Gibbs variants, and device sync."""

    def test_clip_parameters_and_defaults(self):
        model = ConditionalRestrictedBoltzmannMachine(
            num_visible=3,
            num_condition=2,
            num_hidden=2,
            quadratic_coef=torch.full((3, 2), 5.0),
            linear_bias=torch.full((5,), -5.0),
            cond_hidden_coef=torch.full((2, 2), 4.0),
            cond_visible_coef=torch.full((2, 3), -4.0),
            h_range=(-1.0, 1.0),
            j_range=(-0.5, 0.5),
            device="cpu",
        )
        self.assertLessEqual(model.linear_bias.abs().max().item(), 1.0)
        self.assertLessEqual(model.quadratic_coef.abs().max().item(), 0.5)
        self.assertLessEqual(model.cond_hidden_coef.abs().max().item(), 0.5)
        self.assertLessEqual(model.cond_visible_coef.abs().max().item(), 0.5)

        with torch.no_grad():
            model.linear_bias.fill_(3.0)
            model.quadratic_coef.fill_(-3.0)
            model.cond_hidden_coef.fill_(3.0)
            model.cond_visible_coef.fill_(-3.0)
        model.clip_parameters(h_range=(-2.0, 2.0), j_range=(-1.5, 1.5), c_range=(-0.25, 0.25))
        self.assertAlmostEqual(model.linear_bias.max().item(), 2.0)
        self.assertAlmostEqual(model.quadratic_coef.min().item(), -1.5)
        self.assertAlmostEqual(model.cond_hidden_coef.max().item(), 0.25)
        self.assertAlmostEqual(model.cond_visible_coef.min().item(), -0.25)

        # Clip when use_cond_visible=False
        no_vis = ConditionalRestrictedBoltzmannMachine(
            3, 2, 2, use_cond_visible=False, h_range=(-0.5, 0.5), j_range=(-0.5, 0.5), device="cpu"
        )
        no_vis.clip_parameters()
        self.assertLessEqual(no_vis.cond_hidden_coef.abs().max().item(), 0.5)

    def test_state_dict_serialization_and_device_sync(self):
        for use_cond_visible in (True, False):
            with self.subTest(use_cond_visible=use_cond_visible):
                src = ConditionalRestrictedBoltzmannMachine(
                    3, 2, 2, use_cond_visible=use_cond_visible, device="cpu"
                )
                with torch.no_grad():
                    for param in src.parameters():
                        param.normal_(0.0, 0.5)

                dst = ConditionalRestrictedBoltzmannMachine(
                    3, 2, 2, use_cond_visible=use_cond_visible, device="cpu"
                )
                dst.load_state_dict(src.state_dict(), strict=True)
                v = torch.randint(0, 2, (4, 3)).float()
                c = torch.randn(4, 2)
                torch.testing.assert_close(src.free_energy(v, c), dst.free_energy(v, c))

                # Dtype move updates property and keeps device intact
                dst.to(dtype=torch.float64)
                self.assertEqual(dst.dtype, torch.float64)
                self.assertEqual(dst.device, torch.device("cpu"))

                # _apply device sync (nested module move contract)
                dst._apply(lambda t: t.to(torch.device("meta")))
                self.assertEqual(dst.device, torch.device("meta"))
                self.assertEqual(dst.quadratic_coef.device, torch.device("meta"))

    def test_gibbs_sample_variants(self):
        model = ConditionalRestrictedBoltzmannMachine(3, 2, 2, device="cpu")
        c_batch = torch.randn(4, 2)
        v_init = torch.randint(0, 2, (4, 3)).float()

        out_init = model.gibbs_sample(
            condition=c_batch, num_steps=2, s_visible=v_init, return_prob=True
        )
        self.assertEqual(out_init.shape, (4, 5))
        self.assertTrue(torch.all((out_init >= 0.0) & (out_init <= 1.0)))

        out_uncond = model.gibbs_sample(num_steps=1, num_samples=5)
        self.assertEqual(out_uncond.shape, (5, 5))

        out_broadcast = model.gibbs_sample(
            condition=torch.randn(1, 2), num_steps=1, num_samples=3
        )
        self.assertEqual(out_broadcast.shape, (3, 5))

        out_1d = model.gibbs_sample(condition=torch.randn(2), num_steps=1)
        self.assertEqual(out_1d.shape, (1, 5))

        out_2d = model.gibbs_sample(condition=c_batch, num_steps=1)
        self.assertEqual(out_2d.shape, (4, 5))


if __name__ == "__main__":
    unittest.main()
