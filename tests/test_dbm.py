"""Unit tests for DeepBoltzmannMachine (DBM)."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import torch
from kaiwu.classical import SimulatedAnnealingOptimizer

import kaiwu.torch_plugin as plugin
from kaiwu.torch_plugin import DeepBoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.abstract_boltzmann_machine import AbstractBoltzmannMachine


def _all_binary_states(num_bits: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Enumerate all 2**num_bits binary vectors."""
    return torch.tensor(list(itertools.product((0.0, 1.0), repeat=num_bits)), dtype=dtype)


class LocalSimulatedAnnealingSampler:
    """Deterministic license-free Ising SA sampler maximizing ``s^T J s``."""

    def __init__(
        self,
        num_reads: int = 8,
        num_sweeps: int = 40,
        t_start: float = 5.0,
        t_end: float = 1.0,
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
                    delta_e = 4.0 * spins[idx] * field[idx]
                    if delta_e <= 0.0 or self.rng.random() < np.exp(-delta_e / temp):
                        spins[idx] = -spins[idx]
                        field += 2.0 * spins[idx] * coupling[:, idx]
            rows.append(spins.copy())
        return np.stack(rows, axis=0)


class _BadSampler:
    """Sampler returning malformed or non-finite spin outputs."""

    def __init__(self, output):
        self.output = output

    def solve(self, _ising_mat):
        return self.output


class TestDBMConstructionAndValidation(unittest.TestCase):
    """Exports, inheritance, parameter validation, and runtime contracts."""

    def test_exported_and_attributes(self):
        self.assertIn("DeepBoltzmannMachine", plugin.__all__)
        model = DeepBoltzmannMachine(4, (3, 2), device="cpu")
        self.assertIsInstance(model, AbstractBoltzmannMachine)
        self.assertEqual(model.num_visible, 4)
        self.assertEqual(model.hidden_dims, (3, 2))
        self.assertEqual(model.layer_dims, (4, 3, 2))
        self.assertEqual(model.num_hidden_layers, 2)
        self.assertEqual(model.num_hidden, 5)
        self.assertEqual(model.num_nodes, 9)
        self.assertEqual(model.dtype, torch.float32)
        self.assertEqual(model.device, torch.device("cpu"))
        self.assertEqual(len(model.hidden_biases), 2)
        self.assertEqual(model.visible_bias.shape, (4,))

    def test_invalid_constructor_arguments(self):
        for bad_int in (0, -3, True, 1.5, "4"):
            with self.assertRaises(ValueError):
                DeepBoltzmannMachine(bad_int, (3, 2), device="cpu")
            with self.assertRaises(ValueError):
                DeepBoltzmannMachine(4, (3, bad_int), device="cpu")

        for bad_seq in ("32", b"32", 4, None):
            with self.assertRaises(TypeError):
                DeepBoltzmannMachine(4, bad_seq, device="cpu")
        with self.assertRaises(ValueError):
            DeepBoltzmannMachine(4, (), device="cpu")
        with self.assertRaises(ValueError):
            DeepBoltzmannMachine(4, (3, 2), dtype=torch.int32, device="cpu")
        with self.assertRaises(ValueError):
            DeepBoltzmannMachine(4, (3, 2), h_range=(2.0, -2.0), device="cpu")
        with self.assertRaises(ValueError):
            DeepBoltzmannMachine(4, (3, 2), j_range=(0.0, float("nan")), device="cpu")
        with self.assertRaises(ValueError):
            DeepBoltzmannMachine(4, (3, 2), h_range=(False, 1.0), device="cpu")
        with self.assertRaises(ValueError):
            DeepBoltzmannMachine(4, (3, 2), h_range=(1.0,), device="cpu")
        with self.assertRaises(TypeError):
            DeepBoltzmannMachine(4, (3, 2), weights=torch.zeros(4, 3), device="cpu")
        with self.assertRaises(ValueError):
            DeepBoltzmannMachine(4, (3, 2), weights=[torch.zeros(4, 3)], device="cpu")
        with self.assertRaises(TypeError):
            DeepBoltzmannMachine(4, (3, 2), weights=[torch.zeros(4, 3), "bad"], device="cpu")
        with self.assertRaises(ValueError):
            DeepBoltzmannMachine(
                4, (3, 2), weights=[torch.zeros(4, 3), torch.zeros(3, 5)], device="cpu"
            )
        with self.assertRaises(ValueError):
            DeepBoltzmannMachine(
                4, (3, 2), biases=[torch.zeros(4), torch.zeros(3)], device="cpu"
            )

    def test_invalid_runtime_inputs(self):
        model = DeepBoltzmannMachine(3, (2, 2), device="cpu")
        with self.assertRaises(TypeError):
            model.energy("bad")
        with self.assertRaises(ValueError):
            model.energy(torch.zeros(2, 5))
        with self.assertRaises(TypeError):
            model.split_states([[0.0] * 7])
        with self.assertRaises(ValueError):
            model.energy([torch.zeros(2, 3), torch.zeros(2, 2)])
        with self.assertRaises(ValueError):
            model.energy([torch.zeros(2, 3), torch.zeros(3, 2), torch.zeros(2, 2)])
        with self.assertRaises(ValueError):
            model.mean_field_inference(torch.zeros(2, 3), num_steps=0)
        with self.assertRaises(ValueError):
            model.mean_field_inference(torch.zeros(2, 3), tol=-1e-3)
        with self.assertRaises(ValueError):
            model.mean_field_inference(torch.zeros(2, 3), tol=True)
        with self.assertRaises(TypeError):
            model.mean_field_inference(torch.zeros(2, 3), init_means=torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            model.mean_field_inference(torch.zeros(2, 3), init_means=[torch.zeros(2, 2)])
        with self.assertRaises(ValueError):
            model.mean_field_inference(
                torch.zeros(2, 3), init_means=[torch.zeros(3, 2), torch.zeros(2, 2)]
            )
        with self.assertRaises(ValueError):
            model.gibbs_sample(s_all=torch.zeros(2, 7), s_visible=torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            model.gibbs_sample(s_all=torch.zeros(2, 7), num_samples=3)
        with self.assertRaises(ValueError):
            model.gibbs_sample(s_visible=torch.zeros(2, 3), num_samples=3)
        with self.assertRaises(ValueError):
            model.gibbs_sample()
        with self.assertRaises(TypeError):
            model.clamped_ising_matrix([1.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            model.clamped_ising_matrix(torch.zeros(2, 3))
        with self.assertRaises(TypeError):
            model.clamped_sample(_BadSampler(np.ones((1, 5))), [1.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            model.clamped_sample(_BadSampler(np.ones((1, 5))), torch.zeros(0, 3))
        with self.assertRaises(RuntimeError):
            model.sample(_BadSampler(np.ones((2, 4))))
        with self.assertRaises(RuntimeError):
            model.sample(_BadSampler(np.full((2, 8), np.nan)))


class TestDBMEnergyMeanFieldAndGradients(unittest.TestCase):
    """Exact energy, RBM reduction, mean-field monotonicity, and gradcheck."""

    def _make_model(
        self,
        num_visible: int,
        hidden_dims: tuple[int, ...],
        dtype: torch.dtype = torch.float64,
        seed: int = 42,
    ) -> DeepBoltzmannMachine:
        gen = torch.Generator().manual_seed(seed)
        dims = (num_visible, *hidden_dims)
        weights = [
            torch.randn(dims[i], dims[i + 1], generator=gen, dtype=dtype) * 0.7
            for i in range(len(hidden_dims))
        ]
        biases = [torch.randn(d, generator=gen, dtype=dtype) * 0.5 for d in dims]
        return DeepBoltzmannMachine(
            num_visible=num_visible,
            hidden_dims=hidden_dims,
            weights=weights,
            biases=biases,
            dtype=dtype,
            device="cpu",
        )

    def test_energy_2layer_3layer_and_single_layer_rbm_equivalence(self):
        # Single-layer DBM reduces identically to RBM
        dbm1 = self._make_model(3, (2,), dtype=torch.float64, seed=1)
        rbm = RestrictedBoltzmannMachine(
            3,
            2,
            quadratic_coef=dbm1.weights[0].detach().clone(),
            linear_bias=torch.cat([dbm1.biases[0], dbm1.biases[1]]).detach().clone(),
            device="cpu",
        )
        s1 = _all_binary_states(5, dtype=torch.float64)
        torch.testing.assert_close(dbm1(s1), rbm(s1), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dbm1.get_ising_matrix(), rbm.get_ising_matrix(), atol=1e-12)

        # 2-layer and 3-layer analytic energy match
        for hidden_dims in ((3, 2), (2, 2, 2)):
            with self.subTest(hidden_dims=hidden_dims):
                model = self._make_model(3, hidden_dims, dtype=torch.float64, seed=9)
                gen = torch.Generator().manual_seed(17)
                layer_states = [
                    torch.randint(0, 2, (5, d), generator=gen).double()
                    for d in model.layer_dims
                ]
                expected = -layer_states[0] @ model.biases[0]
                for idx in range(model.num_hidden_layers):
                    expected = (
                        expected
                        - layer_states[idx + 1] @ model.biases[idx + 1]
                        - torch.sum(
                            (layer_states[idx] @ model.weights[idx])
                            * layer_states[idx + 1],
                            dim=-1,
                        )
                    )
                s_all = torch.cat(layer_states, dim=-1)
                torch.testing.assert_close(model.energy(layer_states), expected)
                torch.testing.assert_close(model(s_all), expected)
                split = model.split_states(s_all)
                for a, b in zip(split, layer_states):
                    torch.testing.assert_close(a, b)

    def test_mean_field_variational_free_energy_monotone_and_upper_bound(self):
        for hidden_dims in ((3,), (3, 2), (2, 2, 2)):
            with self.subTest(hidden_dims=hidden_dims):
                model = self._make_model(3, hidden_dims, dtype=torch.float64, seed=21)
                all_v = _all_binary_states(3, dtype=torch.float64)
                all_h = _all_binary_states(model.num_hidden, dtype=torch.float64)

                # Exact visible free energy F(v) = -log sum_h exp(-E(v, h))
                v_rep = all_v.repeat_interleave(all_h.shape[0], dim=0)
                h_rep = all_h.repeat(all_v.shape[0], 1)
                joint_e = model(torch.cat([v_rep, h_rep], dim=-1)).reshape(
                    all_v.shape[0], all_h.shape[0]
                )
                exact_fe = -torch.logsumexp(-joint_e, dim=1)

                # Track variational free energy across sweeps: must be monotonically non-increasing
                mus = [
                    torch.full((all_v.shape[0], d), 0.5, dtype=torch.float64)
                    for d in model.hidden_dims
                ]
                prev_vfe = model.variational_free_energy(all_v, hidden_means=mus)
                for _ in range(12):
                    mus = list(
                        model.mean_field_inference(
                            all_v, num_steps=1, tol=0.0, init_means=mus, return_layers=True
                        )
                    )
                    cur_vfe = model.variational_free_energy(all_v, hidden_means=mus)
                    self.assertTrue(torch.all(cur_vfe <= prev_vfe + 1e-12))
                    # Must always upper-bound exact free energy
                    self.assertTrue(torch.all(cur_vfe >= exact_fe - 1e-12))
                    prev_vfe = cur_vfe

                # Default variational_free_energy(v) matches converged bound
                auto_vfe = model.variational_free_energy(all_v, num_steps=50, tol=1e-10)
                self.assertTrue(torch.all(auto_vfe >= exact_fe - 1e-12))
                if len(hidden_dims) == 1:
                    # Exact for 1-hidden-layer RBM
                    torch.testing.assert_close(auto_vfe, exact_fe, rtol=1e-10, atol=1e-10)

    def test_gradcheck_and_gradient_flow(self):
        model = self._make_model(3, (2, 2), dtype=torch.float64, seed=5)
        v_pos = torch.rand(4, 3, dtype=torch.float64, requires_grad=True)
        s_neg = torch.rand(4, 7, dtype=torch.float64)

        # Autograd gradcheck on forward(s_all), energy([v, h1, h2]), and mean-field objective
        s_in = torch.rand(3, 7, dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(lambda x: model(x).sum(), (s_in,), eps=1e-6, atol=1e-5)
        )
        v_in = torch.rand(3, 3, dtype=torch.float64, requires_grad=True)
        h1_in = torch.rand(3, 2, dtype=torch.float64, requires_grad=True)
        h2_in = torch.rand(3, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b, c: model.energy([a, b, c]).sum(),
                (v_in, h1_in, h2_in),
                eps=1e-6,
                atol=1e-5,
            )
        )

        def fn_mf_objective(v_arg):
            s_pos = model.mean_field_inference(v_arg, num_steps=4, tol=0.0, requires_grad=True)
            return model.objective(s_pos, s_neg)

        self.assertTrue(torch.autograd.gradcheck(fn_mf_objective, (v_pos,), eps=1e-6, atol=1e-5))

        # Parameter gradients through objective verified against central finite differences
        model.zero_grad()
        s_pos = model.mean_field_inference(v_pos.detach(), num_steps=10, tol=1e-8)
        loss = model.objective(s_pos, s_neg)
        loss.backward()
        eps = 1e-6
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0.0)
            analytic = param.grad.detach().clone()
            numerical = torch.zeros_like(analytic)
            with torch.no_grad():
                flat_p = param.view(-1)
                flat_num = numerical.view(-1)
                for idx in range(flat_p.numel()):
                    orig = flat_p[idx].item()
                    flat_p[idx] = orig + eps
                    l_plus = model.objective(s_pos, s_neg).item()
                    flat_p[idx] = orig - eps
                    l_minus = model.objective(s_pos, s_neg).item()
                    flat_p[idx] = orig
                    flat_num[idx] = (l_plus - l_minus) / (2.0 * eps)
            torch.testing.assert_close(analytic, numerical, rtol=1e-5, atol=1e-6)

    def test_variational_free_energy_saturated_backward_finite(self):
        """Saturated mean-field posteriors (mu=0 or mu=1) must keep gradients finite."""
        for dtype in (torch.float32, torch.float64):
            for bias_val in (20.0, -20.0):
                with self.subTest(dtype=dtype, bias_val=bias_val):
                    model = DeepBoltzmannMachine(3, (3, 2), dtype=dtype, device="cpu")
                    with torch.no_grad():
                        for bias in model.biases:
                            bias.fill_(bias_val)
                    v = torch.randint(0, 2, (4, 3)).to(dtype=dtype).requires_grad_(True)
                    mus = model.mean_field_inference(
                        v, num_steps=5, tol=0.0, requires_grad=True, return_layers=True
                    )
                    vfe = model.variational_free_energy(v, hidden_means=mus)
                    self.assertTrue(torch.isfinite(vfe).all())
                    vfe.sum().backward()
                    self.assertIsNotNone(v.grad)
                    self.assertTrue(torch.isfinite(v.grad).all())
                    for name, param in model.named_parameters():
                        self.assertIsNotNone(param.grad, f"Missing grad for {name}")
                        self.assertTrue(
                            torch.isfinite(param.grad).all(),
                            f"Non-finite grad for {name} (dtype={dtype}, bias={bias_val})",
                        )


class TestDBMIsingEquivalenceAndSampling(unittest.TestCase):
    """Joint multi-layer Ising energy equivalence, clamped Ising, and SA training."""

    def _assert_ising_matches_energy(
        self,
        ising_mat: np.ndarray,
        binary_states: torch.Tensor,
        pt_energy: np.ndarray,
        atol: float,
    ) -> None:
        num_vars = binary_states.shape[1]
        self.assertEqual(ising_mat.shape, (num_vars + 1, num_vars + 1))
        np.testing.assert_allclose(ising_mat, ising_mat.T, rtol=0.0, atol=atol)
        np.testing.assert_allclose(np.diag(ising_mat), 0.0, rtol=0.0, atol=atol)

        logical = 2.0 * binary_states.double().numpy() - 1.0
        j_mat = ising_mat.astype(np.float64)
        for aux in (-1.0, 1.0):
            physical = np.column_stack(
                [logical * aux, np.full(binary_states.shape[0], aux, dtype=np.float64)]
            )
            ising_energy = -np.einsum("bi,ij,bj->b", physical, j_mat, physical)
            diffs = ising_energy - pt_energy
            max_dev = float(np.max(np.abs(diffs - diffs[0])))
            self.assertLess(max_dev, atol)

    def test_joint_and_clamped_ising_equivalence_exhaustive(self):
        for hidden_dims in ((3, 2), (2, 2, 2)):
            for dtype, atol in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
                with self.subTest(hidden_dims=hidden_dims, dtype=dtype):
                    torch.manual_seed(101)
                    model = DeepBoltzmannMachine(
                        num_visible=3, hidden_dims=hidden_dims, dtype=dtype, device="cpu"
                    )
                    with torch.no_grad():
                        for param in model.parameters():
                            param.normal_(0.0, 0.7)

                    # Full joint Ising matrix across all layers (v, h1, ..., hL)
                    ising_mat = model.to_ising_matrix()
                    np.testing.assert_allclose(model.get_ising_matrix(), ising_mat, atol=0.0)
                    all_states = _all_binary_states(model.num_nodes, dtype=dtype)
                    with torch.no_grad():
                        pt_e = model(all_states).double().numpy()
                    self._assert_ising_matches_energy(ising_mat, all_states, pt_e, atol)

                    # Clamped hidden-only Ising matrix given visible v
                    v_fixed = torch.tensor([1.0, 0.0, 1.0], dtype=dtype)
                    clamped_mat = model.clamped_ising_matrix(v_fixed)
                    np.testing.assert_allclose(
                        model.clamped_ising_matrix(v_fixed.unsqueeze(0)), clamped_mat, atol=0.0
                    )
                    all_h = _all_binary_states(model.num_hidden, dtype=dtype)
                    v_rep = v_fixed.unsqueeze(0).expand(all_h.shape[0], -1)
                    with torch.no_grad():
                        pt_clamped = model(torch.cat([v_rep, all_h], dim=-1)).double().numpy()
                    self._assert_ising_matches_energy(clamped_mat, all_h, pt_clamped, atol)

    def test_bfloat16_ising_export(self):
        model = DeepBoltzmannMachine(2, (2, 2), dtype=torch.bfloat16, device="cpu")
        mat = model.to_ising_matrix()
        self.assertEqual(mat.shape, (7, 7))
        self.assertEqual(mat.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(mat)))
        cmat = model.clamped_ising_matrix(torch.ones(2, dtype=torch.bfloat16))
        self.assertEqual(cmat.shape, (5, 5))
        self.assertEqual(cmat.dtype, np.float32)

    def test_kaiwu_sa_optimizer_contract_and_clamped_sample(self):
        model = DeepBoltzmannMachine(3, (2, 2), device="cpu")
        sa_opt = SimulatedAnnealingOptimizer(
            initial_temperature=10.0,
            alpha=0.95,
            cutoff_temperature=0.1,
            iterations_per_t=10,
            size_limit=4,
        )
        local_sa = LocalSimulatedAnnealingSampler(num_reads=4, num_sweeps=20, seed=3)
        seen = []

        def fake_solve(mat):
            seen.append(mat.copy())
            return local_sa.solve(mat)

        v_batch = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        with patch.object(sa_opt, "solve", side_effect=fake_solve):
            joint = model.sample(sa_opt)
            clamped_1d = model.clamped_sample(sa_opt, v_batch[0])
            clamped_2d = model.clamped_sample(sa_opt, v_batch)

        self.assertEqual(len(seen), 4)
        np.testing.assert_allclose(seen[0], model.get_ising_matrix())
        np.testing.assert_allclose(seen[1], model.clamped_ising_matrix(v_batch[0]))
        self.assertEqual(joint.shape, (4, 7))
        self.assertEqual(clamped_1d.shape, (4, 7))
        self.assertEqual(clamped_2d.shape, (8, 7))
        torch.testing.assert_close(clamped_1d[:, :3], v_batch[0].unsqueeze(0).expand(4, -1))

    def test_end_to_end_joint_ising_training_and_denoising_convergence(self):
        torch.manual_seed(2026)
        np.random.seed(2026)

        patterns = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        wrong = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        model = DeepBoltzmannMachine(
            num_visible=6,
            hidden_dims=(6, 4),
            h_range=(-4.0, 4.0),
            j_range=(-4.0, 4.0),
            device="cpu",
        )
        train_sampler = LocalSimulatedAnnealingSampler(
            num_reads=12, num_sweeps=35, t_start=5.0, t_end=1.0, seed=2026
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        def vfe_gap() -> float:
            with torch.no_grad():
                return float(
                    (
                        model.variational_free_energy(wrong).mean()
                        - model.variational_free_energy(patterns).mean()
                    ).item()
                )

        gap_before = vfe_gap()
        for _ in range(50):
            s_pos = model.mean_field_inference(patterns, num_steps=15, tol=1e-5)
            s_neg = model.sample(train_sampler)
            optimizer.zero_grad()
            loss = model.objective(s_pos, s_neg)
            loss.backward()
            optimizer.step()
            model.clip_parameters()

        gap_after = vfe_gap()
        self.assertGreater(gap_after, gap_before + 2.0)

        # Denoising reconstruction of 1-bit corrupted patterns
        corrupted = patterns.clone()
        corrupted[:, 0] = 1.0 - corrupted[:, 0]
        recon_prob = model.reconstruct(corrupted, num_mf_steps=20, bernoulli=False)
        recon_bits = (recon_prob >= 0.5).float()
        self.assertGreaterEqual((recon_bits == patterns).float().mean().item(), 0.9)

        # Also exercise bernoulli=True reconstruction
        recon_bin = model.reconstruct(corrupted, num_mf_steps=10, bernoulli=True)
        self.assertTrue(torch.isin(recon_bin, torch.tensor([0.0, 1.0])).all())


class TestDBMClipSerializationGibbsAndDevice(unittest.TestCase):
    """Parameter clipping, serialization, Gibbs variants, and device sync."""

    def test_clip_parameters_and_defaults(self):
        model = DeepBoltzmannMachine(
            num_visible=3,
            hidden_dims=(3, 2),
            weights=[torch.full((3, 3), 5.0), torch.full((3, 2), -5.0)],
            biases=[torch.full((3,), 4.0), torch.full((3,), -4.0), torch.full((2,), 4.0)],
            h_range=(-1.0, 1.0),
            j_range=(-0.5, 0.5),
            device="cpu",
        )
        for bias in model.biases:
            self.assertLessEqual(bias.abs().max().item(), 1.0)
        for weight in model.weights:
            self.assertLessEqual(weight.abs().max().item(), 0.5)

        with torch.no_grad():
            for bias in model.biases:
                bias.fill_(3.0)
            for weight in model.weights:
                weight.fill_(-3.0)
        model.clip_parameters(h_range=(-2.0, 2.0), j_range=(-1.5, 1.5))
        for bias in model.biases:
            self.assertAlmostEqual(bias.max().item(), 2.0)
        for weight in model.weights:
            self.assertAlmostEqual(weight.min().item(), -1.5)

    def test_state_dict_roundtrip_and_device_sync(self):
        src = DeepBoltzmannMachine(3, (4, 2), device="cpu")
        with torch.no_grad():
            for param in src.parameters():
                param.normal_(0.0, 0.5)

        dst = DeepBoltzmannMachine(3, (4, 2), device="cpu")
        dst.load_state_dict(src.state_dict(), strict=True)
        v = torch.randint(0, 2, (4, 3)).float()
        torch.testing.assert_close(
            src.variational_free_energy(v), dst.variational_free_energy(v)
        )

        dst.to(dtype=torch.float64)
        self.assertEqual(dst.dtype, torch.float64)
        self.assertEqual(dst.device, torch.device("cpu"))

        dst._apply(lambda t: t.to(torch.device("meta")))
        self.assertEqual(dst.device, torch.device("meta"))
        self.assertEqual(dst.weights[0].device, torch.device("meta"))

    def test_gibbs_sample_modes(self):
        model = DeepBoltzmannMachine(3, (3, 2), device="cpu")
        s_init = torch.randint(0, 2, (4, 8)).float()
        v_init = torch.randint(0, 2, (4, 3)).float()

        out_full = model.gibbs_sample(num_steps=2, s_all=s_init, return_prob=False)
        self.assertEqual(out_full.shape, (4, 8))
        self.assertTrue(torch.isin(out_full, torch.tensor([0.0, 1.0])).all())

        out_clamped = model.gibbs_sample(
            num_steps=2, s_visible=v_init, clamp_visible=True, return_prob=True
        )
        self.assertEqual(out_clamped.shape, (4, 8))
        torch.testing.assert_close(out_clamped[:, :3], v_init)
        self.assertTrue(torch.all((out_clamped >= 0.0) & (out_clamped <= 1.0)))

        out_scratch = model.gibbs_sample(num_steps=2, num_samples=5)
        self.assertEqual(out_scratch.shape, (5, 8))


if __name__ == "__main__":
    unittest.main()
