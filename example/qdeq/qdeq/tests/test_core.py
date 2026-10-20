import unittest

import torch
import torch.nn as nn

from qdeq.core import QDEQ
from qdeq.ising_sa import IsingSolverConfig as QDeqIsingSolverConfig
from qdeq.solvers import anderson_solver, ising_sa_solver


class LinearFixedPoint(nn.Module):
    """Toy fθ(z, x) = z @ Wᵀ * 0.3 + x, used across qdeq's tests."""

    def __init__(self, dim: int, contraction: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim) * contraction)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z, x):
        return torch.einsum("ij,bjl->bil", self.weight, z) + self.bias.view(1, -1, 1) + x


class TestQDEQEval(unittest.TestCase):
    def test_eval_mode_returns_raw_solver_result(self):
        torch.manual_seed(0)
        bsz, d, L = 2, 4, 1
        func = LinearFixedPoint(d)
        func.eval()
        x = torch.randn(bsz, d, L)
        z_init = torch.zeros(bsz, d, L)

        deq = QDEQ(f_solver=anderson_solver, f_thres=40, stop_mode="rel")
        deq.eval()

        z_out, info = deq(func, z_init, x)

        with torch.no_grad():
            expected = anderson_solver(lambda z: func(z, x), z_init, threshold=40, stop_mode="rel")

        torch.testing.assert_close(z_out, expected["result"])
        self.assertEqual(info["nstep"], expected["nstep"])
        self.assertFalse(z_out.requires_grad)


class TestQDEQTrainGrad(unittest.TestCase):
    def _bptt_reference_grads(self, dim, contraction, x, n_unroll=300):
        func = LinearFixedPoint(dim, contraction)
        z = torch.zeros_like(x)
        for _ in range(n_unroll):
            z = func(z, x)
        loss = (z ** 2).sum()
        grads = torch.autograd.grad(loss, [func.weight, func.bias, x])
        return z.detach(), grads

    def test_training_mode_gradients_match_bptt_reference(self):
        torch.manual_seed(0)
        bsz, d, L = 2, 4, 1
        contraction = 0.3
        x_ref = torch.randn(bsz, d, L, requires_grad=True)
        z_star_ref, (grad_w_ref, grad_b_ref, grad_x_ref) = self._bptt_reference_grads(
            d, contraction, x_ref
        )

        func = LinearFixedPoint(d, contraction)
        x = x_ref.detach().clone().requires_grad_(True)
        z_init = torch.zeros(bsz, d, L)

        deq = QDEQ(f_solver=anderson_solver, f_thres=40, b_thres=40, stop_mode="rel")
        deq.train()

        z_out, info = deq(func, z_init, x)
        self.assertTrue(z_out.requires_grad)
        torch.testing.assert_close(z_out.detach(), z_star_ref, atol=1e-3, rtol=1e-3)

        loss = (z_out ** 2).sum()
        loss.backward()

        torch.testing.assert_close(func.weight.grad, grad_w_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(func.bias.grad, grad_b_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(x.grad, grad_x_ref, atol=1e-3, rtol=1e-3)

        self.assertIn("backward_nstep", info)


class TestQDEQDefaultSolver(unittest.TestCase):
    def test_no_f_solver_defaults_to_ising(self):
        torch.manual_seed(0)
        bsz, d, L = 1, 2, 1
        func = LinearFixedPoint(d)
        func.eval()
        x = torch.randn(bsz, d, L)
        z_init = torch.zeros(bsz, d, L)

        deq = QDEQ()  # no f_solver passed at all
        deq.eval()
        z_out, info = deq(func, z_init, x)

        # Only the vendored Ising solver's result dict carries this key;
        # anderson_solver's does not. This confirms QDEQ picked Ising by
        # default rather than requiring the caller to pass f_solver.
        self.assertEqual(info.get("solver"), "ising")
        self.assertTrue(torch.isfinite(z_out).all())


class QuantAwareFixedPoint(nn.Module):
    """Toy fθ that tracks set_quant(...) calls instead of actually quantizing."""

    def __init__(self, dim: int, contraction: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim) * contraction)
        self.quant_log: list[bool] = []

    def set_quant(self, enabled: bool) -> None:
        self.quant_log.append(enabled)

    def forward(self, z, x):
        return torch.einsum("ij,bjl->bil", self.weight, z) + x


class TestQDEQQuantOrchestration(unittest.TestCase):
    def test_training_mode_toggles_quant_off_then_on(self):
        torch.manual_seed(0)
        bsz, d, L = 1, 2, 1
        func = QuantAwareFixedPoint(d)
        x = torch.randn(bsz, d, L, requires_grad=True)
        z_init = torch.zeros(bsz, d, L)

        deq = QDEQ(f_solver=anderson_solver, f_thres=20, b_thres=20)
        deq.train()
        z_out, _info = deq(func, z_init, x)
        (z_out ** 2).sum().backward()

        self.assertEqual(func.quant_log, [False, True])

    def test_eval_mode_never_touches_quant(self):
        torch.manual_seed(0)
        bsz, d, L = 1, 2, 1
        func = QuantAwareFixedPoint(d)
        x = torch.randn(bsz, d, L)
        z_init = torch.zeros(bsz, d, L)

        deq = QDEQ(f_solver=anderson_solver, f_thres=20)
        deq.eval()
        deq(func, z_init, x)

        self.assertEqual(func.quant_log, [])

    def test_func_without_set_quant_is_unaffected(self):
        # LinearFixedPoint has no set_quant method -- QDEQ must not require one.
        torch.manual_seed(0)
        bsz, d, L = 1, 2, 1
        func = LinearFixedPoint(d)
        x = torch.randn(bsz, d, L, requires_grad=True)
        z_init = torch.zeros(bsz, d, L)

        deq = QDEQ(f_solver=anderson_solver, f_thres=20, b_thres=20)
        deq.train()
        z_out, _info = deq(func, z_init, x)
        (z_out ** 2).sum().backward()  # must not raise

    def test_bound_method_func_still_gets_quant_orchestration(self):
        # hasattr(bound_method, "set_quant") is False even when the method's
        # owning instance has set_quant -- QDEQ must resolve through
        # func.__self__ so passing an nn.Module's bound forward method (a
        # calling style QDEQ's own docstring documents as supported) doesn't
        # silently skip quantization toggling.
        torch.manual_seed(0)
        bsz, d, L = 1, 2, 1
        func = QuantAwareFixedPoint(d)
        bound_forward = func.forward
        self.assertFalse(hasattr(bound_forward, "set_quant"))  # sanity check

        x = torch.randn(bsz, d, L, requires_grad=True)
        z_init = torch.zeros(bsz, d, L)

        deq = QDEQ(f_solver=anderson_solver, f_thres=20, b_thres=20)
        deq.train()
        z_out, _info = deq(bound_forward, z_init, x)
        (z_out ** 2).sum().backward()

        self.assertEqual(func.quant_log, [False, True])

    def test_quant_restored_on_forward_solver_exception(self):
        # If f_solver raises mid-solve, set_quant(True) must still run before
        # the exception propagates, so a failed training step doesn't leave
        # quantization stuck disabled for whatever call comes next.
        torch.manual_seed(0)
        bsz, d, L = 1, 2, 1
        func = QuantAwareFixedPoint(d)
        x = torch.randn(bsz, d, L)
        z_init = torch.zeros(bsz, d, L)

        def failing_solver(f, z0, **kwargs):
            raise RuntimeError("boom")

        deq = QDEQ(f_solver=failing_solver)
        deq.train()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            deq(func, z_init, x)

        self.assertEqual(func.quant_log, [False, True])


class TestQDEQBareDefaultGradients(unittest.TestCase):
    def _bptt_reference(self, dim, contraction, x_ref, n_unroll=300):
        ref_func = LinearFixedPoint(dim, contraction)
        z = torch.zeros_like(x_ref)
        for _ in range(n_unroll):
            z = ref_func(z, x_ref)
        (z ** 2).sum().backward()
        return ref_func.weight.grad, x_ref.grad

    def test_bare_default_ising_forward_anderson_backward_matches_bptt_reference(self):
        # QDEQ()'s zero-argument default is f_solver=ising_sa_solver(...,
        # solver_variant="v5_dynamic"), b_solver=anderson_solver -- matching
        # this project's actual NGPTQuantizedIsingDecoder (Ising forward,
        # Anderson backward), verified against its source. Anderson backward
        # is deterministic and already shown accurate on this toy problem
        # (TestQDEQTrainGrad, atol=rtol=1e-3), so only the Ising forward
        # solve is stochastic here -- tolerance is looser than
        # TestQDEQTrainGrad's but much tighter than an Ising-backward test.
        torch.manual_seed(3)
        bsz, d, L = 1, 2, 1
        contraction = 0.3

        x_ref = torch.randn(bsz, d, L, requires_grad=True)
        grad_w_ref, grad_x_ref = self._bptt_reference(d, contraction, x_ref)

        func = LinearFixedPoint(d, contraction)
        x = x_ref.detach().clone().requires_grad_(True)
        z_init = torch.zeros(bsz, d, L)

        deq = QDEQ()  # bare default: Ising v5_dynamic forward, anderson_solver backward
        deq.train()
        z_out, info = deq(func, z_init, x)
        self.assertTrue(torch.isfinite(z_out).all())

        loss = (z_out ** 2).sum()
        loss.backward()

        self.assertIn("backward_nstep", info)
        torch.testing.assert_close(func.weight.grad, grad_w_ref, atol=0.05, rtol=0.05)
        torch.testing.assert_close(x.grad, grad_x_ref, atol=0.05, rtol=0.05)

    def test_explicit_ising_as_both_solvers_produces_finite_reasonable_gradients(self):
        # Ising as BOTH f_solver and b_solver (opting out of the anderson
        # backward default) is not QDEQ's default, but must still work when
        # explicitly requested -- SA is stochastic and coefficient-quantized,
        # so this only checks finiteness and the right neighborhood, not
        # tight agreement.
        torch.manual_seed(3)
        bsz, d, L = 1, 2, 1
        contraction = 0.3

        x_ref = torch.randn(bsz, d, L, requires_grad=True)
        grad_w_ref, grad_x_ref = self._bptt_reference(d, contraction, x_ref)

        func = LinearFixedPoint(d, contraction)
        x = x_ref.detach().clone().requires_grad_(True)
        z_init = torch.zeros(bsz, d, L)

        ising = ising_sa_solver(QDeqIsingSolverConfig(solver_variant="v5_dynamic"))
        deq = QDEQ(f_solver=ising, b_solver=ising)
        deq.train()
        z_out, info = deq(func, z_init, x)
        self.assertTrue(torch.isfinite(z_out).all())

        loss = (z_out ** 2).sum()
        loss.backward()

        self.assertTrue(torch.isfinite(func.weight.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIn("backward_nstep", info)
        torch.testing.assert_close(func.weight.grad, grad_w_ref, atol=0.3, rtol=0.3)
        torch.testing.assert_close(x.grad, grad_x_ref, atol=0.3, rtol=0.3)


if __name__ == "__main__":
    unittest.main()
