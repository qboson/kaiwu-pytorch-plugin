"""Exact equivalence tests between model energies and Ising matrices.

The Ising conversion is the bridge from the PyTorch models to the Kaiwu
solvers: ``AbstractBoltzmannMachine.sample()`` submits the matrix produced
by ``get_ising_matrix()`` and maps the returned spins back to binary states
with ``x = (s * s_aux + 1) / 2``. A wrong coefficient or sign in that
conversion silently corrupts every sampled state — issue #171 fixed exactly
such a bug in the RBM bias term after it had already shipped.

These tests brute-force enumerate every binary state of small models and
verify the invariant that ties the two energy functions together:

    E_model(x) + s^T M s = const   for every state x,

where ``M`` is the (N+1, N+1) Ising matrix, ``s = [2x - 1, 1]`` embeds the
binary state as spins with the auxiliary spin fixed to +1, and the constant
may depend on the model but not on the state. This is the same relation the
existing ``test_bm.py`` checks for two hand-picked states with exact float
equality; here it is checked for *all* states with a tolerance, in float64
and in the default float32, for the full matrices of the RBM and BM, for
the BM conditional (hidden-side) matrix, and for the Bernoulli side of the
GBRBM after integrating out its Gaussian units analytically.
"""

import itertools
import os
import sys
import unittest

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

import kaiwu

# Extend namespace package __path__ so that local src is preferred
kaiwu.__path__ = list(kaiwu.__path__) + [os.path.join(src_root, "kaiwu")]

for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith(
        "kaiwu.torch_plugin."
    ):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def enumerate_binary_states(num_nodes, dtype=torch.float64):
    """Return every binary state of ``num_nodes`` variables as a tensor.

    Args:
        num_nodes (int): Number of binary variables.
        dtype (torch.dtype, optional): Tensor dtype of the result.

    Returns:
        torch.Tensor: Tensor of shape (2**num_nodes, num_nodes).
    """
    rows = list(itertools.product([0.0, 1.0], repeat=num_nodes))
    return torch.tensor(rows, dtype=dtype)


def spins_with_auxiliary(states):
    """Map binary states to spin vectors with the auxiliary spin fixed to +1.

    Args:
        states (torch.Tensor): Binary states of shape (B, N).

    Returns:
        torch.Tensor: Spin states of shape (B, N + 1), last column all ones.
    """
    spins = 2.0 * states - 1.0
    return torch.cat([spins, torch.ones(len(spins), 1)], dim=1)


def ising_quadratic_form(ising_matrix, states):
    """Evaluate ``s^T M s`` for the spin embedding of binary states.

    Args:
        ising_matrix (torch.Tensor): Ising matrix of shape (N + 1, N + 1).
        states (torch.Tensor): Binary states of shape (B, N).

    Returns:
        torch.Tensor: Quadratic-form values of shape (B,).
    """
    spins = spins_with_auxiliary(states)
    return torch.einsum("bi,ij,bj->b", spins, ising_matrix, spins)


def assert_energy_ising_equivalence(testcase, model, states, tolerance):
    """Assert ``E_model + s^T M s`` is constant over the given states.

    Args:
        testcase (unittest.TestCase): Test case used for assertions.
        model: Model exposing ``forward`` and ``get_ising_matrix``.
        states (torch.Tensor): Binary states of shape (B, N) matching the
            model dtype.
        tolerance (float): Maximum allowed spread of the invariant.
    """
    energies = model(states)
    ising_matrix = torch.tensor(
        model.get_ising_matrix(), dtype=states.dtype
    )
    total = energies + ising_quadratic_form(ising_matrix, states)
    spread = (total.max() - total.min()).item()
    testcase.assertLessEqual(
        spread,
        tolerance,
        msg=(
            "E_model + s^T M s must be constant over all states; "
            f"observed spread {spread:.3e} exceeds {tolerance:.1e}"
        ),
    )


class TestRestrictedBoltzmannMachineIsingEquivalence(unittest.TestCase):
    """Full-enumeration equivalence for the RBM Ising conversion."""

    def _random_rbm(self, num_visible, num_hidden, seed):
        """Build an RBM with float64 random parameters."""
        generator = torch.Generator().manual_seed(seed)
        model = RestrictedBoltzmannMachine(
            num_visible,
            num_hidden,
            quadratic_coef=torch.randn(
                (num_visible, num_hidden), generator=generator, dtype=torch.float64
            ),
            linear_bias=torch.randn(
                (num_visible + num_hidden,), generator=generator, dtype=torch.float64
            ),
            device=torch.device("cpu"),
        )
        return model

    def test_random_models_all_states(self):
        """The invariant holds for every state of several random RBMs.

        ``RestrictedBoltzmannMachine._to_ising_matrix`` allocates its matrix
        in the default dtype (float32) regardless of the parameter dtype, so
        even float64 models see float32 rounding in the matrix. The
        tolerance below still catches coefficient- or sign-level regressions
        (the #171 bug class), which shift the invariant by O(1) amounts.
        """
        for seed, (num_visible, num_hidden) in [
            (1, (3, 2)),
            (2, (4, 2)),
            (3, (2, 3)),
        ]:
            with self.subTest(seed=seed, shape=(num_visible, num_hidden)):
                model = self._random_rbm(num_visible, num_hidden, seed)
                states = enumerate_binary_states(num_visible + num_hidden)
                assert_energy_ising_equivalence(self, model, states, tolerance=1e-5)

    def test_default_float32(self):
        """The invariant survives float32 rounding at default precision."""
        generator = torch.Generator().manual_seed(4)
        model = RestrictedBoltzmannMachine(
            3,
            2,
            quadratic_coef=torch.randn((3, 2), generator=generator),
            linear_bias=torch.randn((5,), generator=generator),
            device=torch.device("cpu"),
        )
        states = enumerate_binary_states(5, dtype=torch.float32)
        assert_energy_ising_equivalence(self, model, states, tolerance=1e-4)

    def test_purely_linear_energy(self):
        """A zero-weight RBM isolates the bias conversion formula.

        With ``quadratic_coef = 0`` the Ising matrix contains only the
        ``linear_bias / 4`` terms, which is exactly the term issue #171
        fixed; any error there shows up here without quadratic masking.
        The 1e-5 tolerance reflects the float32 matrix allocation (see
        ``test_random_models_all_states``).
        """
        model = RestrictedBoltzmannMachine(
            3,
            2,
            quadratic_coef=torch.zeros((3, 2), dtype=torch.float64),
            linear_bias=torch.tensor(
                [0.7, -1.3, 2.1, -0.4, 1.9], dtype=torch.float64
            ),
            device=torch.device("cpu"),
        )
        states = enumerate_binary_states(5)
        assert_energy_ising_equivalence(self, model, states, tolerance=1e-5)


class TestBoltzmannMachineIsingEquivalence(unittest.TestCase):
    """Full-enumeration equivalence for the BM Ising conversions."""

    def _random_bm(self, num_nodes, seed):
        """Build a BM with float64 random parameters."""
        generator = torch.Generator().manual_seed(seed)
        return BoltzmannMachine(
            num_nodes,
            quadratic_coef=torch.randn(
                (num_nodes, num_nodes), generator=generator, dtype=torch.float64
            ),
            linear_bias=torch.randn(
                (num_nodes,), generator=generator, dtype=torch.float64
            ),
            device=torch.device("cpu"),
        )

    def test_random_models_all_states(self):
        """The invariant holds for every state of several random BMs."""
        for seed, num_nodes in [(11, 4), (12, 5)]:
            with self.subTest(seed=seed, num_nodes=num_nodes):
                model = self._random_bm(num_nodes, seed)
                states = enumerate_binary_states(num_nodes)
                assert_energy_ising_equivalence(self, model, states, tolerance=1e-9)

    def test_auxiliary_spin_gauge_invariance(self):
        """Flipping every spin (including the auxiliary) preserves s^T M s.

        ``sample()`` maps solver spins back with ``x = (s * s_aux + 1) / 2``,
        which is only well defined because the quadratic form cannot tell
        the globally flipped sector from the original one.
        """
        model = self._random_bm(4, 13)
        matrix = torch.tensor(model.get_ising_matrix(), dtype=torch.float64)
        states = enumerate_binary_states(4)
        spins_plus = spins_with_auxiliary(states)
        spins_minus = -spins_plus
        values_plus = torch.einsum("bi,ij,bj->b", spins_plus, matrix, spins_plus)
        values_minus = torch.einsum("bi,ij,bj->b", spins_minus, matrix, spins_minus)
        self.assertTrue(torch.allclose(values_plus, values_minus, atol=1e-12))

    def test_conditional_hidden_matrix_all_states(self):
        """The hidden-side submatrix matches energies for fixed visible units.

        For each fixed visible assignment ``v``, the conditional Ising
        matrix produced by ``_hidden_to_ising_matrix(v)`` must satisfy the
        same constant-sum invariant over all hidden completions ``h``, with
        the model energy evaluated on the full state ``[v, h]``.
        """
        model = self._random_bm(5, 14)
        num_nodes = 5
        num_visible = 2
        visible_choices = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        hidden_states = enumerate_binary_states(num_nodes - num_visible)
        for visible in visible_choices:
            with self.subTest(visible=visible.tolist()):
                submatrix = torch.tensor(
                    model._hidden_to_ising_matrix(visible), dtype=torch.float64
                )
                full_states = torch.cat(
                    [
                        visible.expand(len(hidden_states), -1),
                        hidden_states,
                    ],
                    dim=1,
                )
                energies = model(full_states)
                ising_values = ising_quadratic_form(submatrix, hidden_states)
                total = energies + ising_values
                spread = (total.max() - total.min()).item()
                self.assertLessEqual(
                    spread,
                    1e-9,
                    msg=(
                        "E_model([v, h]) + s_h^T M_sub s_h must be constant "
                        f"over hidden states; spread {spread:.3e}"
                    ),
                )


class TestGaussianBernoulliIsingEquivalence(unittest.TestCase):
    """Equivalence for the GBRBM Ising conversion of the Bernoulli side."""

    def _effective_bernoulli_energy(self, model, hidden_states):
        """Closed-form Bernoulli-side energy after integrating out Gaussians.

        Completing the square of the Gaussian integral
        ``Z(h) = integral exp(-E(v, h)) dv`` gives

            -log Z(h) = -1/2 h^T (W^T D W) h
                        - (W^T D mu + b)^T h + const(h),

        with ``D = diag(1/var)``. This expression is derived independently
        of the matrix construction in ``_to_ising_matrix``, so a wrong
        coefficient there cannot cancel out here.

        Args:
            model: GBRBM instance.
            hidden_states (torch.Tensor): Bernoulli states (B, num_bernoulli).

        Returns:
            torch.Tensor: Effective energies of shape (B,).
        """
        precision = torch.diag(1.0 / model.var.detach())
        weights = model.quadratic_coef.detach()
        quadratic = weights.t() @ precision @ weights
        linear = weights.t() @ (model.mu.detach() / model.var.detach())
        linear = linear + model.linear_bias.detach()
        return (
            -0.5 * torch.einsum("bi,ij,bj->b", hidden_states, quadratic, hidden_states)
            - hidden_states @ linear
        )

    def test_random_models_all_states(self):
        """The invariant holds for every Bernoulli state of random GBRBMs.

        Parameters are replaced via ``.data`` assignments (the style of
        ``test_gbrbm.py``) because the constructor re-initializes supplied
        values and creates ``quadratic_coef``/``linear_bias`` in float32
        regardless of the ``dtype`` argument.
        """
        for seed in (21, 22):
            with self.subTest(seed=seed):
                generator = torch.Generator().manual_seed(seed)
                model = GaussianBernoulliRestrictedBoltzmannMachine(
                    num_visible=2,
                    num_hidden=3,
                    dtype=torch.float64,
                    device=torch.device("cpu"),
                )
                model.mu.data = torch.randn(
                    (2,), generator=generator, dtype=torch.float64
                )
                model.log_var.data = torch.log(
                    torch.rand((2,), generator=generator, dtype=torch.float64) + 0.5
                )
                model.quadratic_coef.data = torch.randn(
                    (2, 3), generator=generator, dtype=torch.float64
                )
                model.linear_bias.data = torch.randn(
                    (3,), generator=generator, dtype=torch.float64
                )
                hidden_states = enumerate_binary_states(model.num_bernoulli)
                energies = self._effective_bernoulli_energy(model, hidden_states)
                matrix = torch.tensor(
                    model.get_ising_matrix(), dtype=torch.float64
                )
                total = energies + ising_quadratic_form(matrix, hidden_states)
                spread = (total.max() - total.min()).item()
                self.assertLessEqual(
                    spread,
                    1e-9,
                    msg=(
                        "E_eff(h) + s^T M s must be constant over Bernoulli "
                        f"states; spread {spread:.3e}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
