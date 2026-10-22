"""Contract tests for the public ``QuadraticLinearSolver``.

``solve(A, l)`` encodes ``f(x) = 0.5 x^T A x + l^T x`` over binary ``x`` as an
``(n+1) x (n+1)`` Ising matrix in the auxiliary-spin layout used by the Kaiwu
SDK: the top-left block carries the pair coefficients, the last column carries
the fields, and the last row stays zero. For ``y = [2x - 1, +1]`` the matrix must
satisfy

    y^T M y = f(x) - c

with a constant ``c`` that does not depend on ``x``. ``_solve_ising_local_search``
relies on exactly this: it descends on ``y^T M y``, so the affine relation is what
makes it minimize the QUBO objective.

The solver is exported from ``kaiwu.torch_plugin`` and backs
``FeatureSelectionWrapper``; it had no direct tests, and the sign/orientation of
this conversion is the area several open PRs are still arguing about, so these
tests enumerate every state instead of sampling.
"""

import itertools
import unittest

import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import QuadraticLinearSolver


def _states(n):
    """All binary states of dimension ``n`` plus their auxiliary-spin form."""
    for bits in itertools.product([0.0, 1.0], repeat=n):
        x = np.array(bits, dtype=float)
        yield x, np.r_[2.0 * x - 1.0, 1.0]


def _objective(x, quadratic, linear):
    """The QUBO objective ``f(x) = 0.5 x^T A x + l^T x``."""
    return 0.5 * x @ quadratic @ x + linear @ x


class TestQuadraticLinearSolver(unittest.TestCase):
    """Exact, enumeration-based checks of the QUBO -> Ising conversion."""

    def test_affine_equivalence_over_every_state(self):
        """``y^T M y == f(x) - c`` for every state, with one x-independent c."""
        for seed, n in [(0, 1), (1, 2), (7, 3), (11, 4)]:
            with self.subTest(seed=seed, n=n):
                rng = np.random.default_rng(seed)
                quadratic = rng.normal(size=(n, n))
                linear = rng.normal(size=n)
                ising = QuadraticLinearSolver().solve(quadratic, linear)

                x0, y0 = next(_states(n))
                constant = _objective(x0, quadratic, linear) - y0 @ ising @ y0

                for x, y in _states(n):
                    self.assertAlmostEqual(
                        float(y @ ising @ y),
                        float(_objective(x, quadratic, linear) - constant),
                        places=10,
                        msg=f"state {x}",
                    )

    def test_matrix_layout_matches_the_auxiliary_spin_convention(self):
        """Top-left block strictly upper triangular, last row zero."""
        n = 4
        rng = np.random.default_rng(3)
        quadratic = rng.normal(size=(n, n))
        linear = rng.normal(size=n)
        ising = QuadraticLinearSolver().solve(quadratic, linear)

        self.assertEqual(ising.shape, (n + 1, n + 1))
        self.assertTrue(np.allclose(ising[-1, :], 0.0), "last row must stay zero")
        self.assertTrue(
            np.allclose(np.tril(ising[:-1, :-1]), 0.0),
            "the pair block must be strictly upper triangular",
        )
        # the fields live in the auxiliary column; at least one must be nonzero
        self.assertFalse(np.allclose(ising[:-1, -1], 0.0))

    def test_minimizing_the_ising_form_returns_the_qubo_minimizer(self):
        """The property ``local_search`` depends on: argmin(spins) == argmin(x)."""
        for seed in range(5):
            with self.subTest(seed=seed):
                n = 3
                rng = np.random.default_rng(seed)
                quadratic = rng.normal(size=(n, n))
                linear = rng.normal(size=n)
                ising = QuadraticLinearSolver().solve(quadratic, linear)

                qubo_best = min(
                    (x for x, _ in _states(n)),
                    key=lambda x: _objective(x, quadratic, linear),
                )
                ising_best = min(
                    ((x, y @ ising @ y) for x, y in _states(n)),
                    key=lambda pair: pair[1],
                )[0]
                np.testing.assert_array_equal(qubo_best, ising_best)

    def test_non_symmetric_quadratic_term_is_symmetrized(self):
        """``A`` and ``(A + A^T) / 2`` encode the same objective."""
        n = 3
        rng = np.random.default_rng(5)
        quadratic = rng.normal(size=(n, n))
        linear = rng.normal(size=n)

        raw = QuadraticLinearSolver().solve(quadratic, linear)
        symmetric = QuadraticLinearSolver().solve(
            0.5 * (quadratic + quadratic.T), linear
        )
        np.testing.assert_allclose(raw, symmetric, atol=1e-12)

    def test_invalid_inputs_are_rejected(self):
        """Shapes and non-finite coefficients fail with the documented ValueError."""
        solver = QuadraticLinearSolver()
        with self.assertRaises(ValueError):
            solver.solve(np.zeros((3, 2)), np.zeros(3))
        with self.assertRaises(ValueError):
            solver.solve(np.zeros((3, 3)), np.zeros(2))
        with self.assertRaises(ValueError):
            solver.solve(np.array([[0.0, np.inf], [0.0, 0.0]]), np.zeros(2))

        with self.assertRaises(ValueError):
            QuadraticLinearSolver.qubo_matrix_to_ising_matrix(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            QuadraticLinearSolver.qubo_matrix_to_ising_matrix(
                np.array([[0.0, np.nan], [0.0, 0.0]])
            )


if __name__ == "__main__":
    unittest.main()
