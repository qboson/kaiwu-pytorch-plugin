import unittest

import torch

from qdeq.core import QDEQ
from qdeq.solvers import anderson_solver
from qdeq.tests.test_core import LinearFixedPoint


class RecordingContext:
    """A context manager factory that appends "enter"/"exit" to a shared log."""

    def __init__(self, log, name):
        self.log = log
        self.name = name

    def __call__(self):
        return self

    def __enter__(self):
        self.log.append(f"{self.name}_enter")

    def __exit__(self, *exc_info):
        self.log.append(f"{self.name}_exit")
        return False


class TestQDEQContexts(unittest.TestCase):
    def test_solve_ctx_brackets_noqrad_solve_and_reattach_ctx_brackets_reattach(self):
        torch.manual_seed(0)
        bsz, d, L = 2, 4, 1
        func = LinearFixedPoint(d)
        x = torch.randn(bsz, d, L)
        z_init = torch.zeros(bsz, d, L)

        log: list[str] = []
        deq = QDEQ(
            f_solver=anderson_solver,
            f_thres=20,
            b_thres=20,
            solve_ctx=RecordingContext(log, "solve"),
            reattach_ctx=RecordingContext(log, "reattach"),
        )
        deq.train()

        z_out, _info = deq(func, z_init, x)
        loss = (z_out ** 2).sum()
        loss.backward()

        # solve_ctx must fully bracket the no-grad solve, and must be closed
        # before reattach_ctx opens; reattach_ctx must fully bracket the
        # single differentiable call (backward runs later, outside both).
        self.assertEqual(log, ["solve_enter", "solve_exit", "reattach_enter", "reattach_exit"])


if __name__ == "__main__":
    unittest.main()
