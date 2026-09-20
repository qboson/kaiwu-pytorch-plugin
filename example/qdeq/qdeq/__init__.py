"""qdeq: a reusable Deep Equilibrium (DEQ) fixed-point plugin for PyTorch."""

from qdeq.core import QDEQ
from qdeq.ising_sa import IsingSolverConfig
from qdeq.solvers import anderson_solver, ising_sa_solver

__all__ = ["QDEQ", "IsingSolverConfig", "anderson_solver", "ising_sa_solver"]
