"""
QUBO formulation for the molecular generation problem.

From the paper (Methods – Optimization strategy for molecular design):

  "The main idea behind the proposed solution technique is the use of a
   weighted linear model Σ_{i,j>i} β_{ij} A_{ij} that serves as a surrogate
   model to approximate the objective function F_θ(y*, G_c(X, A))."

  Structural constraints (valency):
  "Σ_{j>i} A_{ij} + Σ_{i>j} A_{ji} ≤ v_i  for each atom i"

  Encoded as a quadratic penalty Q_c = Σ_i Q_i where Q_i is given by
  Eq. (2) in the paper (cross-side and same-side pairs differ):

    Q_i = (1 − v_i)(Σ_{j>i} A_{ij} + Σ_{i>j} A_{ji})
        + Σ_{j>i} Σ_{i>k} A_{ij}·A_{ki}
        + 2(Σ_{j>i} Σ_{k>j} A_{ij}·A_{ik} + Σ_{i>j} Σ_{j>k} A_{ji}·A_{ki})

  Note: No slack variables are used (confirmed by Supplementary Table 4
  where QUBO variable counts equal n*(n-1)/2 for various molecule sizes).

Variables:
  x_{ij} ∈ {0,1}  for i < j   (bond indicators, n*(n-1)/2 variables only)

The full QUBO objective:
  min_x  Σ_{i<j} β_{ij} x_{ij}  +  λ · Q_c(x)
where Q_c is expanded into an upper-triangular matrix form.

SA solver:
  Uses D-Wave's Neal (simulated annealing) as a proxy for the D-Wave Advantage
  quantum annealer used in the paper.

Modifications vs. paper (to compensate for classical SA vs. quantum annealer):
  - Modified valency penalty: Q_i = d_i·(d_i − v_i) + ε·(v_i − d_i)²
    eliminates the d_i=0 minimum, making d_i=v_i the sole global minimum.
  - Ring-closing incentive: rewards triangle-closing bonds to encourage rings.
  - Connectivity incentive: small reward for each bond to encourage more bonds.
  - SA improvements: scaled initial temperature, slower cooling, warm-start.
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from qc_mol_design.config import CFG, ATOM_VALENCY


# ─────────────────────────────────────────────────────────────────────────────
# QUBO variable index management
# ─────────────────────────────────────────────────────────────────────────────

class MoleculeQUBO:
    """
    Builds and maintains the QUBO for a molecule with fixed atom types and
    variable bond topology.

    Variable layout in the QUBO vector:
      [bond_0, bond_1, ..., bond_{n_bond-1}]
    where n_bonds = n*(n-1)/2, following the paper's formulation (no slack).
    """

    def __init__(self, atom_symbols: List[str], max_valencies: List[int]) -> None:
        self.atoms         = atom_symbols
        self.max_valencies = max_valencies
        self.n             = len(atom_symbols)

        # ── Bond variable indices (i < j) ────────────────────────────────
        self.bond_pairs: List[Tuple[int, int]] = []
        self.bond_to_idx: Dict[Tuple[int, int], int] = {}
        k = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.bond_pairs.append((i, j))
                self.bond_to_idx[(i, j)] = k
                k += 1
        self.n_bonds = k
        self.n_vars  = k  # No slack variables — paper Eq.(2)

        # ── Precompute triangle index triples for ring incentive ─────────
        # For each triple (i, j, k) with i < j < k, store the three
        # bond indices and the three "closing" pairs.  This avoids
        # recomputing bond_idx on every build() call.
        self._triangles: List[Tuple[int, int, int, int, int, int]] = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                for k in range(j + 1, self.n):
                    idx_ij = self.bond_to_idx[(i, j)]
                    idx_ik = self.bond_to_idx[(i, k)]
                    idx_jk = self.bond_to_idx[(j, k)]
                    # Store as (bond_a, bond_b, closing_bond) for each
                    # orientation of the triangle
                    self._triangles.append((idx_ij, idx_ik, idx_jk))
                    self._triangles.append((idx_ij, idx_jk, idx_ik))
                    self._triangles.append((idx_ik, idx_jk, idx_ij))

    def bond_idx(self, i: int, j: int) -> int:
        """Return QUBO variable index for bond (i,j), normalising i < j."""
        if i > j:
            i, j = j, i
        return self.bond_to_idx[(i, j)]

    def bond_vector(self, adj: np.ndarray) -> np.ndarray:
        """Flatten upper-triangular adjacency → bond feature vector (n_bonds,)."""
        vec = np.zeros(self.n_bonds, dtype=np.float32)
        for k, (i, j) in enumerate(self.bond_pairs):
            vec[k] = adj[i, j]
        return vec

    def decode_sample(self, sample: Dict[int, int]) -> np.ndarray:
        """Decode a QUBO sample → adjacency matrix (n×n)."""
        adj = np.zeros((self.n, self.n), dtype=np.int32)
        for k, (i, j) in enumerate(self.bond_pairs):
            v = int(sample.get(k, 0))
            adj[i, j] = adj[j, i] = v
        return adj

    # ── QUBO matrix construction ──────────────────────────────────────────

    def build(
        self,
        surrogate_beta:      Optional[np.ndarray] = None,   # (n_bonds,) or None
        penalty:             float = 100.0,
        previous_solutions:  Optional[List[np.ndarray]] = None,  # list of bond vectors
        diversity_gamma:     float = 0.0,     # diversity penalty strength
        perturbation:        float = 0.0,     # random diagonal perturbation
        valency_deficit_eps: float = 0.0,     # ε for modified penalty
        ring_incentive:      float = 0.0,     # reward for triangle-closing bonds
        connectivity_incentive: float = 0.0,  # reward for additional bonds
    ) -> np.ndarray:
        """
        Build upper-triangular QUBO matrix Q of shape (n_vars, n_vars).

        Objective: x^T Q x  is minimised.

        The structural constraint penalty encodes valency constraints
        Σ_{j≠i} A_{ij} ≤ v_i as a QUBO without slack variables, consistent
        with the paper's Eq. (2) and confirmed by Supplementary Table 4
        (QUBO variable counts equal n*(n-1)/2).

        Compensated-mode penalty (not used by strict reproduction):
          Q_i = d_i·(d_i − v_i) + ε·(v_i − d_i)²

        Paper Eq. (2) is not d_i·(d_i-v_i): its cross-side incident-edge
        products have coefficient 1 and its same-side products coefficient 2.

        Expansion of modified penalty:
          Q_i = (1 + ε)·d_i² − v_i·(1 + 2ε)·d_i + ε·v_i²
              = (1 − v_i·(1+2ε) + ε)·Σ Aij
                + 2·(1+ε)·Σ_{j<k} Aij·Aik
                + const(ε·v_i²)

        Properties (with ε > 0):
          - Q_i = ε·v_i² when d_i = 0  (penalised — atoms should form bonds)
          - Q_i = 0 when d_i = v_i       (sole global minimum)
          - Q_i > 0 when d_i > v_i       (penalizes excess bonds)

        Additional terms (not in paper, compensate for classical SA):
          - ring_incentive: for each triangle (i,j,k), if bonds (i,j) and
            (i,k) are present, reward bond (j,k) via negative quadratic
            coupling.  This encourages closing triangles → ring formation.
          - connectivity_incentive: small negative linear term for every
            bond variable, encouraging more bonds overall.
          - diversity_gamma: penalises solutions similar to previous_solutions.
          - perturbation: random diagonal noise to break symmetry.
        """
        Q = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)

        # ── Surrogate objective ─────────────────────────────────────────
        if surrogate_beta is not None:
            beta = np.asarray(surrogate_beta).ravel()
            for k in range(min(len(beta), self.n_bonds)):
                Q[k, k] += float(beta[k])

        # ── Valency constraint penalty ───────────────────────────────────
        # Strict mode follows paper Eq. (2) exactly: cross-side incident
        # edge pairs have coefficient 1; same-side pairs have coefficient 2.
        n = self.n
        eps = valency_deficit_eps
        if eps > 0 and eps <= max(self.max_valencies) - 1:
            raise ValueError(
                "valency_deficit_eps must exceed max(valency)-1 in compensated mode"
            )
        for i in range(n):
            vi = self.max_valencies[i]
            lower = [self.bond_idx(k, i) for k in range(i)]
            upper = [self.bond_idx(i, j) for j in range(i + 1, n)]
            incident = lower + upper

            if eps == 0.0:
                for idx in incident:
                    Q[idx, idx] += penalty * (1.0 - vi)

                # Eq. (2): edges whose other endpoints straddle i.
                for left_idx in lower:
                    for right_idx in upper:
                        lo, hi = sorted((left_idx, right_idx))
                        Q[lo, hi] += penalty

                # Eq. (2): edges whose other endpoints lie on the same side.
                for side in (lower, upper):
                    for p_idx in range(len(side)):
                        for q_idx in range(p_idx + 1, len(side)):
                            lo, hi = sorted((side[p_idx], side[q_idx]))
                            Q[lo, hi] += 2.0 * penalty
            else:
                # Historical compensated mode; explicitly not the paper Qc.
                linear_coeff = 1.0 - vi * (1.0 + 2.0 * eps) + eps
                for idx in incident:
                    Q[idx, idx] += penalty * linear_coeff
                quad_coeff = 2.0 * (1.0 + eps)
                for p_idx in range(len(incident)):
                    for q_idx in range(p_idx + 1, len(incident)):
                        lo, hi = sorted((incident[p_idx], incident[q_idx]))
                        Q[lo, hi] += penalty * quad_coeff

        # ── Ring-closing incentive (triangle reward) ─────────────────────
        # For each triangle (i,j,k), if bonds (a,b) and (a,c) are both
        # present, reward the closing bond (b,c) by adding negative
        # quadratic coupling between each existing bond and the closing bond.
        # This approximates a 3-body "if two sides present, reward third"
        # interaction using only pairwise QUBO terms.
        if ring_incentive > 0:
            for idx_a, idx_b, idx_close in self._triangles:
                lo1 = min(idx_a, idx_close)
                hi1 = max(idx_a, idx_close)
                lo2 = min(idx_b, idx_close)
                hi2 = max(idx_b, idx_close)
                Q[lo1, hi1] -= ring_incentive
                Q[lo2, hi2] -= ring_incentive

        # ── Connectivity incentive (bond count reward) ───────────────────
        # Small negative linear term for every bond variable encourages
        # more bonds overall.  Combined with valency constraints, this
        # naturally leads to connected, fully-valent structures with rings
        # (a connected graph with > n-1 bonds must contain cycles).
        if connectivity_incentive > 0:
            for k in range(self.n_bonds):
                Q[k, k] -= connectivity_incentive

        # ── Diversity penalty: discourage similarity to previous solutions ─
        if previous_solutions is not None and diversity_gamma > 0:
            for bvec in previous_solutions:
                b = np.asarray(bvec).ravel()
                scale = diversity_gamma / self.n_bonds
                for k in range(min(len(b), self.n_bonds)):
                    if b[k] > 0.5:
                        Q[k, k] += scale   # penalise keeping a previous 1 bit
                    else:
                        Q[k, k] -= scale   # penalise keeping a previous 0 bit

        # ── Random perturbation: break symmetry for exploration ──────────
        if perturbation > 0:
            diag_noise = np.random.uniform(
                -perturbation, perturbation, size=self.n_vars
            )
            Q[np.diag_indices_from(Q)] += diag_noise

        return Q


# ─────────────────────────────────────────────────────────────────────────────
# QUBO solver hierarchy
# ─────────────────────────────────────────────────────────────────────────────
# 1. Simulated Quantum Annealing (SQA) via dwave.samplers — closest classical
#    approximation to D-Wave quantum annealing.  SQA uses path-integral Monte
#    Carlo to simulate quantum tunnelling, which is the key mechanism that
#    allows D-Wave hardware to find ring-containing molecular structures.
# 2. D-Wave Neal (classical SA) — standard simulated annealing fallback.
# 3. Built-in Metropolis SA — no external dependency fallback.

def solve_qubo(
    Q:              np.ndarray,
    num_reads:      int  = 100,
    num_sweeps:     int  = 10000,
    initial_states: Optional[List[Dict[int, int]]] = None,
    cim_prepared: Optional[dict] = None,
) -> List[Dict[int, int]]:
    """
    Solve a QUBO  min_{x∈{0,1}^n}  x^T Q x.

    Backend selection (env var ``QC_QUBO_SOLVER``):
      • ``kaiwu``        → Kaiwu SDK (Boson Quantum); see _solve_kaiwu.
      • ``neal``         → force D-Wave Neal (classical SA) directly.
      • ``sqa`` (unset)  → default priority chain below.

    Solver priority (best → worst):
      1. Simulated Quantum Annealing (SQA) — simulates quantum tunnelling
      2. D-Wave Neal (classical SA)
      3. Built-in Metropolis SA (no external dependency)

    Args:
        Q: upper-triangular QUBO matrix
        num_reads: number of independent annealing runs
        num_sweeps: number of sweeps per run
        initial_states: optional list of initial states for warm-starting.
            Each state is a dict {var_idx: 0_or_1}.

    Returns a list of solution dicts {var_idx: 0_or_1}, sorted best first.
    """
    backend = os.environ.get("QC_QUBO_SOLVER", "").lower()

    # Opt-in Kaiwu backend (leaves the default SQA path untouched).
    if backend == "kaiwu":
        return _solve_kaiwu(Q, num_reads, num_sweeps, initial_states, prepared=cim_prepared)

    # Opt-in Kaiwu CIM backend (Coherent Ising Machine, 14-bit precision).
    if backend == "kaiwu_cim":
        return _solve_kaiwu_cim(Q, num_reads, num_sweeps, initial_states, prepared=cim_prepared)

    # Opt-in D-Wave Neal backend (classical SA), for solver comparison.
    if backend == "neal":
        return _solve_neal(Q, num_reads, num_sweeps, initial_states)

    # An explicitly requested SQA comparison arm must never silently become a
    # different solver.  Let dependency/runtime failures surface so the run is
    # marked failed instead of being mislabeled as SQA in run_meta.json.
    if backend == "sqa":
        return _solve_sqa(Q, num_reads, num_sweeps, initial_states)

    # With no explicit backend, retain the convenient fallback chain.
    try:
        return _solve_sqa(Q, num_reads, num_sweeps, initial_states)
    except Exception:
        pass

    # Fallback to Neal SA
    try:
        return _solve_neal(Q, num_reads, num_sweeps, initial_states)
    except ImportError:
        return _solve_builtin_sa(Q, num_reads, num_sweeps, initial_states)


def _solve_sqa(
    Q:              np.ndarray,
    num_reads:      int,
    num_sweeps:     int,
    initial_states: Optional[List[Dict[int, int]]] = None,
) -> List[Dict[int, int]]:
    """Simulated Quantum Annealing via dwave.samplers.

    SQA uses path-integral Monte Carlo to simulate the quantum tunnelling
    effect that allows D-Wave quantum annealers to escape local minima.
    This is the key mechanism that enables D-Wave to find ring-containing
    molecular structures where classical SA gets stuck in tree minima.

    The PathIntegralAnnealingSampler implements the Suzuki-Trotter
    decomposition of the quantum Ising model, where quantum fluctuations
    are simulated via an imaginary-time path integral.
    """
    from dwave.samplers import PathIntegralAnnealingSampler
    n = Q.shape[0]
    Q_dict = {
        (i, j): float(Q[i, j])
        for i in range(n)
        for j in range(i, n)
        if Q[i, j] != 0.0
    }
    sampler = PathIntegralAnnealingSampler()

    kwargs = {
        "num_reads": num_reads, "num_sweeps": num_sweeps,
        "seed": int(getattr(CFG, "seed", 0)) or None,
    }
    # SQA-specific parameters:
    # - num_trotter_slices: number of Trotter replicas (default varies)
    #   More slices = better quantum effect approximation but slower
    #   Paper uses D-Wave Advantage with ~5000 qubits; we use a
    #   moderate number of Trotter slices for balance.
    kwargs["num_trotter_slices"] = 4

    if initial_states and len(initial_states) > 0:
        init_list = []
        for r in range(num_reads):
            init = initial_states[r % len(initial_states)]
            init_list.append(init)
        kwargs["initial_states"] = init_list

    response = sampler.sample_qubo(Q_dict, **kwargs)
    return [dict(s) for s in response.samples()]


def _solve_neal(
    Q:              np.ndarray,
    num_reads:      int,
    num_sweeps:     int,
    initial_states: Optional[List[Dict[int, int]]] = None,
) -> List[Dict[int, int]]:
    import neal
    n = Q.shape[0]
    Q_dict = {
        (i, j): float(Q[i, j])
        for i in range(n)
        for j in range(i, n)
        if Q[i, j] != 0.0
    }
    sampler  = neal.SimulatedAnnealingSampler()

    # Neal supports initial_state parameter per read
    kwargs = {
        "num_reads": num_reads, "num_sweeps": num_sweeps,
        "seed": int(getattr(CFG, "seed", 0)) or None,
    }
    if initial_states and len(initial_states) > 0:
        # Convert list of dicts to Neal's initial_state format
        # Neal expects a dict mapping variable index to initial value
        # for each read.  We distribute initial states round-robin.
        init_list = []
        for r in range(num_reads):
            init = initial_states[r % len(initial_states)]
            init_list.append(init)
        kwargs["initial_states"] = init_list

    response = sampler.sample_qubo(Q_dict, **kwargs)
    return [dict(s) for s in response.samples()]


def _solve_builtin_sa(
    Q:              np.ndarray,
    num_reads:      int,
    num_sweeps:     int,
    initial_states: Optional[List[Dict[int, int]]] = None,
) -> List[Dict[int, int]]:
    """Fallback Metropolis SA implementation (no external dependency).

    Improvements over basic SA:
      - Scaled initial temperature proportional to QUBO energy scale
      - Slower cooling schedule for more exploration
      - Warm-start from provided initial states
    """
    n = Q.shape[0]

    def energy(x: np.ndarray) -> float:
        return float(x @ Q @ x)

    # Scale initial temperature to QUBO energy scale
    q_max = float(np.abs(Q).max())
    T_init = max(1.0, q_max * CFG.sa_initial_temp_scale)
    cooling_rate = CFG.sa_cooling_rate

    solutions = []
    for read_idx in range(num_reads):
        # Warm-start: use initial state if available, else random
        if initial_states and len(initial_states) > 0:
            init = initial_states[read_idx % len(initial_states)]
            x = np.array([float(init.get(i, 0)) for i in range(n)],
                         dtype=np.float64)
        else:
            x = np.random.randint(0, 2, n).astype(np.float64)

        T = T_init
        e = energy(x)
        for _ in range(num_sweeps):
            flip = random.randint(0, n - 1)
            x[flip] ^= 1
            new_e = energy(x)
            delta = new_e - e
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                e = new_e
            else:
                x[flip] ^= 1
            T *= cooling_rate
        solutions.append(({i: int(x[i]) for i in range(n)}, e))

    solutions.sort(key=lambda t: t[1])
    return [s for s, _ in solutions]


# ─────────────────────────────────────────────────────────────────────────────
# Kaiwu SDK backend  (Boson Quantum — classical SA optimiser / CIM)
# ─────────────────────────────────────────────────────────────────────────────
# Enable with:  export QC_QUBO_SOLVER=kaiwu
# Requires a license (even for the local classical SA): set credentials from
# https://platform.qboson.com via the environment:
#     export KAIWU_USER_ID=...
#     export KAIWU_SDK_CODE=...

_KAIWU_LICENSE_READY = False


def _ensure_kaiwu_license() -> None:
    """Activate the Kaiwu license once per process (idempotent).

    Two paths:
      • Credentials in the environment (KAIWU_USER_ID + KAIWU_SDK_CODE) →
        (re)activate via ``kw.license.init``.  Obtain them from
        https://platform.qboson.com.
      • No credentials → rely on a license file already activated on this
        machine (``kw.license.init`` persists one across processes).  If none
        exists, Kaiwu's solver raises its own clear license error.
    """
    global _KAIWU_LICENSE_READY
    if _KAIWU_LICENSE_READY:
        return
    if os.environ.get("KAIWU_CREDENTIALS_FILE"):
        from .cim_hardware import initialize_license
        initialize_license()
        _KAIWU_LICENSE_READY = True
        return
    uid  = os.environ.get("KAIWU_USER_ID")
    code = os.environ.get("KAIWU_SDK_CODE")
    if uid and code:
        import kaiwu as kw
        kw.license.init(user_id=uid, sdk_code=code)
    _KAIWU_LICENSE_READY = True


def _solve_kaiwu(
    Q:              np.ndarray,
    num_reads:      int,
    num_sweeps:     int,
    initial_states: Optional[List[Dict[int, int]]] = None,
    prepared: Optional[dict] = None,
) -> List[Dict[int, int]]:
    """Solve  min_x  x^T Q x  with the Kaiwu SDK simulated-annealing optimiser.

    Drop-in replacement for _solve_sqa / _solve_neal: identical signature and
    return contract (list of {var_idx: 0/1} dicts, lowest QUBO energy first).

    Pipeline
    --------
    1. QUBO → Ising via ``kw.qubo.qubo_matrix_to_ising_matrix``.  Kaiwu appends
       ONE auxiliary spin, so the Ising matrix is (n+1)×(n+1) and
       ``ising_energy(s) + offset == qubo_energy(x)``.
    2. ``kw.classical.SimulatedAnnealingOptimizer.solve`` returns a pool of spin
       solutions, each of length n+1, with Ising energy  H(s) = −sᵀ M s.
    3. Decode each spin row to binary with the (brute-force-validated, gauge-
       invariant) rule  ``x_i = (s_i · s_aux + 1) / 2`` ,  ``s_aux = s[-1]``.
    4. Score with  xᵀ Q x , sort ascending, return best-first.

    Parameter mapping
    -----------------
    ``num_reads``  → kept solution-pool size (``size_limit``).
    ``num_sweeps`` → total Metropolis steps, spread across the temperature
                     schedule (``iterations_per_t × n_temperature_levels``).
    The initial temperature is scaled to the Ising coupling magnitude so the
    anneal is not frozen by this QUBO's very large (~1e9) coefficients.

    To target the real Coherent Ising Machine instead, swap the optimiser for
    ``kw.cim.CIMOptimizer(task_name=..., project_no=...)`` (needs cloud quota);
    its ``.solve(ising_matrix)`` returns the same spin-pool format, so steps
    1/3/4 are unchanged.
    """
    import kaiwu as kw

    _ensure_kaiwu_license()

    Q = np.asarray(Q, dtype=np.float64)
    n = Q.shape[0]

    # 1. QUBO → Ising  ((n+1)×(n+1) matrix + constant offset) ────────────────
    # Kaiwu 1.1 exposes matrix conversion under ``kw.qubo``.  Older internal
    # builds placed it under ``kw.core``; prefer the public 1.1 API while
    # retaining compatibility with those builds.
    converter = getattr(kw.qubo, "qubo_matrix_to_ising_matrix", None)
    if converter is None:
        converter = kw.core.qubo_matrix_to_ising_matrix
    ising_mat, _offset = converter(Q)
    ising_mat = np.asarray(ising_mat, dtype=np.float64)
    if prepared is not None:
        from .spin_encoding import physical_matrix
        ising_mat = np.asarray(physical_matrix(prepared["ising"], prepared), dtype=np.float64)

    # Temperature schedule scaled to the off-diagonal coupling magnitude ─────
    off = ising_mat.copy()
    np.fill_diagonal(off, 0.0)
    scale = float(np.abs(off).max())
    temp_scale = float(getattr(CFG, "sa_initial_temp_scale", 1.0))
    alpha      = float(getattr(CFG, "sa_cooling_rate", 0.99))
    if not (0.0 < alpha < 1.0):            # guard against bad config values
        alpha = 0.99
    T0     = max(1.0, scale * max(temp_scale, 1e-3))
    ratio  = 1e-3                          # cutoff / initial → ~3 decades
    cutoff = T0 * ratio
    # 1 "sweep" = one update per variable (D-Wave convention), so the requested
    # work is num_sweeps * n single-spin flips, spread across the temperature
    # levels.  (Without the * n the anneal does far too few flips on big QUBOs.)
    n_temps     = max(1, int(math.ceil(math.log(ratio) / math.log(alpha))))
    total_flips = max(int(num_sweeps) * n, n_temps)
    iters_per_t = max(1, int(round(total_flips / n_temps)))

    seed = int(getattr(CFG, "seed", 0)) or None

    optimizer = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature = T0,
        alpha               = alpha,
        cutoff_temperature  = cutoff,
        iterations_per_t    = iters_per_t,
        size_limit          = max(1, int(num_reads)),
        rand_seed           = seed,
        process_num         = 1,
    )

    # Optional warm start: Kaiwu accepts ONE spin init vector (length n+1) ────
    init_solution = None
    if initial_states:
        first = initial_states[0]
        x0    = np.array([1.0 if int(first.get(i, 0)) else 0.0
                          for i in range(n)], dtype=np.float64)
        init_solution = np.concatenate([2.0 * x0 - 1.0, [1.0]])   # s_aux = +1

    # 2. Solve → spin-solution pool, shape (k, n+1) ──────────────────────────
    raw = np.asarray(optimizer.solve(ising_mat, init_solution=init_solution))
    if raw.ndim == 1:
        raw = raw[None, :]
    if prepared is not None:
        from .spin_encoding import logical_spins
        raw = logical_spins(raw, prepared)

    # 3. Decode spins → binary,  4. score with xᵀQx & sort best-first ────────
    scored: List[Tuple[float, Dict[int, int]]] = []
    for row in raw:
        s = np.asarray(row, dtype=np.float64)
        if s.shape[0] != n + 1:            # defensive: skip malformed rows
            continue
        s_aux = s[-1] if s[-1] != 0 else 1.0
        x = (((s[:-1] * s_aux) + 1.0) / 2.0 > 0.5).astype(int)
        scored.append((float(x @ Q @ x), {i: int(x[i]) for i in range(n)}))

    if not scored:                         # never hand back an empty result
        raise RuntimeError("SA returned no valid spin rows; tuning forbids fallback")

    scored.sort(key=lambda t: t[0])
    return [d for _, d in scored]


# ─────────────────────────────────────────────────────────────────────────────
# Kaiwu CIM backend  (Coherent Ising Machine — 14-bit precision hardware)
# ─────────────────────────────────────────────────────────────────────────────
# Enable with:  export QC_QUBO_SOLVER=kaiwu_cim
# Requires Kaiwu license + CIM cloud quota (Boson Quantum CPQC platform).
# Set credentials via the environment:
#     export KAIWU_USER_ID=...
#     export KAIWU_SDK_CODE=...
# Set the project number and checkpoint dir via config or env:
#     export KAIWU_CIM_PROJECT_NO=26072131
#     export KAIWU_CIM_CACHE_DIR=/tmp/kaiwu_cim_cache
#
# Precision constraint: CIM accepts signed 14-bit integer Ising matrices
# (max element ±8191).  The adaptive tier selection below ensures the
# submitted matrix always fits this constraint with minimal overhead.

_CIM_PROJECT_NO = os.environ.get("KAIWU_CIM_PROJECT_NO", "")
_CIM_CACHE_DIR  = os.environ.get("KAIWU_CIM_CACHE_DIR", "/tmp/kaiwu_cim_cache")


def _solve_kaiwu_cim(Q, num_reads=100, num_sweeps=10000, initial_states=None, bit_width=14, prepared=None):
    """Submit the full QUBO to CIM with audited precision adaptation."""
    from qc_mol_design.optimization.cim_hardware import solve_hardware
    return solve_hardware(Q, num_reads=num_reads, prepared=prepared)


def _decode_cim_pool(
    raw:   np.ndarray,
    Q:     np.ndarray,
    n:     int,
    n_ising: int,
) -> List[Dict[int, int]]:
    """Decode a pool of CIM spin solutions → QUBO binary dicts, sorted."""
    raw = np.asarray(raw)
    if raw.ndim == 1:
        raw = raw[None, :]
    if (raw.ndim != 2 or raw.shape[1] != n_ising or n_ising != n + 1
            or not len(raw) or not np.isin(raw, [-1, 1]).all()):
        raise RuntimeError("CIM returned an empty or malformed spin pool; no SA fallback")
    scored: List[Tuple[float, Dict[int, int]]] = []
    for row in raw:
        s = np.asarray(row, dtype=np.float64)
        if s.shape[0] != n_ising:
            continue
        s_aux = s[-1] if s[-1] != 0 else 1.0
        x = (((s[:-1] * s_aux) + 1.0) / 2.0 > 0.5).astype(int)
        scored.append((float(x @ Q @ x), {i: int(x[i]) for i in range(n)}))

    scored.sort(key=lambda t: t[0])
    return [d for _, d in scored]
