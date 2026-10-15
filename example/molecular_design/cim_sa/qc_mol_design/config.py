"""
Global configuration for QC-assisted molecular design reproduction.

Reference: Ajagekar & You, npj Computational Materials 9, 143 (2023).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Property discretisation bins  (Supplementary Table 1 of the paper)
# Equal-frequency binning on the ZINC-12k training set (≈1000 molecules/bin)
# ─────────────────────────────────────────────────────────────────────────────
PROP_BINS: Dict[str, List[float]] = {
    # Supplementary Table 1: exact bin edges from equal-frequency binning
    "qed":  [0.138, 0.526, 0.615, 0.673, 0.721, 0.759, 0.790, 0.819, 0.847, 0.881, 0.947],
    "logp": [-4.580, 0.522, 1.295, 1.799, 2.222, 2.583, 2.945, 3.313, 3.688, 4.166, 6.663],
    "sas":  [1.399, 2.122, 2.316, 2.506, 2.694, 2.889, 3.093, 3.365, 3.754, 4.232, 6.791],
}

# Number of bins per property (len(edges) - 1)
N_BINS_PER_PROP: Dict[str, int] = {k: len(v) - 1 for k, v in PROP_BINS.items()}

# Maximum atom valencies used for structural constraints
ATOM_VALENCY: Dict[str, int] = {
    "C": 4, "N": 3, "O": 2, "F": 1, "P": 3,
    "S": 2, "Cl": 1, "Br": 1, "I": 1, "Si": 4,
}

# Nine heavy atom types referenced in the paper's Methods section
NINE_HEAVY_ATOMS: List[str] = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ATOM_FEATURE_SCHEMA = "paper_nine_heavy_atom_onehot_v1"


@dataclass
class Config:
    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Data ──────────────────────────────────────────────────────────────
    data_path: str = "zinc250k.csv"
    n_train:   int = 10_000
    n_valid:   int = 1_000
    n_test:    int = 1_000

    # ──olecular graph featurisation ─────────────────────────────────────
    # Methods p.9: node features distinguish exactly nine heavy-atom types.
    atom_feat_dim: int = 9
    max_degree:    int = 6           # degree-specific weight matrices: 0 … max_degree

    # ── GraphConv fingerprint (fixed weights, paper §Methods) ─────────────
    gcn_layers:  int = 2             # number of graph conv layers
    gcn_hidden:  int = 128           # hidden dim per conv layer
    fp_dim:      int = 256           # final fingerprint dimension (paper: 256)

    # ── CRBM ──────────────────────────────────────────────────────────────
    n_props:      int = 3            # number of simultaneously modelled properties
    n_hidden:     int = 64           # latent variable dimension h (paper: 64)

    @property
    def n_vis(self) -> int:
        """Total visible layer size = sum of all property bin counts."""
        return sum(N_BINS_PER_PROP[p] for p in ("qed", "logp", "sas"))

    @property
    def prop_bin_counts(self) -> List[int]:
        """Per-property bin counts in the order [qed, logp, sas]."""
        return [N_BINS_PER_PROP[p] for p in ("qed", "logp", "sas")]

    @property
    def prop_offsets(self) -> List[int]:
        """Start index of each property group within y vector."""
        counts = self.prop_bin_counts
        offsets = [0]
        for c in counts[:-1]:
            offsets.append(offsets[-1] + c)
        return offsets

    # ── Training ──────────────────────────────────────────────────────────
    # Figure 5a reports the energy-model learning curves through epoch 200.
    epochs:     int   = 200
    batch_size: int   = 128
    lr:         float = 1e-3
    weight_decay: float = 1e-5
    cd_k:       int   = 1            # contrastive divergence steps
    grad_clip:  float = 1.0
    patience:   int   = 10           # early stopping patience (paper: "consecutive steps")

    # ── QC-assisted training (SA-based) ───────────────────────────────────
    qc_sa_reads:       int   = 10       # SA reads per CRBM batch update
    qc_sa_sweeps:      int   = 500      # SA sweeps per read
    qc_per_sample:     bool  = True     # per-sample QUBO (paper) vs batch-mean (faster)
    qc_max_per_sample: int   = 0        # 0 = solve every batch element (strict paper mode)
    qc_onehot_penalty: float = 0.0      # 0 => derive a safe penalty from each CRBM QUBO

    # ── Molecular generation ──────────────────────────────────────────────
    n_explore:      int   = 10       # exploration phase steps (paper: 10)
    n_optimize:     int   = 90       # surrogate-guided steps (paper: 90)
    valency_penalty: float = 100.0   # λ in QUBO valency constraint
    penalty_scale:  float = 1.0     # compensated mode only; not used in strict mode
    sa_reads:       int   = 100      # SA reads for generation QUBO
    sa_sweeps:      int   = 10_000   # SA sweeps for generation QUBO (increased for ring search)
    n_ref_mols:     int   = 25       # paper/SI: fixed reference molecules

    reference_manifest: str = "references_25.json"
    qubo_mode:      str   = "strict"       # strict | compensated
    lambda_mode:    str   = "fixed"        # paper: fixed λ; dynamic_l1 is non-strict
    cim_penalty_ratio: float = 4.0         # opt-in hardware protocol: lambda / ||beta||inf
    # ── QUBO ring/connectivity incentives ────────────────────────────────────
    # Classical SA cannot replicate D-Wave quantum tunneling, so we add
    # explicit incentives to encourage ring formation and connectivity.
    valency_deficit_penalty: float = 4.0    # compensated mode only; > max(v_i)-1
    ring_incentive:          float = 5.0    # reward for triangle-closing bonds (needs to be strong enough)
    connectivity_incentive:  float = 1.0    # reward for additional bonds (encourages rings)
    ring_incentive_adaptive: bool  = True   # double ring_incentive if no rings found in exploration

    # ── SA solver tuning ─────────────────────────────────────────────────────
    sa_initial_temp_scale:   float = 0.1    # T_init = max(1.0, |Q|_max * scale)
    sa_cooling_rate:         float = 0.999  # slower cooling for more exploration
    max_samples_per_step:    int   = 50     # check up to N SA samples per step

    # ── Diversity for exploration phase ───────────────────────────────────
    diversity_gamma:          float = 10.0   # diversity penalty strength in QUBO
    exploration_perturbation: float = 5.0    # initial random diagonal perturbation
    tabu_list_size:           int   = 20     # max entries in tabu list

    # ── Target property for generation ────────────────────────────────────
    target_prop:  str               = "qed"
    target_range: Tuple[float, float] = (0.76, 0.79)

    # ── Device ────────────────────────────────────────────────────────────
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Debug mode (small data, few epochs) ───────────────────────────────
    debug: bool = False

    def __post_init__(self) -> None:
        if self.debug:
            self.n_train  = 500
            self.n_valid  = 100
            self.n_test   = 100
            self.epochs   = 5
            self.n_explore = 2
            self.n_optimize = 5
            self.sa_reads  = 10
            self.sa_sweeps = 200


# Singleton default configuration
CFG = Config()
