"""
Molecular design pipeline  (Figure 1c of the paper).

From the paper (Methods – Generating molecules):
  "Ten exploration steps are performed for each optimization problem,
   followed by 90 sampling steps to minimize the associated free energy of
   the trained conditional energy-based model."

Algorithm:
  Input:  reference molecule (fixed atom types X), target property y*
  Output: generated molecules satisfying the property target

  Phase 1 – Exploration (n_explore steps):
    • Solve QUBO with no surrogate (uniform β = 0)
    • Collect (bond_vector, CRBM_free_energy) pairs for surrogate fitting

  Phase 2 – Surrogate-guided optimization (n_optimize steps):
    • Fit linear surrogate: F̂(A) ≈ β^T A_vec   (sklearn LinearRegression)
    • Build QUBO with current β as objective coefficients
    • Solve QUBO → new adjacency matrix A*
    • Evaluate A* with trained CRBM and RDKit
    • Add to dataset; refit surrogate

  Validity check:
    Every generated adjacency is converted to canonical SMILES via RDKit.
    Invalid / sanitisation-failing molecules are discarded automatically.
"""
from __future__ import annotations

import random
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

from qc_mol_design.config import CFG, ATOM_VALENCY, PROP_BINS, N_BINS_PER_PROP
from qc_mol_design.data.featurizer import graph_from_atoms_adj, mol_to_graph
from qc_mol_design.data.loader import discretize
from qc_mol_design.models.crbm import CRBM
from qc_mol_design.models.graph_conv import NeuralFingerprint
from qc_mol_design.optimization.qubo import MoleculeQUBO, solve_qubo
from qc_mol_design.optimization.tuning_solver import solve_tuning as solve_qubo
from qc_mol_design.optimization.tuning_solver import prioritize_valid

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Molecule helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_connected(adj: np.ndarray) -> bool:
    """Check if adjacency matrix represents a connected graph via BFS."""
    n = adj.shape[0]
    if n <= 1:
        return True
    visited = {0}
    queue = [0]
    while queue:
        node = queue.pop(0)
        for neighbor in range(n):
            if adj[node, neighbor] and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited) == n


def _count_rings(smiles: str) -> int:
    """Count the number of rings in a molecule from its SMILES."""
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return mol.GetRingInfo().NumRings()
    except Exception:
        return 0


def adj_to_smiles(
    atom_symbols: List[str], adj: np.ndarray,
    formal_charges: Optional[List[int]] = None,
) -> Optional[str]:
    """Convert adjacency matrix + atom list → canonical SMILES, or None.

    Multi-strategy bond type inference from a binary (0/1) adjacency matrix.
    The QUBO only decides whether a bond EXISTS, not its type (single/double/
    aromatic).  We must infer bond types from valence rules.

    Strategy 1 – Manual aromatic detection + RDKit kekulization (BEST):
      Detect rings where all atoms are aromatic-capable (degree <= aromatic
      valency), mark bonds as AROMATIC, set correct H counts using aromatic
      valency, then call SanitizeMol.  For non-ring atoms, greedily upgrade
      double bonds with O-priority (C=O first).  This is the most reliable
      strategy for generating correct aromatic molecules from QUBO output.

    Strategy 0 – Pure RDKit kekulization (fallback):
      Build mol with all SINGLE bonds, set explicit H counts based on
      max_valency - degree, then call Chem.SanitizeMol().  This produces
      saturated (non-aromatic) molecules but may succeed when Strategy 1
      fails due to unusual valence combinations.

    Strategy 2 – Manual aromatic + double bond inference (second fallback):
      Similar to Strategy 1 but with a different implementation path when
      RDKit kekulization fails.

    Strategy 3 – All single bonds (saturated fallback).

    Strategy 4 – Extract largest connected component.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    n = len(atom_symbols)
    charges = list(formal_charges) if formal_charges is not None else [0] * n
    if len(charges) != n:
        raise ValueError("formal_charges length must match atom_symbols")

    def _add_atoms(rw_mol) -> None:
        for atom_index, symbol in enumerate(atom_symbols):
            atom = Chem.Atom(symbol)
            atom.SetFormalCharge(int(charges[atom_index]))
            rw_mol.AddAtom(atom)


    # ── Precompute degree and valence info from adjacency ─────────────────
    degree = [int(adj[i].sum()) for i in range(n)]
    charge_valency = {
        ("N", 1): 4, ("N", -1): 2,
        ("O", 1): 3, ("O", -1): 1,
        ("S", 1): 3, ("S", -1): 1,
    }
    max_v = [charge_valency.get((s, q), ATOM_VALENCY.get(s, 4))
             for s, q in zip(atom_symbols, charges)]

    # Aromatic valency: the effective valency when an atom is in an aromatic
    # ring.  E.g., aromatic C has valency 3 (contributes 1 electron to pi
    # system), so a ring C with degree 3 (2 ring bonds + 1 substituent) and
    # aromatic valency 3 is aromatic-capable.
    _AROMATIC_VALENCY = {"C": 3, "N": 2, "O": 2, "S": 2, "P": 3, "Se": 2}

    # ── Strategy 1: Manual aromatic detection + RDKit kekulization ──────
    # This is the PRIMARY strategy because it correctly handles aromatic
    # rings.  RDKit's SanitizeMol alone (Strategy 0) does NOT detect
    # aromaticity from single bonds + explicit H — it needs bonds to be
    # explicitly set to AROMATIC type or alternating SINGLE/DOUBLE (Kekulé).
    try:
        rw = Chem.RWMol()
        _add_atoms(rw)

        # Add all bonds as SINGLE
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    rw.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
        mol = rw.GetMol()

        # Detect rings
        Chem.FastFindRings(mol)
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()

        # Identify aromatic-capable rings and mark them
        # A ring is aromatic-capable if all atoms have degree <= their
        # aromatic valency.  For C: degree <= 3, for N: degree <= 2, etc.
        aromatic_atoms = set()
        for ring in rings:
            if len(ring) < 5 or len(ring) > 7:
                continue
            all_capable = True
            for aidx in ring:
                atom = mol.GetAtomWithIdx(aidx)
                deg = atom.GetDegree()
                sym = atom.GetSymbol()
                arom_v = _AROMATIC_VALENCY.get(sym, None)
                if arom_v is not None:
                    if deg > arom_v:
                        all_capable = False
                        break
                else:
                    # Non-aromatic-capable atom (F, Cl, etc.) in ring
                    all_capable = False
                    break
            if all_capable:
                ring_set = set(ring)
                # Mark all bonds within the ring as AROMATIC
                for aidx in ring:
                    for nbr in mol.GetAtomWithIdx(aidx).GetNeighbors():
                        if nbr.GetIdx() in ring_set:
                            bond = mol.GetBondBetweenAtoms(aidx, nbr.GetIdx())
                            if bond is not None and not bond.GetIsAromatic():
                                bond.SetBondType(Chem.rdchem.BondType.AROMATIC)
                                bond.SetIsAromatic(True)
                # Mark ring atoms as aromatic
                for aidx in ring:
                    mol.GetAtomWithIdx(aidx).SetIsAromatic(True)
                    mol.GetAtomWithIdx(aidx).SetNoImplicit(False)
                    aromatic_atoms.add(aidx)

        # Set explicit H counts with double-bond inference for non-ring atoms
        n_H_arr = [0] * n
        for aidx in range(n):
            sym = atom_symbols[aidx]
            if aidx in aromatic_atoms and sym in _AROMATIC_VALENCY:
                # Aromatic atoms: H = aromatic_valency - degree
                # E.g., benzene C with degree 2 → H = 3 - 2 = 1
                n_H_arr[aidx] = max(0, _AROMATIC_VALENCY[sym] - degree[aidx])
            else:
                # Non-aromatic atoms: H = max_valency - degree (saturated)
                # Double bonds will be inferred below, reducing H count
                n_H_arr[aidx] = max(0, max_v[aidx] - degree[aidx])

        # Double-bond inference for non-aromatic atoms
        _BOND_PRIORITY = {
            "O": 0, "S": 1, "N": 2, "P": 3, "C": 4, "Si": 5,
            "F": 6, "Cl": 6, "Br": 6, "I": 6,
        }
        upgrade_candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    continue
                # Skip bonds involving aromatic atoms (already handled)
                if i in aromatic_atoms or j in aromatic_atoms:
                    continue
                # Both atoms must have spare valency (n_H > 0)
                if n_H_arr[i] > 0 and n_H_arr[j] > 0:
                    priority = min(
                        _BOND_PRIORITY.get(atom_symbols[i], 6),
                        _BOND_PRIORITY.get(atom_symbols[j], 6),
                    )
                    upgrade_candidates.append((priority, i, j))

        upgrade_candidates.sort()
        double_count = [0] * n
        for _, i, j in upgrade_candidates:
            if n_H_arr[i] <= 0 or n_H_arr[j] <= 0:
                continue
            sym_i, sym_j = atom_symbols[i], atom_symbols[j]
            is_cc = (sym_i == "C" and sym_j == "C")
            # Prevent conjugated chains: C-C double bonds limited to 1 per C
            if is_cc and (double_count[i] >= 1 or double_count[j] >= 1):
                continue
            # Heteroatom bonds: allow up to 2 per atom (enables C=O on
            # ring atoms, amide C=O with C also bonded to N, etc.)
            if not is_cc and (double_count[i] >= 2 or double_count[j] >= 2):
                continue
            # Reduce n_H and set bond type to DOUBLE
            n_H_arr[i] -= 1
            n_H_arr[j] -= 1
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
            double_count[i] += 1
            double_count[j] += 1

        # Apply H counts
        for aidx in range(n):
            atom = mol.GetAtomWithIdx(aidx)
            atom.SetNumExplicitHs(max(0, n_H_arr[aidx]))
            atom.SetNoImplicit(True)

        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        if smi and "." not in smi:
            return smi
    except Exception:
        pass   # Fall through to Strategy 0

    # ── Strategy 0: Pure RDKit kekulization (fallback) ──────────────────
    # Build mol with all SINGLE bonds, set explicit H counts as
    # max_valency - degree, then call SanitizeMol.  This produces saturated
    # molecules (cyclohexane, not benzene) but may succeed when Strategy 1
    # fails due to unusual valence combinations.
    try:
        rw = Chem.RWMol()
        _add_atoms(rw)

        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    rw.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
        mol = rw.GetMol()

        for aidx in range(n):
            atom = mol.GetAtomWithIdx(aidx)
            sym = atom.GetSymbol()
            mv = max_v[aidx]
            deg = atom.GetDegree()
            n_H = max(0, mv - deg)
            atom.SetNumExplicitHs(n_H)
            atom.SetNoImplicit(True)

        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        if smi and "." not in smi:
            return smi
    except Exception:
        pass

    # ── Strategy 2: manual aromatic + double bond inference ────────────
    # Fallback when RDKit kekulization fails (e.g. unusual valence combos).
    try:
        rw = Chem.RWMol()
        _add_atoms(rw)

        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    rw.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
        mol = rw.GetMol()

        Chem.FastFindRings(mol)
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()

        # Phase A: Mark aromatic rings (5-7 membered, all atoms with spare valency)
        modified = False
        aromatic_atoms = set()
        for ring in rings:
            if len(ring) < 5 or len(ring) > 7:
                continue
            all_spare = True
            for aidx in ring:
                atom = mol.GetAtomWithIdx(aidx)
                deg = atom.GetDegree()
                sym = atom.GetSymbol()
                arom_v = _AROMATIC_VALENCY.get(sym, None)
                if arom_v is not None:
                    if deg > arom_v:
                        all_spare = False
                        break
                else:
                    all_spare = False
                    break
            if all_spare:
                ring_set = set(ring)
                for aidx in ring:
                    for nbr in mol.GetAtomWithIdx(aidx).GetNeighbors():
                        if nbr.GetIdx() in ring_set:
                            bond = mol.GetBondBetweenAtoms(aidx, nbr.GetIdx())
                            if bond is not None and not bond.GetIsAromatic():
                                bond.SetBondType(Chem.rdchem.BondType.AROMATIC)
                                bond.SetIsAromatic(True)
                                modified = True
                for aidx in ring:
                    mol.GetAtomWithIdx(aidx).SetIsAromatic(True)
                    mol.GetAtomWithIdx(aidx).SetNoImplicit(False)
                    aromatic_atoms.add(aidx)

        # Phase B: Infer double bonds from valence gaps
        _BOND_PRIORITY = {
            "O": 0, "S": 1, "N": 2, "P": 3, "C": 4, "Si": 5,
            "F": 6, "Cl": 6, "Br": 6, "I": 6,
        }

        valence_gaps = []
        for aidx in range(n):
            if aidx in aromatic_atoms:
                valence_gaps.append(0)
                continue
            atom = mol.GetAtomWithIdx(aidx)
            deg = atom.GetDegree()
            sym = atom.GetSymbol()
            mv = max_v[aidx]
            valence_gaps.append(mv - deg)

        upgrade_candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    continue
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is None or bond.GetIsAromatic():
                    continue
                if i in aromatic_atoms or j in aromatic_atoms:
                    continue
                if valence_gaps[i] > 0 and valence_gaps[j] > 0:
                    priority = min(
                        _BOND_PRIORITY.get(atom_symbols[i], 6),
                        _BOND_PRIORITY.get(atom_symbols[j], 6),
                    )
                    upgrade_candidates.append((priority, i, j))

        upgrade_candidates.sort()

        double_bond_count = [0] * n
        for _, i, j in upgrade_candidates:
            gap_i = valence_gaps[i]
            gap_j = valence_gaps[j]
            if gap_i <= 0 or gap_j <= 0:
                continue
            sym_i = atom_symbols[i]
            sym_j = atom_symbols[j]
            is_cc = (sym_i == "C" and sym_j == "C")
            max_doubles_i = 1 if (is_cc and sym_i == "C") else 2
            max_doubles_j = 1 if (is_cc and sym_j == "C") else 2
            if double_bond_count[i] >= max_doubles_i or \
               double_bond_count[j] >= max_doubles_j:
                continue
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is None:
                continue
            bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
            valence_gaps[i] -= 1
            valence_gaps[j] -= 1
            double_bond_count[i] += 1
            double_bond_count[j] += 1
            modified = True

        if modified:
            try:
                for aidx in range(n):
                    if aidx not in aromatic_atoms and valence_gaps[aidx] >= 0:
                        mol.GetAtomWithIdx(aidx).SetNumExplicitHs(valence_gaps[aidx])
                        mol.GetAtomWithIdx(aidx).SetNoImplicit(True)
                Chem.SanitizeMol(mol)
                smi = Chem.MolToSmiles(mol)
                if smi and "." not in smi:
                    return smi
            except Exception:
                pass
    except Exception:
        pass

    # ── Strategy 3: all single bonds (saturated fallback) ────────────────
    try:
        rw2 = Chem.RWMol()
        _add_atoms(rw2)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    rw2.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
        mol2 = rw2.GetMol()
        Chem.SanitizeMol(mol2)
        smi2 = Chem.MolToSmiles(mol2)
        if smi2 and "." not in smi2:
            return smi2
    except Exception:
        pass

    # ── Strategy 4: Extract largest connected component ──────────────────
    # If all strategies produce disconnected molecules, extract the
    # largest connected component and try converting it.
    try:
        rw3 = Chem.RWMol()
        _add_atoms(rw3)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    rw3.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
        mol3 = rw3.GetMol()
        # Set explicit H counts
        for aidx in range(n):
            atom = mol3.GetAtomWithIdx(aidx)
            sym = atom.GetSymbol()
            mv = max_v[aidx]
            deg = atom.GetDegree()
            n_H = max(0, mv - deg)
            atom.SetNumExplicitHs(n_H)
            atom.SetNoImplicit(True)
        # Find connected components
        fragments = Chem.GetMolFrags(mol3, asMols=True)
        if fragments:
            largest = max(fragments, key=lambda m: m.GetNumAtoms())
            try:
                Chem.SanitizeMol(largest)
                smi3 = Chem.MolToSmiles(largest)
                if smi3:
                    return smi3
            except Exception:
                pass
    except Exception:
        pass

    return None


def compute_rdkit_props(smiles: str) -> Optional[Dict[str, float]]:
    """Return dict of {qed, logp, sas} for a SMILES string, or None."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.QED import qed as rdkit_qed
    from rdkit.Chem.Crippen import MolLogP
    RDLogger.logger().setLevel(RDLogger.ERROR)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            from rdkit.Contrib.SA_Score import sascorer
            sas = float(sascorer.calculateScore(mol))
        except Exception:
            sas = float(mol.GetNumHeavyAtoms()) / 10.0 + 1.0
        return {
            "qed":  float(rdkit_qed(mol)),
            "logp": float(MolLogP(mol)),
            "sas":  sas,
        }
    except Exception:
        return None


def _make_target_y(prop: str, target_range: Tuple[float, float]) -> torch.Tensor:
    """
    Build the concatenated one-hot y* vector for the target property.
    Non-target properties get uniform (average) bins.
    Returns (1, n_vis) FloatTensor.
    """
    mid = (target_range[0] + target_range[1]) / 2.0
    parts = []
    for p in ("qed", "logp", "sas"):
        n = N_BINS_PER_PROP[p]
        if p == prop:
            idx = int(discretize(np.array([mid]), p)[0])
            oh  = np.zeros(n, dtype=np.float32)
            oh[idx] = 1.0
        else:
            oh = np.ones(n, dtype=np.float32) / n  # soft / uniform
        parts.append(oh)
    y = np.concatenate(parts)
    return torch.FloatTensor(y).unsqueeze(0)   # (1, n_vis)


# ─────────────────────────────────────────────────────────────────────────────
# Molecular designer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DesignResult:
    """Complete auditable output for one reference-target trajectory."""

    trajectory: List[Dict]
    samples_raw: List[Dict]
    samples_unique: List[Dict]
    solver_reads: List[Dict]

    @property
    def valid_samples(self) -> List[Dict]:
        return [row for row in self.samples_raw if row.get("valid", False)]




class MolecularDesigner:
    """
    Implements Figure 1c of the paper:
      iterative QUBO-based navigation of chemical space guided by CRBM free energy.
    """

    def __init__(self, gcn: NeuralFingerprint, crbm: CRBM) -> None:
        self.gcn  = gcn.to(CFG.device).eval()
        self.crbm = crbm.to(CFG.device).eval()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _fingerprint(
        self, atom_symbols: List[str], adj: np.ndarray,
        X_ref: Optional[np.ndarray] = None,
        smiles: Optional[str] = None,
    ) -> np.ndarray:
        """Compute GCN fingerprint while keeping reference features fixed.

        Strict generation optimises only adjacency A.  When X_ref is supplied,
        generated SMILES are used solely for RDKit validation/properties and
        never to replace the fixed atom-feature matrix.
        """
        if X_ref is not None:
            X = X_ref
            adj_f = adj.astype(np.float32)
            deg = adj_f.sum(axis=1).astype(np.int32)
        elif smiles is not None:
            graph = mol_to_graph(smiles)
            if graph is not None:
                X, adj_f, deg = graph
            else:
                X, adj_f, deg = graph_from_atoms_adj(atom_symbols, adj)
        else:
            X, adj_f, deg = graph_from_atoms_adj(atom_symbols, adj)
        X_t   = torch.FloatTensor(X[None]).to(CFG.device)
        A_t   = torch.FloatTensor(adj_f[None]).to(CFG.device)
        deg_t = torch.LongTensor(deg[None]).to(CFG.device)
        with torch.no_grad():
            fp = self.gcn(X_t, A_t, deg_t)
        return fp.cpu().numpy()[0]

    def _free_energy(
        self, fp: np.ndarray, prop_or_y, bin_index: Optional[int] = None
    ) -> float:
        """Evaluate target-marginal or legacy fully-conditioned free energy."""
        f_t = torch.FloatTensor(fp[None]).to(CFG.device)
        with torch.no_grad():
            if torch.is_tensor(prop_or_y):
                fe = self.crbm.free_energy(prop_or_y.to(CFG.device), f_t)
            else:
                if bin_index is None:
                    raise ValueError("bin_index is required for target marginalisation")
                fe = self.crbm.target_free_energy(prop_or_y, bin_index, f_t)
        return float(fe.item())

    def _make_ring_initial_states(
        self, qubo: MoleculeQUBO, atom_symbols: List[str], n_states: int = 5
    ) -> List[Dict[int, int]]:
        """Generate initial states with ring structures for SA warm-start.

        Classical SA tends to find tree structures because they are local
        minima of the QUBO.  By seeding SA with ring-containing initial
        states, we help it find ring-containing global minima.

        Two strategies:
          1. Benzene-like 6-membered ring: select 6 C atoms, set ring bonds
          2. Random spanning tree + one extra edge: guarantees exactly 1 ring
        """
        n = len(atom_symbols)
        initial_states = []

        # Strategy 1: Benzene-like 6-membered ring
        c_indices = [i for i, s in enumerate(atom_symbols) if s == "C"]
        if len(c_indices) >= 6:
            for start in range(0, len(c_indices) - 5, max(1, len(c_indices) // 3)):
                ring_atoms = c_indices[start:start + 6]
                state = {k: 0 for k in range(qubo.n_vars)}
                # Set ring bonds
                for a in range(6):
                    i, j = ring_atoms[a], ring_atoms[(a + 1) % 6]
                    idx = qubo.bond_idx(i, j)
                    state[idx] = 1
                initial_states.append(state)
                if len(initial_states) >= n_states // 2:
                    break

        # Strategy 2: 5-membered ring (pyrrole/furan-like)
        ring_capable = [i for i, s in enumerate(atom_symbols)
                        if s in ("C", "N", "O", "S")]
        if len(ring_capable) >= 5 and len(initial_states) < n_states // 2:
            for start in range(0, len(ring_capable) - 4, max(1, len(ring_capable) // 3)):
                ring_atoms = ring_capable[start:start + 5]
                state = {k: 0 for k in range(qubo.n_vars)}
                for a in range(5):
                    i, j = ring_atoms[a], ring_atoms[(a + 1) % 5]
                    idx = qubo.bond_idx(i, j)
                    state[idx] = 1
                initial_states.append(state)
                if len(initial_states) >= n_states // 2:
                    break

        # Strategy 3: Random spanning tree + extra edge (guarantees 1 ring)
        while len(initial_states) < n_states:
            state = {k: 0 for k in range(qubo.n_vars)}
            # Build random spanning tree via Prim's algorithm
            visited = {random.randint(0, n - 1)}
            for _ in range(n - 1):
                candidates = []
                for v in visited:
                    for u in range(n):
                        if u not in visited:
                            candidates.append((v, u))
                if not candidates:
                    break
                v, u = random.choice(candidates)
                visited.add(u)
                idx = qubo.bond_idx(min(v, u), max(v, u))
                state[idx] = 1
            # Add one extra edge to create a ring
            non_tree_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    idx = qubo.bond_idx(i, j)
                    if state[idx] == 0:
                        non_tree_edges.append(idx)
            if non_tree_edges:
                extra = random.choice(non_tree_edges)
                state[extra] = 1
            initial_states.append(state)

        return initial_states[:n_states]

    # ── Main design loop ─────────────────────────────────────────────────────

    def design(
        self,
        ref_smiles: str,
        target_prop: str = None,
        target_range: Tuple[float, float] = None,
        reference_id: str = "",
        qubo_mode: str = None,
        penalty: float = None,
        n_explore: int = None,
        n_optimize: int = None,
        sa_reads: int = None,
        sa_sweeps: int = None,
        exploration_bitstrings: Optional[List[str]] = None,
    ) -> DesignResult:
        """Run an auditable 10+90 reference-target optimization trajectory."""
        from rdkit import Chem

        prop = target_prop or CFG.target_prop
        t_range = target_range or CFG.target_range
        mode = qubo_mode or getattr(CFG, "qubo_mode", "strict")
        if mode not in ("strict", "compensated"):
            raise ValueError(f"unknown QUBO mode: {mode}")
        lam = CFG.valency_penalty if penalty is None else float(penalty)
        condition_seed = int(CFG.seed)
        n_exp = CFG.n_explore if n_explore is None else int(n_explore)
        n_opt = CFG.n_optimize if n_optimize is None else int(n_optimize)
        sa_r = CFG.sa_reads if sa_reads is None else int(sa_reads)
        sa_s = CFG.sa_sweeps if sa_sweeps is None else int(sa_sweeps)

        mol = Chem.MolFromSmiles(ref_smiles)
        if mol is None:
            raise ValueError(f"invalid reference SMILES: {ref_smiles}")
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        formal_charges = [int(atom.GetFormalCharge()) for atom in mol.GetAtoms()]
        charge_valency = {
            ("N", 1): 4, ("N", -1): 2,
            ("O", 1): 3, ("O", -1): 1,
            ("S", 1): 3, ("S", -1): 1,
        }
        max_valencies = [
            charge_valency.get((symbol, charge), ATOM_VALENCY.get(symbol, 4))
            for symbol, charge in zip(atom_symbols, formal_charges)
        ]

        ref_graph = mol_to_graph(ref_smiles)
        if ref_graph is None:
            raise ValueError(f"cannot build reference graph: {ref_smiles}")
        X_ref, _, _ = ref_graph
        qubo = MoleculeQUBO(atom_symbols, max_valencies)
        target_mid = (t_range[0] + t_range[1]) / 2.0
        target_bin = int(discretize(np.array([target_mid]), prop)[0])
        structural_Q = qubo.build(None, 1.0)

        strict = mode == "strict"
        val_def_eps = 0.0 if strict else CFG.valency_deficit_penalty
        ring_inc = 0.0 if strict else CFG.ring_incentive
        conn_inc = 0.0 if strict else CFG.connectivity_incentive
        div_gamma = 0.0 if strict else CFG.diversity_gamma
        base_perturb = 0.0 if strict else CFG.exploration_perturbation
        ring_states = (
            None if strict
            else self._make_ring_initial_states(qubo, atom_symbols, n_states=5)
        )

        bond_X: List[np.ndarray] = []
        fe_y: List[float] = []
        previous_solutions: List[np.ndarray] = []
        trajectory: List[Dict] = []
        raw: List[Dict] = []
        solver_reads: List[Dict] = []
        seen_selected_smiles: set = set()
        seen_solver_smiles: set = set()
        best_fe = float("inf")
        surrogate = LinearRegression(fit_intercept=False)

        print(f"\n[DESIGN] {reference_id or 'unlabelled'}: {ref_smiles}")
        print(
            f"[DESIGN] atoms={len(atom_symbols)} vars={qubo.n_vars} "
            f"target={prop}{t_range} mode={mode}"
        )

        def effective_lambda(
            beta: Optional[np.ndarray],
        ) -> Tuple[float, float, float]:
            beta_l1 = float(np.abs(beta).sum()) if beta is not None else 0.0
            beta_linf = float(np.abs(beta).max()) if beta is not None else 0.0
            lambda_mode = getattr(CFG, "lambda_mode", "fixed")
            if strict and lambda_mode not in ("fixed", "cim_precision"):
                raise ValueError(
                    "strict reproduction requires the paper's fixed lambda"
                )
            if lambda_mode == "cim_precision":
                if not strict:
                    raise ValueError("CIM precision mode requires the paper structural QUBO")
                ratio = CFG.cim_penalty_ratio
                if not np.isfinite(ratio) or ratio <= 1:
                    raise ValueError("cim_penalty_ratio must exceed one")
                return (ratio * beta_linf if beta_linf else 1.0), beta_l1, beta_linf
            if lambda_mode == "fixed":
                if beta_linf >= lam:
                    raise ValueError(
                        f"fixed lambda {lam} does not exceed "
                        f"||beta||_inf={beta_linf}"
                    )
                return lam, beta_l1, beta_linf
            if lambda_mode != "dynamic_l1":
                raise ValueError(f"unknown lambda mode: {lambda_mode}")
            return (
                max(lam, CFG.penalty_scale * beta_l1), beta_l1, beta_linf
            )

        def decode_forced(bitstring: str) -> Dict[int, int]:
            if len(bitstring) != qubo.n_vars or set(bitstring) - {"0", "1"}:
                raise ValueError("invalid exploration bitstring")
            return {i: int(bit) for i, bit in enumerate(bitstring)}

        def run_step(
            beta: Optional[np.ndarray],
            phase: str,
            phase_step: int,
            forced_bitstring: Optional[str] = None,
        ) -> None:
            nonlocal best_fe
            global_step = len(raw)
            solver_seed = condition_seed + global_step
            eff_lam, beta_l1, beta_linf = effective_lambda(beta)
            perturbation = (
                base_perturb * (1.0 - phase_step / max(n_exp, 1))
                if phase == "explore" else 0.0
            )
            structural_multiplier = 1.0 if phase == "explore" else eff_lam
            Q = qubo.build(
                beta,
                structural_multiplier,
                previous_solutions=previous_solutions if phase == "explore" else None,
                diversity_gamma=div_gamma if phase == "explore" else 0.0,
                perturbation=perturbation,
                valency_deficit_eps=val_def_eps,
                ring_incentive=ring_inc,
                connectivity_incentive=conn_inc,
            )
            solve_started = time.perf_counter()
            prepared = None
            if getattr(CFG, "lambda_mode", "fixed") == "cim_precision":
                from qc_mol_design.optimization.cim_precision import prepare_balanced_qubo
                Q, prepared = prepare_balanced_qubo(
                    structural_Q, beta if phase == "opt" else None,
                    penalty_ratio=CFG.cim_penalty_ratio,
                )
                eff_lam = prepared["metadata"]["effective_lambda"]
                structural_multiplier = eff_lam
            previous_seed = CFG.seed
            CFG.seed = solver_seed
            cim_call = {}
            if os.environ.get("QC_QUBO_SOLVER") == "sqa":
                os.environ["QC_SQA_STEP"] = str(global_step)
            validity_cache = {}
            def valid_solver_sample(sample):
                bits = tuple(sample[i] for i in range(qubo.n_vars))
                if bits not in validity_cache:
                    adjacency = qubo.decode_sample(sample)
                    smiles = (adj_to_smiles(atom_symbols, adjacency, formal_charges=formal_charges)
                              if _is_connected(adjacency) else None)
                    validity_cache[bits] = bool(smiles and compute_rdkit_props(smiles) is not None)
                return validity_cache[bits]
            try:
                if forced_bitstring is None:
                    if os.environ.get("QC_QUBO_SOLVER") == "kaiwu_cim":
                        os.environ["KAIWU_CIM_TASK_CONTEXT"] = (
                            f"{prop}_{t_range[0]}_{t_range[1]}_{reference_id}_{global_step}"
                        )
                    samples = solve_qubo(
                        Q, sa_r, sa_s,
                        initial_states=ring_states if not strict else None,
                        **({"valid_sample": valid_solver_sample}
                           if int(os.environ.get("QC_MAX_FEASIBILITY_BATCHES", "1")) > 1 else {}),
                        **({"cim_prepared": prepared} if prepared is not None
                           and os.environ.get("QC_QUBO_SOLVER") in ("kaiwu_cim", "kaiwu", "sqa") else {}),
                    )
                    if os.environ.get("QC_QUBO_SOLVER") == "sqa":
                        from qc_mol_design.optimization.sqa_solver import LAST_CALL
                        cim_call = dict(LAST_CALL)
                    if os.environ.get("QC_QUBO_SOLVER") == "kaiwu_cim":
                        from qc_mol_design.optimization.cim_hardware import LAST_CALL
                        if not LAST_CALL.get("used_cim"):
                            raise RuntimeError("Missing CIM hardware provenance")
                        cim_call = {
                            "used_cim": True,
                            "cim_task_name": LAST_CALL["task_name"],
                            "cim_task_names": LAST_CALL.get("task_names", [LAST_CALL["task_name"]]),
                            "cim_hardware_calls": LAST_CALL.get("hardware_calls", 1),
                            "cim_project_no": LAST_CALL["project_no"],
                            "cim_cache_reused": LAST_CALL["cache_reused"],
                            "cim_returned_reads": LAST_CALL["returned_reads"],
                            "cim_quantization_relative_error": LAST_CALL["quantization_relative_frobenius_error"],
                            "cim_quantization_scale": LAST_CALL["scale"],
                            "cim_quantization_max_abs_error": LAST_CALL["quantization_max_abs_error"],
                        }
                        for field in ("solver_repeats", "feasibility_batches", "history_replayed", "sampling_regime"):
                            if field in LAST_CALL:
                                cim_call[field] = LAST_CALL[field]
                        for field in ("precision_policy", "guidance_integer_linf", "guidance_relative_error",
                                      "nonzero_guidance", "structural_integer_exact", "penalty_ratio"):
                            if field in LAST_CALL:
                                cim_call["cim_" + field] = LAST_CALL[field]
                else:
                    samples = [decode_forced(forced_bitstring)]
            finally:
                CFG.seed = previous_seed
            solve_seconds = time.perf_counter() - solve_started
            if not samples:
                raise RuntimeError("QUBO solver returned no samples")

            evaluated = []
            for sample in samples:
                adj = qubo.decode_sample(sample)
                bvec = qubo.bond_vector(adj)
                q_energy = float(bvec @ Q @ bvec)
                evaluated.append((q_energy, sample, adj, bvec))
            evaluated.sort(key=lambda item: item[0])
            selection_policy = os.environ.get("QC_CANDIDATE_SELECTION", "energy")
            def candidate_valid(item):
                adjacency = item[2]
                if not _is_connected(adjacency):
                    return False
                smiles = adj_to_smiles(atom_symbols, adjacency, formal_charges=formal_charges)
                return bool(smiles and compute_rdkit_props(smiles) is not None)
            evaluated = prioritize_valid(evaluated, candidate_valid, selection_policy)
            selected_bvec = evaluated[0][3]
            selected_adj = evaluated[0][2]
            selected_connected = _is_connected(selected_adj)
            selected_smiles_for_model = (
                adj_to_smiles(
                    atom_symbols, selected_adj, formal_charges=formal_charges
                )
                if selected_connected else None
            )
            selected_props_for_model = (
                compute_rdkit_props(selected_smiles_for_model)
                if selected_smiles_for_model else None
            )
            selected_valid = selected_props_for_model is not None

            # The paper explicitly discards invalid molecules.  Keep the
            # attempted read in samples_raw for validity auditing, but do not
            # evaluate its free energy or append it to regression dataset D.
            free_energy = None
            if selected_valid:
                fp = self._fingerprint(atom_symbols, selected_adj, X_ref=X_ref)
                free_energy = self._free_energy(fp, prop, target_bin)
                bond_X.append(selected_bvec.copy())
                fe_y.append(free_energy)
                previous_solutions.append(selected_bvec.copy())
                if len(previous_solutions) > CFG.tabu_list_size:
                    previous_solutions.pop(0)
                best_fe = min(best_fe, free_energy)

            selected_raw = None
            for rank, (q_energy, _sample, adj, bvec) in enumerate(evaluated):
                connected = _is_connected(adj)
                smiles = (
                    adj_to_smiles(atom_symbols, adj, formal_charges=formal_charges)
                    if connected else None
                )
                props = compute_rdkit_props(smiles) if smiles else None
                valid = props is not None
                solver_duplicate = bool(valid and smiles in seen_solver_smiles)
                if valid:
                    seen_solver_smiles.add(smiles)
                target_hit = bool(
                    valid and t_range[0] <= props[prop] < t_range[1]
                )
                if not connected:
                    invalid_reason = "disconnected"
                elif smiles is None:
                    invalid_reason = "conversion_failed"
                elif props is None:
                    invalid_reason = "property_failed"
                else:
                    invalid_reason = ""
                structural_energy = float(bvec @ structural_Q @ bvec)
                surrogate_energy = (
                    float(beta @ bvec) if beta is not None else None
                )
                row = {
                    **cim_call,
                    "reference_id": reference_id,
                    "reference_smiles": ref_smiles,
                    "n_heavy_atoms": len(atom_symbols),
                    "n_qubo_vars": qubo.n_vars,
                    "target_name": prop,
                    "target_prop": prop,
                    "target_low": t_range[0],
                    "target_high": t_range[1],
                    "target_lo": t_range[0],
                    "target_hi": t_range[1],
                    "step": global_step,
                    "phase": phase,
                    "phase_step": phase_step,
                    "global_step": global_step,
                    "sample_rank": rank,
                    "candidate_selection_policy": selection_policy,
                    "solver_repeats": cim_call.get("solver_repeats", int(os.environ.get("QC_SOLVER_REPEATS", "1"))),
                    "solver_returned_reads": len(samples),
                    "feasibility_batches": cim_call.get("feasibility_batches", len(samples) // (sa_r * int(os.environ.get("QC_SOLVER_REPEATS", "1")))),
                    "selected_for_trajectory": rank == 0,
                    "A_bitstring": "".join(str(int(v)) for v in bvec),
                    "Qc": structural_energy,
                    "qubo_energy": q_energy,
                    "structural_qubo_energy": structural_energy,
                    "free_energy": free_energy if rank == 0 else None,
                    "surrogate_energy": surrogate_energy,
                    "surrogate_prediction": surrogate_energy,
                    "lambda": eff_lam if phase == "opt" else None,
                    "structural_multiplier": structural_multiplier,
                    "beta_l1": beta_l1,
                    "beta_linf": beta_linf,
                    "lambda_minus_beta_linf": (
                        eff_lam - beta_linf if phase == "opt" else None
                    ),
                    "lambda_to_beta_linf_ratio": (
                        eff_lam / beta_linf
                        if phase == "opt" and beta_linf > 0 else None
                    ),
                    "qubo_mode": mode,
                    "lambda_mode": getattr(CFG, "lambda_mode", "fixed"),
                    "solve_seconds": solve_seconds,
                    "solver_seed": solver_seed,
                    "connected": connected,
                    "valid": valid,
                    "invalid_reason": invalid_reason,
                    "canonical_smiles": smiles or "",
                    "smiles": smiles or "",
                    "duplicate_solver_read": solver_duplicate,
                    "duplicate": None,
                    "target_hit": target_hit,
                    "in_range": target_hit,
                    "qed": props["qed"] if valid else None,
                    "logp": props["logp"] if valid else None,
                    "sas": props["sas"] if valid else None,
                }
                solver_reads.append(row)
                if rank == 0:
                    selected_raw = row

            assert selected_raw is not None
            if os.environ.get("QC_HISTORY_PREFIX_FILE"):
                from .history_prefix import validate_selected
                validate_selected(selected_raw)
            selected_smiles = selected_raw["smiles"]
            selected_duplicate = bool(
                selected_raw["valid"]
                and selected_smiles in seen_selected_smiles
            )
            if selected_raw["valid"]:
                seen_selected_smiles.add(selected_smiles)
            selected_raw["duplicate"] = selected_duplicate
            raw.append(selected_raw)
            if cim_call:
                import json
                from pathlib import Path
                step_log = Path(os.environ.get("QC_SQA_CACHE_DIR") or os.environ["KAIWU_CIM_CACHE_DIR"]) / "selected_steps.jsonl"
                step_log.parent.mkdir(parents=True, exist_ok=True)
                with step_log.open("a") as handle:
                    handle.write(json.dumps(selected_raw) + "\n")
            if selected_raw["valid"]:
                trajectory.append({
                    **selected_raw,
                    "free_energy": free_energy,
                    "best_free_energy_so_far": best_fe,
                })

        for step in range(n_exp):
            forced = (
                exploration_bitstrings[step]
                if exploration_bitstrings is not None else None
            )
            run_step(None, "explore", step, forced_bitstring=forced)

        for step in range(n_opt):
            beta = None
            if len(bond_X) >= 2:
                surrogate.fit(np.stack(bond_X), np.asarray(fe_y))
                beta = np.asarray(surrogate.coef_, dtype=np.float64)
            run_step(beta, "opt", step)

        if trajectory:
            best_curve = np.asarray(
                [row["best_free_energy_so_far"] for row in trajectory],
                dtype=np.float64,
            )
            time_axis = np.linspace(0.0, 1.0, len(best_curve))
            start_fe = float(best_curve[0])
            final_fe = float(best_curve[-1])
            if start_fe > final_fe:
                normalized_regret = (best_curve - final_fe) / (start_fe - final_fe)
            else:
                normalized_regret = np.zeros_like(best_curve)
            integrate = getattr(np, "trapezoid", np.trapz)
            auc_best_fe = float(integrate(best_curve, time_axis))
            auc_normalized_regret = float(integrate(normalized_regret, time_axis))
            for row, regret in zip(trajectory, normalized_regret):
                row["normalized_best_fe_regret"] = float(regret)
                row["best_free_energy"] = final_fe
                row["auc_best_free_energy_normalized_time"] = auc_best_fe
                row["auc_normalized_best_fe_regret"] = auc_normalized_regret

        unique = [
            dict(row) for row in raw
            if row["valid"] and not row["duplicate"]
        ]
        print(
            f"[DESIGN] completed trajectory={len(trajectory)} "
            f"raw={len(raw)} solver_reads={len(solver_reads)} "
            f"valid={sum(r['valid'] for r in raw)} "
            f"unique={len(unique)}"
        )
        return DesignResult(
            trajectory=trajectory,
            samples_raw=raw,
            samples_unique=unique,
            solver_reads=solver_reads,
        )

    def _design_legacy(
        self,
        ref_smiles:   str,
        target_prop:  str   = None,
        target_range: Tuple[float, float] = None,
        penalty:      float = None,
        n_explore:    int   = None,
        n_optimize:   int   = None,
        sa_reads:     int   = None,
        sa_sweeps:    int   = None,
    ) -> List[Dict]:
        """
        Run the full design pipeline for a reference molecule.

        Returns a list of generated molecule dicts (smiles, properties, phase).
        """
        from rdkit import Chem

        prop         = target_prop   or CFG.target_prop
        t_range      = target_range  or CFG.target_range
        lam          = penalty       or CFG.valency_penalty
        n_exp        = n_explore     or CFG.n_explore
        n_opt        = n_optimize    or CFG.n_optimize
        sa_r         = sa_reads      or CFG.sa_reads
        sa_s         = sa_sweeps     or CFG.sa_sweeps

        # ── Parse reference molecule and compute reference atom features ──
        mol = Chem.MolFromSmiles(ref_smiles)
        if mol is None:
            raise ValueError(f"Invalid reference SMILES: {ref_smiles}")

        atom_symbols  = [a.GetSymbol() for a in mol.GetAtoms()]
        max_valencies = [ATOM_VALENCY.get(s, 4) for s in atom_symbols]

        # Compute reference atom features from the FULL reference molecule
        ref_graph = mol_to_graph(ref_smiles)
        if ref_graph is None:
            raise ValueError(f"Cannot build graph for reference: {ref_smiles}")
        X_ref, _, _ = ref_graph   # (n_atoms, atom_feat_dim) — cached

        print(f"\n[DESIGN] Reference: {ref_smiles}")
        print(f"[DESIGN] Atoms ({len(atom_symbols)}): {atom_symbols}")
        print(f"[DESIGN] Target: {prop} ∈ {t_range}")

        # ── Pre-build QUBO structure ─────────────────────────────────────
        qubo    = MoleculeQUBO(atom_symbols, max_valencies)
        y_star  = _make_target_y(prop, t_range)
        surrogate = LinearRegression(fit_intercept=False)
        bond_X: List[np.ndarray] = []
        fe_y:   List[float]      = []
        generated: List[Dict]    = []
        seen_bvec: set = set()
        seen_smi_global: set = set()

        # Tabu list
        tabu_bvecs: List[np.ndarray] = []
        tabu_max   = getattr(CFG, 'tabu_list_size', 20)

        # Ring incentive (may be adapted during exploration)
        ring_inc = getattr(CFG, 'ring_incentive', 0.0)
        val_def_eps = getattr(CFG, 'valency_deficit_penalty', 0.0)
        conn_inc = getattr(CFG, 'connectivity_incentive', 0.0)

        # CIM mode: increase valency deficit eps to force full-valent solutions
        # (penalty=100 + eps=5.0 guarantees full-valent while fitting 14-bit)
        cim_mode = os.environ.get("QC_QUBO_SOLVER", "").lower() == "kaiwu_cim"
        if cim_mode and val_def_eps < 5.0:
            val_def_eps = 5.0

        # Generate ring-seeded initial states for SA warm-start
        ring_init_states = self._make_ring_initial_states(qubo, atom_symbols, n_states=5)

        # ── Inner: solve QUBO, evaluate, collect ─────────────────────────
        max_samples = getattr(CFG, 'max_samples_per_step', 50)

        def _step(
            beta: Optional[np.ndarray],
            phase: str,
            previous_solutions: Optional[List[np.ndarray]] = None,
            diversity_gamma: float = 0.0,
            perturbation: float = 0.0,
            use_ring_init: bool = False,
        ) -> int:
            # Adaptive penalty: paper specifies λ >> ‖β‖₁
            # CIM mode: use fixed penalty + higher eps for full-valent solutions
            # (14-bit precision doesn't need the huge adaptive penalty that
            # classical SA requires; quantum tunnelling compensates).
            cim_mode = os.environ.get("QC_QUBO_SOLVER", "").lower() == "kaiwu_cim"
            if cim_mode:
                # Fixed penalty capped at 100 to ensure 14-bit fit
                eff_lam = min(lam, 100.0)
            elif beta is not None and len(beta) > 0:
                beta_l1 = float(np.abs(beta).sum())
                eff_lam = max(lam, CFG.penalty_scale * beta_l1)
            else:
                eff_lam = lam

            Q = qubo.build(
                beta, eff_lam,
                previous_solutions=previous_solutions,
                diversity_gamma=diversity_gamma,
                perturbation=perturbation,
                valency_deficit_eps=val_def_eps,
                ring_incentive=ring_inc,
                connectivity_incentive=conn_inc,
            )

            # Choose initial states: ring-seeded or default
            init = ring_init_states if use_ring_init else None
            samples = solve_qubo(Q, sa_r, sa_s, initial_states=init)
            valid   = 0
            for sample in samples[:max_samples]:
                adj  = qubo.decode_sample(sample)

                # Early connectivity check — skip disconnected graphs
                if not _is_connected(adj):
                    continue

                smi  = adj_to_smiles(atom_symbols, adj)
                if smi is None or smi in seen_smi_global:
                    continue
                # Skip multi-fragment molecules
                if "." in smi:
                    continue
                seen_smi_global.add(smi)

                fp   = self._fingerprint(atom_symbols, adj, X_ref=X_ref, smiles=smi)
                fe   = self._free_energy(fp, y_star)
                bvec = qubo.bond_vector(adj)

                # Only append unique bond vectors to the surrogate dataset
                bvec_key = bvec.tobytes()
                if bvec_key not in seen_bvec:
                    seen_bvec.add(bvec_key)
                    bond_X.append(bvec)
                    fe_y.append(fe)

                    # Update tabu list with new unique bond vector
                    tabu_bvecs.append(bvec.copy())
                    if len(tabu_bvecs) > tabu_max:
                        tabu_bvecs.pop(0)

                props = compute_rdkit_props(smi)
                if props:
                    in_range = t_range[0] <= props[prop] <= t_range[1]
                    generated.append({
                        "smiles":   smi,
                        "phase":    phase,
                        "in_range": in_range,
                        **props,
                    })
                    valid += 1
                if valid >= 5:
                    break
            return valid

        # ── Phase 1: Exploration with diversity mechanisms ────────────────
        div_gamma = getattr(CFG, 'diversity_gamma', 0.0)
        base_perturb = getattr(CFG, 'exploration_perturbation', 0.0)

        print("[PHASE 1] Exploration")
        for step in range(n_exp):
            # Alternate between ring-seeded and random starts
            use_ring = (step % 2 == 0)
            perturb = base_perturb * (1.0 - step / max(n_exp, 1))

            n_v = _step(
                None, f"explore_{step}",
                previous_solutions=tabu_bvecs if div_gamma > 0 else None,
                diversity_gamma=div_gamma,
                perturbation=perturb,
                use_ring_init=use_ring,
            )

            # Retry with increased perturbation if no new molecules found
            if n_v == 0:
                perturb *= 2.0
                n_v = _step(
                    None, f"explore_{step}_retry",
                    previous_solutions=tabu_bvecs if div_gamma > 0 else None,
                    diversity_gamma=div_gamma,
                    perturbation=perturb,
                    use_ring_init=True,  # Force ring init on retry
                )

            print(f"  Step {step+1:2d}/{n_exp} | found {n_v} valid | "
                  f"dataset: {len(bond_X)} | tabu: {len(tabu_bvecs)}")

        # ── Adaptive ring incentive ──────────────────────────────────────
        if getattr(CFG, 'ring_incentive_adaptive', False):
            ring_count = sum(1 for m in generated if _count_rings(m["smiles"]) > 0)
            if ring_count == 0 and ring_inc > 0:
                ring_inc *= 2.0
                print(f"[DESIGN] No rings found in exploration. "
                      f"Increasing ring_incentive to {ring_inc:.1f}")
            elif ring_count > 0:
                print(f"[DESIGN] Ring-containing molecules: {ring_count}/{len(generated)}")

        # ── Phase 2: Surrogate-guided optimisation ────────────────────────
        print("[PHASE 2] Surrogate-guided optimisation")
        for step in range(n_opt):
            if len(bond_X) >= 2:
                surrogate.fit(bond_X, fe_y)
                beta = surrogate.coef_
            else:
                beta = None

            # Alternate ring-seeded starts in optimization too
            use_ring = (step % 3 == 0)
            n_v = _step(beta, f"opt_{step}", use_ring_init=use_ring)
            beta_str = f"{np.mean(beta):.4f}" if beta is not None else "N/A"
            print(f"  Step {step+1:2d}/{n_opt} | beta_avg={beta_str} | "
                  f"found {n_v} valid | dataset: {len(bond_X)}")

        # ── Summary ───────────────────────────────────────────────────────
        in_range = [m for m in generated if m["in_range"]]
        total    = len(generated)
        ring_total = sum(1 for m in generated if _count_rings(m["smiles"]) > 0)
        print(f"\n[DESIGN] Done. Generated={total}, "
              f"In-range={len(in_range)} ({100*len(in_range)/max(total,1):.1f}%), "
              f"Rings={ring_total}/{total}")
        if in_range:
            for key in ("qed", "logp", "sas"):
                vals = [m[key] for m in in_range]
                print(f"  {key.upper()}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

        return generated
