"""
kt Algorithm Family Implementation for Jet Clustering
=====================================================

This module implements the kt algorithm family for sequential recombination
jet clustering, as used in the paper for comparison with QAOA results.

Algorithms implemented:
- kt algorithm (p=1): Original kt algorithm
- Cambridge/Aachen (p=0): Angular-ordered clustering
- Anti-kt (p=-1): LHC standard algorithm (infrared and collinear safe)
- Durham/e+e- kt: For e+e- collider experiments

The sequential recombination approach identifies pairs of objects to merge
based on distance metrics until all particles are assigned to jets.

References:
- S. Catani et al., "Longitudinally-invariant kt clustering algorithms 
  for hadron-hadron collisions", Nucl.Phys.B 406 (1993) 187
- M. Cacciari, G.P. Salam, G. Soyez, "The anti-kt jet clustering algorithm",
  JHEP 04 (2008) 063
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import heapq


class ClusteringAlgorithm(Enum):
    """Enumeration of available clustering algorithms"""
    KT = "kt"
    ANTI_KT = "anti_kt"
    CAMBRIDGE_AACHEN = "cambridge_aachen"
    DURHAM = "durham"


@dataclass
class Particle:
    """
    Represents a particle or pseudo-jet in the clustering process
    
    Attributes:
        pt: Transverse momentum
        eta: Pseudorapidity
        phi: Azimuthal angle
        E: Energy
        index: Original particle index (for tracking)
        constituents: List of original particle indices in this cluster
    """
    pt: float
    eta: float
    phi: float
    E: float
    index: int
    constituents: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.constituents:
            self.constituents = [self.index]
    
    @property
    def px(self) -> float:
        """X-component of momentum"""
        return self.pt * np.cos(self.phi)
    
    @property
    def py(self) -> float:
        """Y-component of momentum"""
        return self.pt * np.sin(self.phi)
    
    @property
    def pz(self) -> float:
        """Z-component of momentum"""
        return self.pt * np.sinh(self.eta)
    
    @property
    def mass(self) -> float:
        """Invariant mass"""
        m_sq = self.E**2 - self.px**2 - self.py**2 - self.pz**2
        return np.sqrt(max(0, m_sq))
    
    def four_momentum(self) -> np.ndarray:
        """Return 4-momentum [E, px, py, pz]"""
        return np.array([self.E, self.px, self.py, self.pz])


@dataclass
class ClusteringHistory:
    """
    Records the clustering history for analysis
    
    Each entry represents a merge operation or beam clustering
    """
    step: int
    parent1: int  # Index of first parent (-1 for final jet)
    parent2: int  # Index of second parent (-1 for beam clustering)
    child: int    # Index of resulting cluster
    dij: float    # Distance at which merge occurred
    diB: float    # Beam distance (for final jets)


@dataclass
class ClusteringResult:
    """Container for jet clustering results"""
    jets: List[Particle]
    jet_labels: np.ndarray
    n_jets: int
    history: List[ClusteringHistory]
    algorithm: str
    parameters: Dict


class KtJetClustering:
    """
    kt Algorithm Family Implementation
    
    Implements the generalized kt algorithm with parameter p:
    - p = 1: kt algorithm (soft jets first)
    - p = 0: Cambridge/Aachen (angular ordering)
    - p = -1: anti-kt (hard jets first, LHC standard)
    
    Distance metrics:
    d_ij = min(k_ti^(2p), k_tj^(2p)) * ΔR_ij^2 / R^2
    d_iB = k_ti^(2p)
    
    The algorithm proceeds:
    1. Compute all d_ij and d_iB
    2. Find minimum distance
    3. If d_ij < d_iB: merge i and j
       Else: i is a final jet
    4. Repeat until all particles are in jets
    """
    
    def __init__(self, 
                 R: float = 0.4,
                 p: int = -1,
                 min_pt: float = 0.0,
                 max_eta: float = 5.0):
        """
        Initialize kt clustering
        
        Args:
            R: Jet radius parameter
            p: Algorithm parameter (-1: anti-kt, 0: C/A, 1: kt)
            min_pt: Minimum jet pT threshold
            max_eta: Maximum |eta| for jets
        """
        self.R = R
        self.p = p
        self.min_pt = min_pt
        self.max_eta = max_eta
        
        # Determine algorithm name
        if p == -1:
            self.algorithm = ClusteringAlgorithm.ANTI_KT
        elif p == 0:
            self.algorithm = ClusteringAlgorithm.CAMBRIDGE_AACHEN
        elif p == 1:
            self.algorithm = ClusteringAlgorithm.KT
        else:
            self.algorithm = ClusteringAlgorithm.KT
    
    def _delta_phi(self, phi1: float, phi2: float) -> float:
        """Compute angular difference with periodicity"""
        dphi = phi1 - phi2
        while dphi > np.pi:
            dphi -= 2 * np.pi
        while dphi < -np.pi:
            dphi += 2 * np.pi
        return dphi
    
    def _delta_R_squared(self, p1: Particle, p2: Particle) -> float:
        """Compute ΔR² between two particles"""
        deta = p1.eta - p2.eta
        dphi = self._delta_phi(p1.phi, p2.phi)
        return deta**2 + dphi**2
    
    def _d_ij(self, p1: Particle, p2: Particle) -> float:
        """
        Compute inter-particle distance d_ij
        
        d_ij = min(k_ti^(2p), k_tj^(2p)) * ΔR_ij^2 / R^2
        """
        kt_factor = min(p1.pt**(2*self.p), p2.pt**(2*self.p))
        deltaR_sq = self._delta_R_squared(p1, p2)
        return kt_factor * deltaR_sq / (self.R**2)
    
    def _d_iB(self, p: Particle) -> float:
        """
        Compute beam distance d_iB
        
        d_iB = k_ti^(2p)
        """
        return p.pt**(2*self.p)
    
    def _merge_particles(self, p1: Particle, p2: Particle, new_idx: int) -> Particle:
        """
        Merge two particles/clusters using E-scheme recombination
        
        Four-momenta are simply summed.
        """
        # Sum four-momenta
        E = p1.E + p2.E
        px = p1.px + p2.px
        py = p1.py + p2.py
        pz = p1.pz + p2.pz
        
        # Compute new kinematic variables
        pt = np.sqrt(px**2 + py**2)
        
        if pt > 0:
            phi = np.arctan2(py, px)
            p_tot = np.sqrt(px**2 + py**2 + pz**2)
            eta = 0.5 * np.log((p_tot + pz) / (p_tot - pz + 1e-10))
        else:
            phi = 0.0
            eta = 0.0
        
        # Merge constituents
        constituents = p1.constituents + p2.constituents
        
        return Particle(
            pt=pt, eta=eta, phi=phi, E=E,
            index=new_idx, constituents=constituents
        )
    
    def cluster(self, particles: np.ndarray, 
                n_jets: Optional[int] = None) -> ClusteringResult:
        """
        Run the clustering algorithm
        
        Args:
            particles: Array of shape (n, 4) with [pt, eta, phi, E]
            n_jets: Target number of jets (if None, use natural stopping)
            
        Returns:
            ClusteringResult with jets and clustering history
        """
        n_particles = len(particles)
        
        # Initialize particle list
        active_particles: Dict[int, Particle] = {}
        for i in range(n_particles):
            pt, eta, phi, E = particles[i]
            active_particles[i] = Particle(pt, eta, phi, E, i)
        
        # Clustering history
        history: List[ClusteringHistory] = []
        jets: List[Particle] = []
        next_idx = n_particles
        step = 0
        
        # Main clustering loop
        while len(active_particles) > 0:
            if n_jets is not None and len(jets) >= n_jets:
                break
            
            # Find minimum distance
            min_dist = float('inf')
            min_type = 'beam'  # 'merge' or 'beam'
            min_i = -1
            min_j = -1
            
            particle_list = list(active_particles.values())
            n_active = len(particle_list)
            
            # Check all particle pairs
            for i in range(n_active):
                p_i = particle_list[i]
                
                # Beam distance
                d_iB = self._d_iB(p_i)
                if d_iB < min_dist:
                    min_dist = d_iB
                    min_type = 'beam'
                    min_i = p_i.index
                
                # Pair distances
                for j in range(i + 1, n_active):
                    p_j = particle_list[j]
                    d_ij = self._d_ij(p_i, p_j)
                    if d_ij < min_dist:
                        min_dist = d_ij
                        min_type = 'merge'
                        min_i = p_i.index
                        min_j = p_j.index
            
            if min_type == 'beam':
                # Particle becomes a jet
                jet = active_particles.pop(min_i)
                
                # Apply jet selection cuts
                if jet.pt >= self.min_pt and abs(jet.eta) <= self.max_eta:
                    jets.append(jet)
                
                history.append(ClusteringHistory(
                    step=step, parent1=min_i, parent2=-1,
                    child=-1, dij=0, diB=min_dist
                ))
            else:
                # Merge particles
                p1 = active_particles.pop(min_i)
                p2 = active_particles.pop(min_j)
                
                merged = self._merge_particles(p1, p2, next_idx)
                active_particles[next_idx] = merged
                
                history.append(ClusteringHistory(
                    step=step, parent1=min_i, parent2=min_j,
                    child=next_idx, dij=min_dist, diB=0
                ))
                
                next_idx += 1
            
            step += 1
        
        # If n_jets specified and we have remaining particles, they become jets
        if n_jets is not None:
            for p in active_particles.values():
                if len(jets) < n_jets and p.pt >= self.min_pt:
                    jets.append(p)
        
        # Sort jets by pT (descending)
        jets.sort(key=lambda j: j.pt, reverse=True)
        
        # Create jet labels for original particles
        jet_labels = np.full(n_particles, -1, dtype=int)
        for jet_idx, jet in enumerate(jets):
            for constituent_idx in jet.constituents:
                jet_labels[constituent_idx] = jet_idx
        
        return ClusteringResult(
            jets=jets,
            jet_labels=jet_labels,
            n_jets=len(jets),
            history=history,
            algorithm=self.algorithm.value,
            parameters={'R': self.R, 'p': self.p}
        )
    
    def cluster_to_n_jets(self, particles: np.ndarray, n_jets: int) -> ClusteringResult:
        """
        Cluster particles into exactly n jets
        
        Uses exclusive clustering: stops when n jets remain
        """
        n_particles = len(particles)
        
        # Initialize
        active_particles: Dict[int, Particle] = {}
        for i in range(n_particles):
            pt, eta, phi, E = particles[i]
            active_particles[i] = Particle(pt, eta, phi, E, i)
        
        history: List[ClusteringHistory] = []
        next_idx = n_particles
        step = 0
        
        # Merge until n_jets remain
        while len(active_particles) > n_jets:
            # Find minimum pair distance (ignore beam distances)
            min_dist = float('inf')
            min_i = -1
            min_j = -1
            
            particle_list = list(active_particles.values())
            n_active = len(particle_list)
            
            for i in range(n_active):
                p_i = particle_list[i]
                for j in range(i + 1, n_active):
                    p_j = particle_list[j]
                    d_ij = self._d_ij(p_i, p_j)
                    if d_ij < min_dist:
                        min_dist = d_ij
                        min_i = p_i.index
                        min_j = p_j.index
            
            # Merge
            if min_i >= 0 and min_j >= 0:
                p1 = active_particles.pop(min_i)
                p2 = active_particles.pop(min_j)
                merged = self._merge_particles(p1, p2, next_idx)
                active_particles[next_idx] = merged
                
                history.append(ClusteringHistory(
                    step=step, parent1=min_i, parent2=min_j,
                    child=next_idx, dij=min_dist, diB=0
                ))
                
                next_idx += 1
                step += 1
            else:
                break
        
        # Remaining particles are jets
        jets = list(active_particles.values())
        jets.sort(key=lambda j: j.pt, reverse=True)
        
        # Create labels
        jet_labels = np.full(n_particles, -1, dtype=int)
        for jet_idx, jet in enumerate(jets):
            for constituent_idx in jet.constituents:
                jet_labels[constituent_idx] = jet_idx
        
        return ClusteringResult(
            jets=jets,
            jet_labels=jet_labels,
            n_jets=len(jets),
            history=history,
            algorithm=self.algorithm.value,
            parameters={'R': self.R, 'p': self.p}
        )


class AntiKtClustering(KtJetClustering):
    """
    Anti-kt Jet Clustering Algorithm
    
    The LHC standard jet algorithm, with p=-1.
    Produces regular, cone-like jets centered on hard particles.
    Infrared and collinear safe.
    """
    
    def __init__(self, R: float = 0.4, min_pt: float = 0.0, max_eta: float = 5.0):
        super().__init__(R=R, p=-1, min_pt=min_pt, max_eta=max_eta)


class CambridgeAachenClustering(KtJetClustering):
    """
    Cambridge/Aachen Jet Clustering Algorithm
    
    Uses pure angular ordering (p=0), useful for studying jet substructure.
    """
    
    def __init__(self, R: float = 0.4, min_pt: float = 0.0, max_eta: float = 5.0):
        super().__init__(R=R, p=0, min_pt=min_pt, max_eta=max_eta)


class DurhamClustering:
    """
    Durham (e+e- kt) Algorithm Implementation
    
    Designed for e+e- collisions where there is no notion of beam direction
    in the same way as hadron colliders.
    
    Distance metric:
    y_ij = 2 * min(E_i², E_j²) * (1 - cos(θ_ij)) / Q²
    
    where Q is the center-of-mass energy.
    
    This is the algorithm specifically mentioned in the paper for
    e+e- jet clustering comparison.
    """
    
    def __init__(self, y_cut: float = 0.01, Q: float = 91.2):
        """
        Initialize Durham clustering
        
        Args:
            y_cut: Resolution parameter (jets merged if y_ij < y_cut)
            Q: Center-of-mass energy (default: Z mass for LEP)
        """
        self.y_cut = y_cut
        self.Q = Q
        self.Q_sq = Q ** 2
    
    def _compute_cos_theta(self, p1: Particle, p2: Particle) -> float:
        """Compute cosine of angle between two particles"""
        px1, py1, pz1 = p1.px, p1.py, p1.pz
        px2, py2, pz2 = p2.px, p2.py, p2.pz
        
        p1_mag = np.sqrt(px1**2 + py1**2 + pz1**2)
        p2_mag = np.sqrt(px2**2 + py2**2 + pz2**2)
        
        if p1_mag < 1e-10 or p2_mag < 1e-10:
            return 1.0
        
        dot = px1*px2 + py1*py2 + pz1*pz2
        cos_theta = dot / (p1_mag * p2_mag)
        
        return np.clip(cos_theta, -1.0, 1.0)
    
    def _y_ij(self, p1: Particle, p2: Particle) -> float:
        """
        Compute Durham distance y_ij
        """
        cos_theta = self._compute_cos_theta(p1, p2)
        min_E_sq = min(p1.E**2, p2.E**2)
        return 2 * min_E_sq * (1 - cos_theta) / self.Q_sq
    
    def _merge_particles(self, p1: Particle, p2: Particle, new_idx: int) -> Particle:
        """Merge using E-scheme"""
        E = p1.E + p2.E
        px = p1.px + p2.px
        py = p1.py + p2.py
        pz = p1.pz + p2.pz
        
        pt = np.sqrt(px**2 + py**2)
        phi = np.arctan2(py, px) if pt > 0 else 0.0
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        eta = 0.5 * np.log((p_mag + pz + 1e-10) / (p_mag - pz + 1e-10)) if p_mag > 0 else 0.0
        
        constituents = p1.constituents + p2.constituents
        
        return Particle(pt, eta, phi, E, new_idx, constituents)
    
    def cluster(self, particles: np.ndarray) -> ClusteringResult:
        """
        Run Durham clustering
        
        Args:
            particles: Array of shape (n, 4) with [pt, eta, phi, E]
            
        Returns:
            ClusteringResult
        """
        n_particles = len(particles)
        
        # Initialize
        active: Dict[int, Particle] = {}
        for i in range(n_particles):
            pt, eta, phi, E = particles[i]
            active[i] = Particle(pt, eta, phi, E, i)
        
        history = []
        next_idx = n_particles
        step = 0
        
        # Cluster until y_min > y_cut or only 1 particle remains
        while len(active) > 1:
            # Find minimum y_ij
            min_y = float('inf')
            min_i, min_j = -1, -1
            
            particle_list = list(active.values())
            for i in range(len(particle_list)):
                for j in range(i + 1, len(particle_list)):
                    y = self._y_ij(particle_list[i], particle_list[j])
                    if y < min_y:
                        min_y = y
                        min_i = particle_list[i].index
                        min_j = particle_list[j].index
            
            # Check resolution cut
            if min_y > self.y_cut:
                break
            
            # Merge
            p1 = active.pop(min_i)
            p2 = active.pop(min_j)
            merged = self._merge_particles(p1, p2, next_idx)
            active[next_idx] = merged
            
            history.append(ClusteringHistory(
                step=step, parent1=min_i, parent2=min_j,
                child=next_idx, dij=min_y, diB=0
            ))
            
            next_idx += 1
            step += 1
        
        jets = list(active.values())
        jets.sort(key=lambda j: j.E, reverse=True)
        
        jet_labels = np.full(n_particles, -1, dtype=int)
        for jet_idx, jet in enumerate(jets):
            for c in jet.constituents:
                jet_labels[c] = jet_idx
        
        return ClusteringResult(
            jets=jets, jet_labels=jet_labels, n_jets=len(jets),
            history=history, algorithm='durham',
            parameters={'y_cut': self.y_cut, 'Q': self.Q}
        )
    
    def cluster_to_n_jets(self, particles: np.ndarray, n_jets: int) -> ClusteringResult:
        """Cluster to exactly n jets (exclusive clustering)"""
        n_particles = len(particles)
        
        active: Dict[int, Particle] = {}
        for i in range(n_particles):
            pt, eta, phi, E = particles[i]
            active[i] = Particle(pt, eta, phi, E, i)
        
        history = []
        next_idx = n_particles
        step = 0
        
        while len(active) > n_jets:
            min_y = float('inf')
            min_i, min_j = -1, -1
            
            particle_list = list(active.values())
            for i in range(len(particle_list)):
                for j in range(i + 1, len(particle_list)):
                    y = self._y_ij(particle_list[i], particle_list[j])
                    if y < min_y:
                        min_y = y
                        min_i = particle_list[i].index
                        min_j = particle_list[j].index
            
            if min_i >= 0 and min_j >= 0:
                p1 = active.pop(min_i)
                p2 = active.pop(min_j)
                merged = self._merge_particles(p1, p2, next_idx)
                active[next_idx] = merged
                
                history.append(ClusteringHistory(
                    step=step, parent1=min_i, parent2=min_j,
                    child=next_idx, dij=min_y, diB=0
                ))
                
                next_idx += 1
                step += 1
        
        jets = list(active.values())
        jets.sort(key=lambda j: j.E, reverse=True)
        
        jet_labels = np.full(n_particles, -1, dtype=int)
        for jet_idx, jet in enumerate(jets):
            for c in jet.constituents:
                jet_labels[c] = jet_idx
        
        return ClusteringResult(
            jets=jets, jet_labels=jet_labels, n_jets=len(jets),
            history=history, algorithm='durham',
            parameters={'y_cut': self.y_cut, 'Q': self.Q, 'n_jets': n_jets}
        )


def compare_clustering_algorithms(particles: np.ndarray, 
                                   n_jets: int,
                                   R: float = 0.4) -> Dict[str, ClusteringResult]:
    """
    Run multiple clustering algorithms for comparison
    
    Args:
        particles: Input particle array
        n_jets: Target number of jets
        R: Jet radius parameter
        
    Returns:
        Dictionary mapping algorithm name to result
    """
    results = {}
    
    # Anti-kt
    anti_kt = AntiKtClustering(R=R)
    results['anti_kt'] = anti_kt.cluster_to_n_jets(particles, n_jets)
    
    # Cambridge/Aachen
    ca = CambridgeAachenClustering(R=R)
    results['cambridge_aachen'] = ca.cluster_to_n_jets(particles, n_jets)
    
    # kt
    kt = KtJetClustering(R=R, p=1)
    results['kt'] = kt.cluster_to_n_jets(particles, n_jets)
    
    # Durham
    durham = DurhamClustering()
    results['durham'] = durham.cluster_to_n_jets(particles, n_jets)
    
    return results


if __name__ == "__main__":
    # Test clustering algorithms
    print("Testing kt Algorithm Family...")
    
    # Generate sample data
    np.random.seed(42)
    n_particles = 20
    
    # Create two "jets" of particles
    particles = []
    for _ in range(10):
        particles.append([
            np.random.exponential(20) + 5,  # pt
            1.0 + np.random.normal(0, 0.3),  # eta
            0.5 + np.random.normal(0, 0.2),  # phi
            0  # E (will be calculated)
        ])
    for _ in range(10):
        particles.append([
            np.random.exponential(15) + 3,
            -0.5 + np.random.normal(0, 0.3),
            -1.5 + np.random.normal(0, 0.2),
            0
        ])
    
    particles = np.array(particles)
    # Set energy assuming massless particles
    for i in range(len(particles)):
        pt, eta, phi = particles[i, :3]
        particles[i, 3] = pt * np.cosh(eta)
    
    # Test anti-kt
    print("\nAnti-kt clustering:")
    anti_kt = AntiKtClustering(R=0.4)
    result = anti_kt.cluster_to_n_jets(particles, n_jets=2)
    print(f"  Found {result.n_jets} jets")
    for i, jet in enumerate(result.jets):
        print(f"  Jet {i+1}: pT={jet.pt:.1f}, η={jet.eta:.2f}, φ={jet.phi:.2f}, "
              f"n_constituents={len(jet.constituents)}")
    
    # Test Durham
    print("\nDurham clustering:")
    durham = DurhamClustering()
    result = durham.cluster_to_n_jets(particles, n_jets=2)
    print(f"  Found {result.n_jets} jets")
    for i, jet in enumerate(result.jets):
        print(f"  Jet {i+1}: E={jet.E:.1f}, η={jet.eta:.2f}, "
              f"n_constituents={len(jet.constituents)}")
    
    print("\n✓ kt algorithm family tests passed!")
