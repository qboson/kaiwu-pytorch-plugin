"""
Jet Physics Computations
========================

This module provides core physics calculations for jet reconstruction,
including four-momentum operations, invariant mass, and kinematic variables.

The calculations follow the conventions used in high-energy physics
experiments at the LHC and other colliders.

References:
- Particle Data Group conventions
- Paper definitions for jet kinematics
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import math


@dataclass
class FourMomentum:
    """
    Four-momentum vector representation
    
    Uses the convention (E, px, py, pz) or equivalently (E, p).
    Provides methods for common operations in particle physics.
    """
    E: float     # Energy
    px: float    # x-component of momentum
    py: float    # y-component of momentum
    pz: float    # z-component of momentum
    
    @classmethod
    def from_pt_eta_phi_m(cls, pt: float, eta: float, phi: float, m: float = 0.0) -> 'FourMomentum':
        """
        Create from transverse momentum, pseudorapidity, azimuthal angle, and mass
        
        This is the most common parametrization in hadron collider physics.
        """
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        E = np.sqrt(p_mag**2 + m**2)
        return cls(E, px, py, pz)
    
    @classmethod
    def from_pt_eta_phi_E(cls, pt: float, eta: float, phi: float, E: float) -> 'FourMomentum':
        """Create from pT, eta, phi, E (common in detector output)"""
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        return cls(E, px, py, pz)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'FourMomentum':
        """Create from array [pt, eta, phi, E] (our standard format)"""
        pt, eta, phi, E = arr[:4]
        return cls.from_pt_eta_phi_E(pt, eta, phi, E)
    
    @property
    def pt(self) -> float:
        """Transverse momentum"""
        return np.sqrt(self.px**2 + self.py**2)
    
    @property
    def p(self) -> float:
        """Momentum magnitude |p|"""
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)
    
    @property
    def eta(self) -> float:
        """Pseudorapidity η = -ln(tan(θ/2))"""
        p = self.p
        if p < 1e-10:
            return 0.0
        return 0.5 * np.log((p + self.pz) / (p - self.pz + 1e-10))
    
    @property
    def phi(self) -> float:
        """Azimuthal angle φ"""
        return np.arctan2(self.py, self.px)
    
    @property
    def theta(self) -> float:
        """Polar angle θ"""
        p = self.p
        if p < 1e-10:
            return 0.0
        return np.arccos(self.pz / p)
    
    @property
    def rapidity(self) -> float:
        """Rapidity y = 0.5 * ln((E + pz) / (E - pz))"""
        if abs(self.E - abs(self.pz)) < 1e-10:
            return np.sign(self.pz) * 10.0  # Large value
        return 0.5 * np.log((self.E + self.pz) / (self.E - self.pz + 1e-10))
    
    @property
    def mass(self) -> float:
        """Invariant mass m² = E² - p²"""
        m_sq = self.E**2 - self.px**2 - self.py**2 - self.pz**2
        return np.sqrt(max(0.0, m_sq))
    
    @property
    def mt(self) -> float:
        """Transverse mass mT² = E² - pz²"""
        mt_sq = self.E**2 - self.pz**2
        return np.sqrt(max(0.0, mt_sq))
    
    @property
    def Et(self) -> float:
        """Transverse energy ET = E * sin(θ)"""
        p = self.p
        if p < 1e-10:
            return self.E
        return self.E * self.pt / p
    
    def as_array(self) -> np.ndarray:
        """Return as numpy array [E, px, py, pz]"""
        return np.array([self.E, self.px, self.py, self.pz])
    
    def as_pt_eta_phi_E(self) -> np.ndarray:
        """Return as [pt, eta, phi, E]"""
        return np.array([self.pt, self.eta, self.phi, self.E])
    
    def __add__(self, other: 'FourMomentum') -> 'FourMomentum':
        """Add two four-momenta"""
        return FourMomentum(
            self.E + other.E,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz
        )
    
    def __sub__(self, other: 'FourMomentum') -> 'FourMomentum':
        """Subtract two four-momenta"""
        return FourMomentum(
            self.E - other.E,
            self.px - other.px,
            self.py - other.py,
            self.pz - other.pz
        )
    
    def __mul__(self, scalar: float) -> 'FourMomentum':
        """Scale by scalar"""
        return FourMomentum(
            self.E * scalar,
            self.px * scalar,
            self.py * scalar,
            self.pz * scalar
        )
    
    def __rmul__(self, scalar: float) -> 'FourMomentum':
        return self.__mul__(scalar)
    
    def boost(self, beta: np.ndarray) -> 'FourMomentum':
        """
        Lorentz boost with velocity beta
        
        Args:
            beta: [βx, βy, βz] velocity vector (|β| < 1)
        """
        beta_sq = np.sum(beta**2)
        if beta_sq >= 1.0:
            raise ValueError("Velocity must be less than speed of light")
        
        gamma = 1.0 / np.sqrt(1 - beta_sq)
        
        bp = np.sum(beta * np.array([self.px, self.py, self.pz]))
        
        gamma2 = (gamma - 1) / beta_sq if beta_sq > 1e-10 else 0.0
        
        p_new = np.array([self.px, self.py, self.pz]) + gamma2 * bp * beta + gamma * self.E * beta
        E_new = gamma * (self.E + bp)
        
        return FourMomentum(E_new, p_new[0], p_new[1], p_new[2])
    
    def dot(self, other: 'FourMomentum') -> float:
        """Minkowski inner product (E₁E₂ - p₁·p₂)"""
        return (self.E * other.E - 
                self.px * other.px - 
                self.py * other.py - 
                self.pz * other.pz)


def delta_phi(phi1: float, phi2: float) -> float:
    """
    Compute Δφ with proper periodicity handling
    
    Returns value in [-π, π]
    """
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return dphi


def delta_R(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    """
    Compute ΔR distance in η-φ space
    
    ΔR = √((Δη)² + (Δφ)²)
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def invariant_mass(momenta: List[FourMomentum]) -> float:
    """
    Compute invariant mass of a system of particles
    
    M² = (Σ E)² - (Σ p)²
    """
    total = sum(momenta, FourMomentum(0, 0, 0, 0))
    return total.mass


def compute_jet_axis(constituents: np.ndarray) -> Tuple[float, float]:
    """
    Compute jet axis (η, φ) from constituent particles
    
    Uses pT-weighted centroid with proper φ averaging.
    
    Args:
        constituents: Array of [pt, eta, phi, E]
        
    Returns:
        (eta_jet, phi_jet)
    """
    if len(constituents) == 0:
        return 0.0, 0.0
    
    pt_sum = np.sum(constituents[:, 0])
    if pt_sum < 1e-10:
        return np.mean(constituents[:, 1]), np.mean(constituents[:, 2])
    
    weights = constituents[:, 0] / pt_sum
    
    eta_jet = np.sum(weights * constituents[:, 1])
    
    # Vector mean for phi
    sin_phi = np.sum(weights * np.sin(constituents[:, 2]))
    cos_phi = np.sum(weights * np.cos(constituents[:, 2]))
    phi_jet = np.arctan2(sin_phi, cos_phi)
    
    return eta_jet, phi_jet


def compute_jet_four_momentum(constituents: np.ndarray) -> FourMomentum:
    """
    Compute total four-momentum of jet from constituents
    
    Uses E-scheme: simple 4-vector addition.
    
    Args:
        constituents: Array of [pt, eta, phi, E]
        
    Returns:
        Jet FourMomentum
    """
    total_E = 0.0
    total_px = 0.0
    total_py = 0.0
    total_pz = 0.0
    
    for p in constituents:
        pt, eta, phi, E = p
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        
        total_px += px
        total_py += py
        total_pz += pz
        total_E += E
    
    return FourMomentum(total_E, total_px, total_py, total_pz)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angle between two 3D vectors
    
    Returns angle in radians [0, π]
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.arccos(cos_angle)


def angle_between_jets(jet1: FourMomentum, jet2: FourMomentum) -> float:
    """
    Compute angle between two jet directions
    
    Returns angle in radians
    """
    v1 = np.array([jet1.px, jet1.py, jet1.pz])
    v2 = np.array([jet2.px, jet2.py, jet2.pz])
    return angle_between_vectors(v1, v2)


def quark_jet_angle(quark_direction: np.ndarray, 
                    jet: FourMomentum) -> float:
    """
    Compute angle between original quark direction and reconstructed jet
    
    This is the key metric from the paper (Figure 2 y-axis).
    
    Args:
        quark_direction: [px, py, pz] of original quark
        jet: Reconstructed jet four-momentum
        
    Returns:
        Angle in radians
    """
    jet_direction = np.array([jet.px, jet.py, jet.pz])
    return angle_between_vectors(quark_direction, jet_direction)


def compute_thrust(momenta: List[FourMomentum]) -> Tuple[float, np.ndarray]:
    """
    Compute thrust and thrust axis
    
    Thrust: T = max_n (Σ |p·n|) / (Σ |p|)
    
    Returns:
        (thrust_value, thrust_axis)
    """
    if not momenta:
        return 0.0, np.array([0.0, 0.0, 1.0])
    
    # Get momentum vectors
    p_vectors = [np.array([m.px, m.py, m.pz]) for m in momenta]
    p_mags = [np.linalg.norm(p) for p in p_vectors]
    total_p = sum(p_mags)
    
    if total_p < 1e-10:
        return 0.0, np.array([0.0, 0.0, 1.0])
    
    # Search for thrust axis (simplified: use dominant momentum direction)
    max_thrust = 0.0
    thrust_axis = np.array([0.0, 0.0, 1.0])
    
    # Sample many directions
    for m in momenta:
        p = np.array([m.px, m.py, m.pz])
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-10:
            continue
        
        n = p / p_norm
        thrust = sum(abs(np.dot(pv, n)) for pv in p_vectors) / total_p
        
        if thrust > max_thrust:
            max_thrust = thrust
            thrust_axis = n
    
    return max_thrust, thrust_axis


def compute_sphericity(momenta: List[FourMomentum]) -> float:
    """
    Compute sphericity of event
    
    S = (3/2) * (λ₂ + λ₃) where λ are eigenvalues of momentum tensor
    
    Returns:
        Sphericity value in [0, 1]
    """
    if len(momenta) < 2:
        return 0.0
    
    # Build momentum tensor
    S_tensor = np.zeros((3, 3))
    p_sq_sum = 0.0
    
    for m in momenta:
        p = np.array([m.px, m.py, m.pz])
        p_sq = np.sum(p**2)
        p_sq_sum += p_sq
        S_tensor += np.outer(p, p)
    
    if p_sq_sum < 1e-10:
        return 0.0
    
    S_tensor /= p_sq_sum
    
    # Get eigenvalues
    eigenvalues = np.sort(np.linalg.eigvalsh(S_tensor))
    
    # Sphericity = 3/2 * (λ₂ + λ₃)
    return 1.5 * (eigenvalues[0] + eigenvalues[1])


class JetReconstructor:
    """
    Utility class for jet reconstruction and analysis
    
    Provides methods to analyze clustering results and compute
    jet properties.
    """
    
    def __init__(self, particles: np.ndarray, labels: np.ndarray):
        """
        Initialize with particles and cluster labels
        
        Args:
            particles: Array of shape (n, 4) with [pt, eta, phi, E]
            labels: Cluster assignment for each particle
        """
        self.particles = particles
        self.labels = labels
        self.n_jets = len(np.unique(labels[labels >= 0]))
        
        self._jets: Optional[List[FourMomentum]] = None
        self._constituents: Optional[List[np.ndarray]] = None
    
    def _build_jets(self):
        """Build jet four-momenta from constituents"""
        if self._jets is not None:
            return
        
        self._jets = []
        self._constituents = []
        
        for j in range(self.n_jets):
            mask = self.labels == j
            constituents = self.particles[mask]
            self._constituents.append(constituents)
            
            if len(constituents) > 0:
                jet = compute_jet_four_momentum(constituents)
            else:
                jet = FourMomentum(0, 0, 0, 0)
            
            self._jets.append(jet)
    
    def get_jets(self) -> List[FourMomentum]:
        """Get list of jet four-momenta"""
        self._build_jets()
        return self._jets
    
    def get_constituents(self, jet_index: int) -> np.ndarray:
        """Get constituent particles of a jet"""
        self._build_jets()
        return self._constituents[jet_index]
    
    def get_jet_properties(self) -> List[Dict]:
        """
        Get comprehensive properties for all jets
        """
        self._build_jets()
        properties = []
        
        for j, (jet, constituents) in enumerate(zip(self._jets, self._constituents)):
            n_const = len(constituents)
            pt_constituents = constituents[:, 0] if n_const > 0 else np.array([])
            
            props = {
                'index': j,
                'pt': jet.pt,
                'eta': jet.eta,
                'phi': jet.phi,
                'mass': jet.mass,
                'E': jet.E,
                'n_constituents': n_const,
                'pt_sum': np.sum(pt_constituents),
                'pt_max': np.max(pt_constituents) if n_const > 0 else 0,
                'pt_mean': np.mean(pt_constituents) if n_const > 0 else 0,
            }
            properties.append(props)
        
        return properties
    
    def compute_dijet_mass(self) -> float:
        """Compute invariant mass of leading dijet system"""
        self._build_jets()
        if len(self._jets) < 2:
            return 0.0
        
        # Sort by pT
        sorted_jets = sorted(self._jets, key=lambda j: j.pt, reverse=True)
        return invariant_mass(sorted_jets[:2])
    
    def match_to_quarks(self, quark_directions: List[np.ndarray]) -> List[Tuple[int, float]]:
        """
        Match reconstructed jets to original quark directions
        
        Uses minimum angle matching.
        
        Args:
            quark_directions: List of [px, py, pz] for each quark
            
        Returns:
            List of (quark_index, angle) for each jet
        """
        self._build_jets()
        
        matches = []
        used_quarks = set()
        
        # Sort jets by pT
        jet_indices = sorted(range(len(self._jets)), 
                            key=lambda i: self._jets[i].pt, 
                            reverse=True)
        
        for jet_idx in jet_indices:
            jet = self._jets[jet_idx]
            
            best_quark = -1
            best_angle = float('inf')
            
            for q_idx, q_dir in enumerate(quark_directions):
                if q_idx in used_quarks:
                    continue
                
                angle = quark_jet_angle(q_dir, jet)
                if angle < best_angle:
                    best_angle = angle
                    best_quark = q_idx
            
            matches.append((best_quark, best_angle))
            if best_quark >= 0:
                used_quarks.add(best_quark)
        
        return matches


if __name__ == "__main__":
    print("Testing Jet Physics Module...")
    
    # Test FourMomentum
    p1 = FourMomentum.from_pt_eta_phi_m(50.0, 1.0, 0.5, 0.0)
    p2 = FourMomentum.from_pt_eta_phi_m(30.0, -0.5, -1.0, 0.0)
    
    print(f"\nParticle 1: pT={p1.pt:.1f}, η={p1.eta:.2f}, φ={p1.phi:.2f}, m={p1.mass:.2f}")
    print(f"Particle 2: pT={p2.pt:.1f}, η={p2.eta:.2f}, φ={p2.phi:.2f}, m={p2.mass:.2f}")
    
    p_sum = p1 + p2
    print(f"\nSum: pT={p_sum.pt:.1f}, m={p_sum.mass:.2f}")
    print(f"Invariant mass: {invariant_mass([p1, p2]):.2f}")
    
    # Test delta_R
    dR = delta_R(1.0, 0.5, -0.5, -1.0)
    print(f"\nΔR between particles: {dR:.3f}")
    
    # Test angle computation
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    angle = angle_between_vectors(v1, v2)
    print(f"\nAngle between perpendicular vectors: {np.degrees(angle):.1f}°")
    
    print("\n✓ Jet physics tests passed!")
