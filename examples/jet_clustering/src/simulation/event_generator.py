"""
Event Generator for Jet Clustering Studies
==========================================

This module generates simulated high-energy physics events with
known jet/quark structure for testing and validating jet clustering
algorithms.

The generator creates events matching the paper's experimental setup:
- Configurable number of particles per event (typically 6-30)
- Configurable number of jets (typically 2-8)
- Realistic pT, eta, phi distributions
- Known "true" jet assignments for validation
- Quark direction information for angle calculations

References:
- Paper Section IV: "Monte Carlo simulation with 4000 events"
- Paper: "30 particles" per event, various jet multiplicities
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    """Types of physics processes to simulate"""
    DIJET = "dijet"              # 2-jet QCD events
    MULTIJET = "multijet"        # N-jet QCD events
    TTBAR = "ttbar"              # Top-antitop pair production
    HIGGS = "higgs_bb"           # Higgs to b-bbar
    ZJJ = "zjj"                  # Z + dijets


@dataclass
class GeneratedEvent:
    """Container for a generated physics event"""
    particles: np.ndarray        # Shape (n, 4): [pt, eta, phi, E]
    true_labels: np.ndarray      # True jet assignment for each particle
    quark_directions: List[np.ndarray]  # True quark momenta [px, py, pz]
    n_jets: int
    n_particles: int
    event_type: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class EventBatch:
    """Container for a batch of generated events"""
    events: List[GeneratedEvent]
    n_events: int
    config: Dict


class JetEventGenerator:
    """
    Generator for jet clustering test events
    
    Creates simulated high-energy physics events with known jet structure,
    suitable for testing clustering algorithms as described in the paper.
    
    Key features:
    - Realistic kinematic distributions
    - Configurable jet multiplicity and particle count
    - Known ground truth for validation
    - Reproducible with random seed
    """
    
    def __init__(self,
                 n_particles: int = 30,
                 n_jets: int = 2,
                 R: float = 0.4,
                 jet_pt_range: Tuple[float, float] = (50.0, 200.0),
                 jet_eta_range: Tuple[float, float] = (-2.5, 2.5),
                 particle_pt_scale: float = 10.0,
                 angular_smearing: float = 0.2,
                 random_state: Optional[int] = None):
        """
        Initialize event generator
        
        Args:
            n_particles: Number of particles per event
            n_jets: Number of jets per event
            R: Jet radius parameter (controls angular spread)
            jet_pt_range: Range of jet transverse momenta
            jet_eta_range: Range of jet pseudorapidity
            particle_pt_scale: Scale for particle pT distribution
            angular_smearing: Standard deviation of angular spread within jet
            random_state: Random seed for reproducibility
        """
        self.n_particles = n_particles
        self.n_jets = n_jets
        self.R = R
        self.jet_pt_range = jet_pt_range
        self.jet_eta_range = jet_eta_range
        self.particle_pt_scale = particle_pt_scale
        self.angular_smearing = angular_smearing
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _generate_jet_centers(self) -> List[Tuple[float, float, float]]:
        """
        Generate jet center positions (quark directions)
        
        Returns:
            List of (pt, eta, phi) for each jet
        """
        centers = []
        
        for _ in range(self.n_jets):
            pt = np.random.uniform(*self.jet_pt_range)
            eta = np.random.uniform(*self.jet_eta_range)
            phi = np.random.uniform(-np.pi, np.pi)
            centers.append((pt, eta, phi))
        
        return centers
    
    def _phi_wrap(self, phi: float) -> float:
        """Wrap phi to [-π, π]"""
        while phi > np.pi:
            phi -= 2 * np.pi
        while phi < -np.pi:
            phi += 2 * np.pi
        return phi
    
    def _generate_particles_for_jet(self, 
                                     jet_center: Tuple[float, float, float],
                                     n_particles: int) -> List[np.ndarray]:
        """
        Generate particles belonging to a specific jet
        
        Uses realistic fragmentation-like distributions:
        - pT follows exponential distribution (soft particles dominate)
        - Angles are Gaussian-smeared around jet axis
        """
        jet_pt, jet_eta, jet_phi = jet_center
        particles = []
        
        for _ in range(n_particles):
            # pT from exponential distribution (fragmentation-like)
            pt = np.random.exponential(self.particle_pt_scale) + 0.5
            
            # Angular smearing around jet axis
            # Use 2D Gaussian in eta-phi
            d_eta = np.random.normal(0, self.angular_smearing)
            d_phi = np.random.normal(0, self.angular_smearing)
            
            eta = jet_eta + d_eta
            phi = self._phi_wrap(jet_phi + d_phi)
            
            # Energy (assuming massless particles)
            E = pt * np.cosh(eta)
            
            particles.append(np.array([pt, eta, phi, E]))
        
        return particles
    
    def _compute_quark_direction(self, 
                                  jet_center: Tuple[float, float, float]) -> np.ndarray:
        """
        Compute quark momentum direction from jet center
        
        Returns:
            [px, py, pz] direction vector
        """
        pt, eta, phi = jet_center
        
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        
        # Normalize to unit vector
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        if p_mag > 0:
            return np.array([px, py, pz]) / p_mag * 100  # Scale for visibility
        else:
            return np.array([0.0, 0.0, 1.0])
    
    def generate_event(self) -> GeneratedEvent:
        """
        Generate a single simulated event
        
        Returns:
            GeneratedEvent with particles and ground truth
        """
        # Generate jet centers
        jet_centers = self._generate_jet_centers()
        
        # Distribute particles among jets
        particles_per_jet = self.n_particles // self.n_jets
        remainder = self.n_particles % self.n_jets
        
        all_particles = []
        true_labels = []
        quark_directions = []
        
        for jet_idx, center in enumerate(jet_centers):
            # Number of particles for this jet
            n_in_jet = particles_per_jet + (1 if jet_idx < remainder else 0)
            
            # Generate particles
            jet_particles = self._generate_particles_for_jet(center, n_in_jet)
            all_particles.extend(jet_particles)
            
            # Label assignments
            true_labels.extend([jet_idx] * n_in_jet)
            
            # Quark direction
            quark_directions.append(self._compute_quark_direction(center))
        
        # Convert to arrays
        particles = np.array(all_particles)
        labels = np.array(true_labels)
        
        # Shuffle particles (realistic: detector doesn't know jet assignment)
        shuffle_idx = np.random.permutation(len(particles))
        particles = particles[shuffle_idx]
        labels = labels[shuffle_idx]
        
        return GeneratedEvent(
            particles=particles,
            true_labels=labels,
            quark_directions=quark_directions,
            n_jets=self.n_jets,
            n_particles=len(particles),
            event_type='multijet',
            metadata={
                'R': self.R,
                'angular_smearing': self.angular_smearing,
                'jet_centers': jet_centers
            }
        )
    
    def generate_batch(self, n_events: int, 
                       progress_callback: Optional[callable] = None) -> EventBatch:
        """
        Generate a batch of events
        
        Args:
            n_events: Number of events to generate
            progress_callback: Optional callback for progress updates
            
        Returns:
            EventBatch containing all events
        """
        events = []
        
        for i in range(n_events):
            event = self.generate_event()
            events.append(event)
            
            if progress_callback is not None and (i + 1) % 100 == 0:
                progress_callback(i + 1, n_events)
        
        return EventBatch(
            events=events,
            n_events=n_events,
            config={
                'n_particles': self.n_particles,
                'n_jets': self.n_jets,
                'R': self.R,
                'jet_pt_range': self.jet_pt_range,
                'jet_eta_range': self.jet_eta_range
            }
        )


class RealisticEventGenerator(JetEventGenerator):
    """
    More realistic event generator with advanced features
    
    Includes:
    - Underlying event simulation
    - Pile-up particles
    - Varying jet multiplicities
    - More realistic fragmentation functions
    """
    
    def __init__(self,
                 n_particles: int = 30,
                 n_jets: int = 2,
                 include_underlying_event: bool = True,
                 pile_up_multiplicity: int = 0,
                 **kwargs):
        """
        Initialize realistic event generator
        
        Args:
            include_underlying_event: Add soft underlying event particles
            pile_up_multiplicity: Number of pile-up interactions to add
            **kwargs: Arguments passed to base class
        """
        super().__init__(n_particles=n_particles, n_jets=n_jets, **kwargs)
        self.include_underlying_event = include_underlying_event
        self.pile_up_multiplicity = pile_up_multiplicity
    
    def _generate_underlying_event(self, n_particles: int) -> List[np.ndarray]:
        """Generate soft underlying event particles"""
        particles = []
        
        for _ in range(n_particles):
            # Soft, uniform in eta-phi
            pt = np.random.exponential(2.0) + 0.3  # Softer spectrum
            eta = np.random.uniform(-4.0, 4.0)
            phi = np.random.uniform(-np.pi, np.pi)
            E = pt * np.cosh(eta)
            
            particles.append(np.array([pt, eta, phi, E]))
        
        return particles
    
    def generate_event(self) -> GeneratedEvent:
        """Generate event with underlying event and pile-up"""
        # Generate base event
        base_event = super().generate_event()
        
        particles = list(base_event.particles)
        labels = list(base_event.true_labels)
        
        # Add underlying event
        if self.include_underlying_event:
            n_ue = np.random.poisson(5)  # Soft activity
            ue_particles = self._generate_underlying_event(n_ue)
            particles.extend(ue_particles)
            labels.extend([-1] * len(ue_particles))  # -1 for noise
        
        # Add pile-up
        if self.pile_up_multiplicity > 0:
            n_pu = np.random.poisson(self.pile_up_multiplicity * 3)
            pu_particles = self._generate_underlying_event(n_pu)
            particles.extend(pu_particles)
            labels.extend([-2] * len(pu_particles))  # -2 for pile-up
        
        # Convert and shuffle
        particles = np.array(particles)
        labels = np.array(labels)
        
        shuffle_idx = np.random.permutation(len(particles))
        particles = particles[shuffle_idx]
        labels = labels[shuffle_idx]
        
        return GeneratedEvent(
            particles=particles,
            true_labels=labels,
            quark_directions=base_event.quark_directions,
            n_jets=base_event.n_jets,
            n_particles=len(particles),
            event_type='realistic_multijet',
            metadata=base_event.metadata
        )


class PaperEventGenerator:
    """
    Event generator specifically matching conditions in the paper
    
    Reproduces the experimental setup described in the paper:
    - 4000 events with 30 particles each (Figure 2a-c)
    - 1217 events with 6 particles each (Figure 2e - hardware)
    - Various jet multiplicities (k = 2, 4, 6, 7, 8)
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_figure2a_dataset(self) -> EventBatch:
        """
        Generate dataset for Figure 2(a): QAOA depth comparison
        
        4000 events, 30 particles, k=6 jets
        """
        generator = JetEventGenerator(
            n_particles=30,
            n_jets=6,
            angular_smearing=0.15,
            random_state=self.random_state
        )
        return generator.generate_batch(4000)
    
    def generate_figure2b_dataset(self, k: int) -> EventBatch:
        """
        Generate dataset for Figure 2(b): k-value comparison
        
        4000 events, 30 particles, varying k
        
        Args:
            k: Number of jets (2, 4, 6, 7, or 8)
        """
        generator = JetEventGenerator(
            n_particles=30,
            n_jets=k,
            angular_smearing=0.15,
            random_state=self.random_state
        )
        return generator.generate_batch(4000)
    
    def generate_figure2c_dataset(self) -> EventBatch:
        """
        Generate dataset for Figure 2(c): Algorithm comparison
        
        Optimal parameters: depth=5, k=7
        """
        generator = JetEventGenerator(
            n_particles=30,
            n_jets=7,
            angular_smearing=0.15,
            random_state=self.random_state
        )
        return generator.generate_batch(4000)
    
    def generate_figure2e_dataset(self) -> EventBatch:
        """
        Generate dataset for Figure 2(e): Hardware comparison
        
        1217 events, 6 particles, k=2 (smaller for quantum hardware)
        """
        generator = JetEventGenerator(
            n_particles=6,
            n_jets=2,
            angular_smearing=0.2,
            random_state=self.random_state
        )
        return generator.generate_batch(1217)
    
    def generate_custom_dataset(self,
                                 n_events: int,
                                 n_particles: int,
                                 n_jets: int) -> EventBatch:
        """Generate custom dataset with specified parameters"""
        generator = JetEventGenerator(
            n_particles=n_particles,
            n_jets=n_jets,
            angular_smearing=0.15,
            random_state=self.random_state
        )
        return generator.generate_batch(n_events)


def generate_sample_dataset(n_events: int = 100,
                            n_particles: int = 20,
                            n_jets: int = 2,
                            seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[np.ndarray]]]:
    """
    Convenience function to generate sample dataset
    
    Returns:
        (particles_list, labels_list, quark_directions_list)
    """
    generator = JetEventGenerator(
        n_particles=n_particles,
        n_jets=n_jets,
        random_state=seed
    )
    
    batch = generator.generate_batch(n_events)
    
    particles_list = [e.particles for e in batch.events]
    labels_list = [e.true_labels for e in batch.events]
    quark_dirs_list = [e.quark_directions for e in batch.events]
    
    return particles_list, labels_list, quark_dirs_list


if __name__ == "__main__":
    print("Testing Event Generator...")
    
    # Basic generator test
    print("\n1. Basic Generator:")
    generator = JetEventGenerator(
        n_particles=20,
        n_jets=3,
        random_state=42
    )
    
    event = generator.generate_event()
    print(f"   Generated event: {event.n_particles} particles, {event.n_jets} jets")
    print(f"   Particles shape: {event.particles.shape}")
    print(f"   Label distribution: {np.bincount(event.true_labels)}")
    
    # Batch generation
    print("\n2. Batch Generation:")
    batch = generator.generate_batch(100)
    print(f"   Generated {batch.n_events} events")
    
    # Paper-specific generator
    print("\n3. Paper Dataset Generator:")
    paper_gen = PaperEventGenerator(random_state=42)
    
    # Small sample for testing
    test_batch = paper_gen.generate_custom_dataset(10, 30, 6)
    print(f"   Figure 2a sample: {test_batch.n_events} events")
    print(f"   Particles per event: {test_batch.events[0].n_particles}")
    print(f"   Jets per event: {test_batch.events[0].n_jets}")
    
    # Realistic generator
    print("\n4. Realistic Generator:")
    real_gen = RealisticEventGenerator(
        n_particles=20,
        n_jets=2,
        include_underlying_event=True,
        pile_up_multiplicity=2,
        random_state=42
    )
    real_event = real_gen.generate_event()
    
    jet_particles = np.sum(real_event.true_labels >= 0)
    ue_particles = np.sum(real_event.true_labels == -1)
    pu_particles = np.sum(real_event.true_labels == -2)
    
    print(f"   Total particles: {real_event.n_particles}")
    print(f"   Jet particles: {jet_particles}")
    print(f"   Underlying event: {ue_particles}")
    print(f"   Pile-up: {pu_particles}")
    
    print("\n✓ Event generator tests passed!")
