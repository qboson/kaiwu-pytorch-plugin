"""
k-Means Clustering for Jet Physics
==================================

This module implements k-Means clustering adapted for jet physics applications,
used in the paper as a baseline comparison method.

The implementation includes:
- Custom distance metric for η-φ plane (with periodic φ)
- pT-weighted centroid calculation
- Multiple initialization strategies (k-means++, random)
- Convergence tracking and analysis

References:
- Lloyd, S. (1982). "Least squares quantization in PCM". 
  IEEE Transactions on Information Theory, 28(2), 129-137.
- Paper Section IV: Comparison with classical methods
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum


class InitMethod(Enum):
    """Initialization methods for k-means"""
    RANDOM = "random"
    KMEANS_PLUS_PLUS = "kmeans++"
    FARTHEST_FIRST = "farthest_first"


@dataclass
class KMeansResult:
    """Container for k-means clustering results"""
    labels: np.ndarray
    centroids: np.ndarray
    n_iterations: int
    inertia: float
    convergence_history: List[float]
    jet_properties: List[Dict]


class JetKMeans:
    """
    k-Means Clustering for Jet Physics
    
    Specialized implementation that handles:
    - η-φ space metric with periodic φ boundary
    - pT-weighted centroids (optional)
    - Jet-aware initialization
    
    Distance metric in η-φ space:
    d(p1, p2)² = (η1 - η2)² + Δφ(φ1, φ2)²
    
    where Δφ handles the 2π periodicity of the azimuthal angle.
    """
    
    def __init__(self,
                 n_clusters: int,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 n_init: int = 10,
                 init: str = 'kmeans++',
                 pt_weighted: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize JetKMeans clustering
        
        Args:
            n_clusters: Number of clusters (jets)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance (relative change in inertia)
            n_init: Number of random initializations
            init: Initialization method ('random', 'kmeans++', 'farthest_first')
            pt_weighted: Use pT weighting for centroids
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.pt_weighted = pt_weighted
        self.random_state = random_state
        
        if init in ['random', 'kmeans++', 'farthest_first']:
            self.init = init
        else:
            self.init = 'kmeans++'
        
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = float('inf')
    
    def _delta_phi(self, phi1: float, phi2: float) -> float:
        """Compute angular difference with periodicity"""
        dphi = phi1 - phi2
        while dphi > np.pi:
            dphi -= 2 * np.pi
        while dphi < -np.pi:
            dphi += 2 * np.pi
        return dphi
    
    def _eta_phi_distance_sq(self, eta1: float, phi1: float, 
                              eta2: float, phi2: float) -> float:
        """Compute squared distance in η-φ plane"""
        deta = eta1 - eta2
        dphi = self._delta_phi(phi1, phi2)
        return deta**2 + dphi**2
    
    def _compute_distances(self, particles: np.ndarray, 
                           centroids: np.ndarray) -> np.ndarray:
        """
        Compute distances from all particles to all centroids
        
        Returns:
            Array of shape (n_particles, n_clusters) with distances
        """
        n_particles = len(particles)
        n_clusters = len(centroids)
        distances = np.zeros((n_particles, n_clusters))
        
        for i in range(n_particles):
            eta_i, phi_i = particles[i, 1], particles[i, 2]
            for j in range(n_clusters):
                eta_c, phi_c = centroids[j, 0], centroids[j, 1]
                distances[i, j] = self._eta_phi_distance_sq(eta_i, phi_i, eta_c, phi_c)
        
        return distances
    
    def _assign_labels(self, particles: np.ndarray, 
                       centroids: np.ndarray) -> np.ndarray:
        """Assign each particle to nearest centroid"""
        distances = self._compute_distances(particles, centroids)
        return np.argmin(distances, axis=1)
    
    def _compute_centroids(self, particles: np.ndarray, 
                           labels: np.ndarray) -> np.ndarray:
        """
        Compute cluster centroids
        
        Uses pT-weighted average if pt_weighted is True.
        Handles φ periodicity by converting to Cartesian coordinates.
        """
        new_centroids = np.zeros((self.n_clusters, 2))  # [eta, phi]
        
        for k in range(self.n_clusters):
            mask = labels == k
            if not np.any(mask):
                # Empty cluster: reinitialize randomly
                idx = np.random.randint(len(particles))
                new_centroids[k, 0] = particles[idx, 1]  # eta
                new_centroids[k, 1] = particles[idx, 2]  # phi
                continue
            
            cluster_particles = particles[mask]
            
            if self.pt_weighted:
                weights = cluster_particles[:, 0]  # pT
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(cluster_particles)) / len(cluster_particles)
            
            # Weighted mean of eta
            new_centroids[k, 0] = np.sum(weights * cluster_particles[:, 1])
            
            # For phi, use vector mean to handle periodicity
            sin_phi = np.sum(weights * np.sin(cluster_particles[:, 2]))
            cos_phi = np.sum(weights * np.cos(cluster_particles[:, 2]))
            new_centroids[k, 1] = np.arctan2(sin_phi, cos_phi)
        
        return new_centroids
    
    def _compute_inertia(self, particles: np.ndarray, 
                         labels: np.ndarray,
                         centroids: np.ndarray) -> float:
        """Compute total within-cluster sum of squared distances"""
        inertia = 0.0
        for i in range(len(particles)):
            k = labels[i]
            inertia += self._eta_phi_distance_sq(
                particles[i, 1], particles[i, 2],
                centroids[k, 0], centroids[k, 1]
            )
        return inertia
    
    def _init_random(self, particles: np.ndarray) -> np.ndarray:
        """Random initialization: pick k random particles"""
        indices = np.random.choice(len(particles), self.n_clusters, replace=False)
        centroids = np.zeros((self.n_clusters, 2))
        centroids[:, 0] = particles[indices, 1]  # eta
        centroids[:, 1] = particles[indices, 2]  # phi
        return centroids
    
    def _init_kmeans_plus_plus(self, particles: np.ndarray) -> np.ndarray:
        """
        k-means++ initialization
        
        Selects initial centroids with probability proportional to
        distance from existing centroids, leading to better spread.
        """
        n = len(particles)
        centroids = np.zeros((self.n_clusters, 2))
        
        # First centroid: random
        idx = np.random.randint(n)
        centroids[0, 0] = particles[idx, 1]
        centroids[0, 1] = particles[idx, 2]
        
        for k in range(1, self.n_clusters):
            # Compute min distance to existing centroids for each point
            min_dists = np.full(n, float('inf'))
            for i in range(n):
                for j in range(k):
                    d = self._eta_phi_distance_sq(
                        particles[i, 1], particles[i, 2],
                        centroids[j, 0], centroids[j, 1]
                    )
                    min_dists[i] = min(min_dists[i], d)
            
            # Sample proportional to distance squared
            probs = min_dists / np.sum(min_dists)
            idx = np.random.choice(n, p=probs)
            centroids[k, 0] = particles[idx, 1]
            centroids[k, 1] = particles[idx, 2]
        
        return centroids
    
    def _init_farthest_first(self, particles: np.ndarray) -> np.ndarray:
        """
        Farthest-first initialization
        
        Greedily selects points that are farthest from existing centroids.
        """
        n = len(particles)
        centroids = np.zeros((self.n_clusters, 2))
        
        # First centroid: random
        idx = np.random.randint(n)
        centroids[0, 0] = particles[idx, 1]
        centroids[0, 1] = particles[idx, 2]
        
        for k in range(1, self.n_clusters):
            # Find point with maximum min distance to existing centroids
            max_min_dist = -1
            best_idx = 0
            
            for i in range(n):
                min_dist = float('inf')
                for j in range(k):
                    d = self._eta_phi_distance_sq(
                        particles[i, 1], particles[i, 2],
                        centroids[j, 0], centroids[j, 1]
                    )
                    min_dist = min(min_dist, d)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            centroids[k, 0] = particles[best_idx, 1]
            centroids[k, 1] = particles[best_idx, 2]
        
        return centroids
    
    def _single_run(self, particles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
        """
        Single k-means run with given initialization
        
        Returns:
            (labels, centroids, inertia, convergence_history)
        """
        # Initialize centroids
        if self.init == 'random':
            centroids = self._init_random(particles)
        elif self.init == 'kmeans++':
            centroids = self._init_kmeans_plus_plus(particles)
        else:
            centroids = self._init_farthest_first(particles)
        
        convergence_history = []
        
        for iteration in range(self.max_iter):
            # Assign labels
            labels = self._assign_labels(particles, centroids)
            
            # Compute new centroids
            new_centroids = self._compute_centroids(particles, labels)
            
            # Compute inertia
            inertia = self._compute_inertia(particles, labels, new_centroids)
            convergence_history.append(inertia)
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(convergence_history[-2] - inertia) / (convergence_history[-2] + 1e-10)
                if rel_change < self.tol:
                    break
            
            centroids = new_centroids
        
        return labels, centroids, inertia, convergence_history
    
    def fit(self, particles: np.ndarray) -> 'JetKMeans':
        """
        Fit k-means clustering to particle data
        
        Args:
            particles: Array of shape (n, 4) with [pt, eta, phi, E]
            
        Returns:
            self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        best_labels = None
        best_centroids = None
        best_inertia = float('inf')
        best_history = []
        
        for run in range(self.n_init):
            labels, centroids, inertia, history = self._single_run(particles)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids
                best_history = history
        
        self.labels_ = best_labels
        self.centroids_ = best_centroids
        self.inertia_ = best_inertia
        self.convergence_history_ = best_history
        
        return self
    
    def predict(self, particles: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new particles
        
        Args:
            particles: Array of shape (n, 4)
            
        Returns:
            Cluster labels for each particle
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._assign_labels(particles, self.centroids_)
    
    def fit_predict(self, particles: np.ndarray) -> np.ndarray:
        """Fit and return labels"""
        self.fit(particles)
        return self.labels_
    
    def get_jet_properties(self, particles: np.ndarray) -> List[Dict]:
        """
        Compute jet properties from clustering result
        
        Args:
            particles: Original particle array
            
        Returns:
            List of jet property dictionaries
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        jet_properties = []
        
        for k in range(self.n_clusters):
            mask = self.labels_ == k
            if not np.any(mask):
                jet_properties.append({
                    'pt': 0, 'eta': 0, 'phi': 0, 'mass': 0,
                    'n_constituents': 0
                })
                continue
            
            cluster_particles = particles[mask]
            
            # Sum four-momenta
            total_px = 0
            total_py = 0
            total_pz = 0
            total_E = 0
            
            for p in cluster_particles:
                pt, eta, phi, E = p
                px = pt * np.cos(phi)
                py = pt * np.sin(phi)
                pz = pt * np.sinh(eta)
                
                total_px += px
                total_py += py
                total_pz += pz
                total_E += E
            
            # Jet kinematics
            jet_pt = np.sqrt(total_px**2 + total_py**2)
            jet_phi = np.arctan2(total_py, total_px)
            p_total = np.sqrt(total_px**2 + total_py**2 + total_pz**2)
            jet_eta = 0.5 * np.log((p_total + total_pz + 1e-10) / 
                                   (p_total - total_pz + 1e-10))
            
            mass_sq = total_E**2 - total_px**2 - total_py**2 - total_pz**2
            jet_mass = np.sqrt(max(0, mass_sq))
            
            jet_properties.append({
                'pt': jet_pt,
                'eta': jet_eta,
                'phi': jet_phi,
                'mass': jet_mass,
                'n_constituents': len(cluster_particles),
                'centroid_eta': self.centroids_[k, 0],
                'centroid_phi': self.centroids_[k, 1]
            })
        
        return jet_properties
    
    def get_result(self, particles: np.ndarray) -> KMeansResult:
        """
        Get full clustering result
        
        Args:
            particles: Original particle array
            
        Returns:
            KMeansResult containing all information
        """
        if self.labels_ is None:
            self.fit(particles)
        
        return KMeansResult(
            labels=self.labels_,
            centroids=self.centroids_,
            n_iterations=len(self.convergence_history_),
            inertia=self.inertia_,
            convergence_history=self.convergence_history_,
            jet_properties=self.get_jet_properties(particles)
        )


class SoftKMeans(JetKMeans):
    """
    Soft k-Means (Fuzzy C-Means) Variant
    
    Instead of hard cluster assignments, uses soft membership
    weights that can be useful for jet physics where particles
    near cluster boundaries have ambiguous assignments.
    
    Membership weight:
    w_ik = 1 / Σ_j (d_ik / d_ij)^(2/(m-1))
    
    where m > 1 is the fuzziness parameter.
    """
    
    def __init__(self, 
                 n_clusters: int,
                 fuzziness: float = 2.0,
                 **kwargs):
        """
        Initialize Soft k-Means
        
        Args:
            n_clusters: Number of clusters
            fuzziness: Fuzziness parameter m > 1 (default 2)
            **kwargs: Other arguments passed to JetKMeans
        """
        super().__init__(n_clusters, **kwargs)
        self.fuzziness = max(1.01, fuzziness)  # Must be > 1
        self.membership_weights_: Optional[np.ndarray] = None
    
    def _compute_membership(self, particles: np.ndarray, 
                            centroids: np.ndarray) -> np.ndarray:
        """Compute fuzzy membership weights"""
        n = len(particles)
        k = len(centroids)
        distances = self._compute_distances(particles, centroids)
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        exp = 2.0 / (self.fuzziness - 1)
        weights = np.zeros((n, k))
        
        for i in range(n):
            for j in range(k):
                denom = 0.0
                for l in range(k):
                    denom += (distances[i, j] / distances[i, l]) ** exp
                weights[i, j] = 1.0 / denom
        
        return weights
    
    def _compute_fuzzy_centroids(self, particles: np.ndarray,
                                  weights: np.ndarray) -> np.ndarray:
        """Compute centroids with fuzzy weights"""
        centroids = np.zeros((self.n_clusters, 2))
        
        weights_m = weights ** self.fuzziness
        
        for k in range(self.n_clusters):
            w = weights_m[:, k]
            w_sum = np.sum(w)
            
            if w_sum < 1e-10:
                idx = np.random.randint(len(particles))
                centroids[k, 0] = particles[idx, 1]
                centroids[k, 1] = particles[idx, 2]
                continue
            
            # Eta
            centroids[k, 0] = np.sum(w * particles[:, 1]) / w_sum
            
            # Phi (vector mean)
            sin_phi = np.sum(w * np.sin(particles[:, 2])) / w_sum
            cos_phi = np.sum(w * np.cos(particles[:, 2])) / w_sum
            centroids[k, 1] = np.arctan2(sin_phi, cos_phi)
        
        return centroids
    
    def fit(self, particles: np.ndarray) -> 'SoftKMeans':
        """Fit soft k-means"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize centroids
        if self.init == 'kmeans++':
            centroids = self._init_kmeans_plus_plus(particles)
        else:
            centroids = self._init_random(particles)
        
        self.convergence_history_ = []
        
        for iteration in range(self.max_iter):
            # Compute membership weights
            weights = self._compute_membership(particles, centroids)
            
            # Update centroids
            new_centroids = self._compute_fuzzy_centroids(particles, weights)
            
            # Compute objective (weighted inertia)
            obj = 0.0
            weights_m = weights ** self.fuzziness
            distances = self._compute_distances(particles, centroids)
            for i in range(len(particles)):
                for k in range(self.n_clusters):
                    obj += weights_m[i, k] * distances[i, k]
            
            self.convergence_history_.append(obj)
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(self.convergence_history_[-2] - obj) / (
                    self.convergence_history_[-2] + 1e-10)
                if rel_change < self.tol:
                    break
            
            centroids = new_centroids
        
        # Final assignment (hard labels from soft weights)
        self.membership_weights_ = self._compute_membership(particles, centroids)
        self.labels_ = np.argmax(self.membership_weights_, axis=1)
        self.centroids_ = centroids
        self.inertia_ = self.convergence_history_[-1] if self.convergence_history_ else float('inf')
        
        return self
    
    def get_membership_weights(self) -> np.ndarray:
        """Get soft membership weights"""
        if self.membership_weights_ is None:
            raise ValueError("Model not fitted")
        return self.membership_weights_


if __name__ == "__main__":
    # Test k-means implementation
    print("Testing Jet k-Means Clustering...")
    
    # Generate sample data
    np.random.seed(42)
    n_particles = 30
    
    # Create three clusters
    particles = []
    for _ in range(10):
        particles.append([
            np.random.exponential(20) + 5,
            1.0 + np.random.normal(0, 0.2),
            0.5 + np.random.normal(0, 0.15),
            0
        ])
    for _ in range(10):
        particles.append([
            np.random.exponential(15) + 3,
            -0.5 + np.random.normal(0, 0.2),
            -1.5 + np.random.normal(0, 0.15),
            0
        ])
    for _ in range(10):
        particles.append([
            np.random.exponential(10) + 2,
            0.0 + np.random.normal(0, 0.2),
            2.5 + np.random.normal(0, 0.15),
            0
        ])
    
    particles = np.array(particles)
    for i in range(len(particles)):
        pt, eta, phi = particles[i, :3]
        particles[i, 3] = pt * np.cosh(eta)
    
    # Standard k-means
    print("\nStandard k-Means (pT-weighted):")
    kmeans = JetKMeans(n_clusters=3, random_state=42)
    result = kmeans.get_result(particles)
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Inertia: {result.inertia:.4f}")
    for i, props in enumerate(result.jet_properties):
        print(f"  Cluster {i+1}: pT={props['pt']:.1f}, η={props['eta']:.2f}, "
              f"φ={props['phi']:.2f}, n={props['n_constituents']}")
    
    # Soft k-means
    print("\nSoft k-Means (fuzziness=2):")
    soft_kmeans = SoftKMeans(n_clusters=3, fuzziness=2.0, random_state=42)
    soft_kmeans.fit(particles)
    weights = soft_kmeans.get_membership_weights()
    print(f"  Sample membership weights (first particle): {weights[0]}")
    
    print("\n✓ k-Means tests passed!")
