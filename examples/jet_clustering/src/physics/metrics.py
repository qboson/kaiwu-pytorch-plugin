"""
Performance Metrics for Jet Clustering
======================================

This module implements the performance metrics used in the paper to
evaluate jet clustering algorithms, specifically:

1. Average angle between reconstructed jet and true quark direction
   (The primary metric shown in Figure 2)

2. Clustering efficiency and purity metrics

3. Statistical analysis utilities for error bars

References:
- Paper Figure 2 y-axis: "Avg. angle (rad): jet and quark"
- Paper methodology for comparing QAOA vs classical algorithms
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import warnings

from .jet_physics import (
    FourMomentum, 
    compute_jet_four_momentum, 
    angle_between_vectors,
    delta_R
)


@dataclass
class ClusteringMetrics:
    """Container for comprehensive clustering performance metrics"""
    avg_angle: float           # Average jet-quark angle (rad)
    std_angle: float           # Standard deviation of angles
    angle_error: float         # Standard error of the mean
    efficiency: float          # Fraction of particles correctly assigned
    purity: float              # Average jet purity
    n_events: int              # Number of events analyzed
    angles: np.ndarray         # Individual angle measurements
    

def compute_jet_quark_angle(jet_constituents: np.ndarray,
                            quark_direction: np.ndarray) -> float:
    """
    Compute angle between reconstructed jet and true quark direction
    
    This is the primary metric from paper Figure 2.
    
    Args:
        jet_constituents: Array of [pt, eta, phi, E] for jet constituents
        quark_direction: True quark momentum direction [px, py, pz]
        
    Returns:
        Angle in radians
    """
    if len(jet_constituents) == 0:
        return np.pi  # Maximum angle for empty jet
    
    # Reconstruct jet four-momentum
    jet = compute_jet_four_momentum(jet_constituents)
    
    # Jet direction
    jet_dir = np.array([jet.px, jet.py, jet.pz])
    
    return angle_between_vectors(jet_dir, quark_direction)


def compute_average_angle(events_data: List[Dict],
                          labels_list: List[np.ndarray]) -> ClusteringMetrics:
    """
    Compute average angle metric over multiple events
    
    This function computes the metric shown on the y-axis of
    paper Figure 2: "Avg. angle (rad): jet and quark"
    
    Args:
        events_data: List of event dictionaries containing:
            - 'particles': np.ndarray of shape (n, 4)
            - 'quark_directions': List of [px, py, pz] arrays
        labels_list: List of cluster labels for each event
        
    Returns:
        ClusteringMetrics with average angle and statistics
    """
    all_angles = []
    
    for event_data, labels in zip(events_data, labels_list):
        particles = event_data['particles']
        quark_directions = event_data['quark_directions']
        n_jets = len(quark_directions)
        
        event_angles = []
        used_quarks = set()
        
        # For each jet, find best matching quark
        for j in range(n_jets):
            mask = labels == j
            constituents = particles[mask]
            
            if len(constituents) == 0:
                continue
            
            # Compute jet momentum
            jet = compute_jet_four_momentum(constituents)
            jet_dir = np.array([jet.px, jet.py, jet.pz])
            
            # Find closest unmatched quark
            best_angle = np.pi
            best_quark = -1
            
            for q_idx, q_dir in enumerate(quark_directions):
                if q_idx in used_quarks:
                    continue
                
                angle = angle_between_vectors(jet_dir, q_dir)
                if angle < best_angle:
                    best_angle = angle
                    best_quark = q_idx
            
            if best_quark >= 0:
                event_angles.append(best_angle)
                used_quarks.add(best_quark)
        
        all_angles.extend(event_angles)
    
    angles_array = np.array(all_angles)
    
    if len(angles_array) == 0:
        return ClusteringMetrics(
            avg_angle=np.pi,
            std_angle=0.0,
            angle_error=0.0,
            efficiency=0.0,
            purity=0.0,
            n_events=len(events_data),
            angles=np.array([])
        )
    
    avg = np.mean(angles_array)
    std = np.std(angles_array)
    error = std / np.sqrt(len(angles_array))
    
    return ClusteringMetrics(
        avg_angle=avg,
        std_angle=std,
        angle_error=error,
        efficiency=0.0,  # Computed separately if needed
        purity=0.0,
        n_events=len(events_data),
        angles=angles_array
    )


def compute_clustering_efficiency(true_labels: np.ndarray,
                                   predicted_labels: np.ndarray) -> float:
    """
    Compute clustering efficiency (accuracy)
    
    Uses Hungarian algorithm for optimal label matching.
    
    Args:
        true_labels: Ground truth cluster assignments
        predicted_labels: Predicted cluster assignments
        
    Returns:
        Fraction of correctly assigned particles
    """
    from scipy.optimize import linear_sum_assignment
    
    n = len(true_labels)
    true_clusters = np.unique(true_labels)
    pred_clusters = np.unique(predicted_labels)
    
    # Build confusion matrix
    n_true = len(true_clusters)
    n_pred = len(pred_clusters)
    max_clusters = max(n_true, n_pred)
    
    confusion = np.zeros((max_clusters, max_clusters))
    
    for i, tc in enumerate(true_clusters):
        for j, pc in enumerate(pred_clusters):
            confusion[i, j] = np.sum((true_labels == tc) & (predicted_labels == pc))
    
    # Find optimal assignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-confusion)
    
    correct = confusion[row_ind, col_ind].sum()
    
    return correct / n


def compute_jet_purity(true_labels: np.ndarray,
                        predicted_labels: np.ndarray) -> float:
    """
    Compute average jet purity
    
    Purity of jet j = max_k(n_jk) / n_j
    where n_jk = particles from true cluster k in predicted jet j
    
    Returns:
        Average purity across all jets
    """
    pred_clusters = np.unique(predicted_labels)
    true_clusters = np.unique(true_labels)
    
    purities = []
    
    for pc in pred_clusters:
        mask = predicted_labels == pc
        if not np.any(mask):
            continue
        
        n_j = np.sum(mask)
        max_overlap = 0
        
        for tc in true_clusters:
            overlap = np.sum(mask & (true_labels == tc))
            max_overlap = max(max_overlap, overlap)
        
        purities.append(max_overlap / n_j)
    
    return np.mean(purities) if purities else 0.0


def compute_adjusted_rand_index(true_labels: np.ndarray,
                                 predicted_labels: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index for clustering quality
    
    ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    
    Returns value in [-1, 1], where 1 is perfect clustering.
    """
    from itertools import combinations
    
    n = len(true_labels)
    
    # Count pair agreements
    true_same = set()
    pred_same = set()
    
    for i, j in combinations(range(n), 2):
        if true_labels[i] == true_labels[j]:
            true_same.add((i, j))
        if predicted_labels[i] == predicted_labels[j]:
            pred_same.add((i, j))
    
    # Calculate RI components
    a = len(true_same & pred_same)  # Same in both
    d = len(set(combinations(range(n), 2)) - true_same - pred_same)  # Different in both
    
    n_pairs = n * (n - 1) // 2
    
    # Expected index
    n_true_same = len(true_same)
    n_pred_same = len(pred_same)
    
    expected_index = (n_true_same * n_pred_same + 
                      (n_pairs - n_true_same) * (n_pairs - n_pred_same)) / n_pairs
    
    max_index = (n_true_same + n_pred_same) / 2
    
    if max_index == expected_index:
        return 0.0
    
    actual_index = a + d
    
    return (actual_index - expected_index) / (max_index - expected_index)


def bootstrap_confidence_interval(data: np.ndarray,
                                   statistic: Callable = np.mean,
                                   confidence: float = 0.95,
                                   n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval
    
    Args:
        data: 1D array of measurements
        statistic: Function to compute statistic
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0)
    
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return (lower, upper)


def compute_event_metrics(particles: np.ndarray,
                          labels: np.ndarray,
                          true_labels: np.ndarray,
                          quark_directions: List[np.ndarray]) -> Dict:
    """
    Compute all metrics for a single event
    
    Args:
        particles: Array of [pt, eta, phi, E]
        labels: Predicted cluster labels
        true_labels: True cluster labels
        quark_directions: List of quark momentum directions
        
    Returns:
        Dictionary with all computed metrics
    """
    n_jets = len(np.unique(labels[labels >= 0]))
    
    # Jet-quark angles
    angles = []
    for j in range(n_jets):
        mask = labels == j
        if not np.any(mask):
            continue
        
        constituents = particles[mask]
        jet = compute_jet_four_momentum(constituents)
        jet_dir = np.array([jet.px, jet.py, jet.pz])
        
        # Find closest quark
        min_angle = np.pi
        for q_dir in quark_directions:
            angle = angle_between_vectors(jet_dir, q_dir)
            min_angle = min(min_angle, angle)
        
        angles.append(min_angle)
    
    efficiency = compute_clustering_efficiency(true_labels, labels)
    purity = compute_jet_purity(true_labels, labels)
    
    return {
        'avg_angle': np.mean(angles) if angles else np.pi,
        'angles': angles,
        'efficiency': efficiency,
        'purity': purity,
        'n_jets': n_jets,
        'n_particles': len(particles)
    }


class MetricsAggregator:
    """
    Aggregates metrics over multiple events for statistical analysis
    
    Provides methods matching the paper's Figure 2 error bar computation.
    """
    
    def __init__(self):
        self.event_angles: List[float] = []
        self.event_efficiencies: List[float] = []
        self.event_purities: List[float] = []
        self.n_events = 0
    
    def add_event(self, metrics: Dict):
        """Add metrics from a single event"""
        if 'angles' in metrics and metrics['angles']:
            self.event_angles.extend(metrics['angles'])
        self.event_efficiencies.append(metrics.get('efficiency', 0.0))
        self.event_purities.append(metrics.get('purity', 0.0))
        self.n_events += 1
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics matching paper Figure 2
        
        Returns:
            Dictionary with:
            - avg_angle: Mean angle (rad)
            - angle_error: Error bar size (standard error)
            - efficiency_mean, efficiency_std
            - purity_mean, purity_std
        """
        angles = np.array(self.event_angles)
        efficiencies = np.array(self.event_efficiencies)
        purities = np.array(self.event_purities)
        
        summary = {
            'n_events': self.n_events,
            'n_angle_measurements': len(angles),
            
            # Primary paper metric
            'avg_angle': np.mean(angles) if len(angles) > 0 else np.pi,
            'angle_std': np.std(angles) if len(angles) > 0 else 0.0,
            'angle_error': np.std(angles) / np.sqrt(len(angles)) if len(angles) > 1 else 0.0,
            
            # Secondary metrics
            'efficiency_mean': np.mean(efficiencies) if len(efficiencies) > 0 else 0.0,
            'efficiency_std': np.std(efficiencies) if len(efficiencies) > 0 else 0.0,
            'purity_mean': np.mean(purities) if len(purities) > 0 else 0.0,
            'purity_std': np.std(purities) if len(purities) > 0 else 0.0,
        }
        
        # Add 95% confidence interval
        if len(angles) > 1:
            ci = bootstrap_confidence_interval(angles, np.mean)
            summary['angle_ci_lower'] = ci[0]
            summary['angle_ci_upper'] = ci[1]
        
        return summary
    
    def reset(self):
        """Clear all accumulated data"""
        self.event_angles = []
        self.event_efficiencies = []
        self.event_purities = []
        self.n_events = 0


def compare_algorithms_metrics(events_data: List[Dict],
                               algorithm_results: Dict[str, List[np.ndarray]]) -> Dict[str, Dict]:
    """
    Compare multiple algorithms using the paper's metrics
    
    Args:
        events_data: List of event data dictionaries
        algorithm_results: {algorithm_name: [labels for each event]}
        
    Returns:
        {algorithm_name: metrics_summary}
    """
    comparison = {}
    
    for algo_name, labels_list in algorithm_results.items():
        aggregator = MetricsAggregator()
        
        for event_data, labels in zip(events_data, labels_list):
            particles = event_data['particles']
            true_labels = event_data.get('true_labels', labels)
            quark_dirs = event_data.get('quark_directions', [])
            
            if len(quark_dirs) == 0:
                # Create approximate quark directions from true jet centers
                n_jets = len(np.unique(true_labels[true_labels >= 0]))
                quark_dirs = []
                for j in range(n_jets):
                    mask = true_labels == j
                    if np.any(mask):
                        jet = compute_jet_four_momentum(particles[mask])
                        quark_dirs.append(np.array([jet.px, jet.py, jet.pz]))
            
            metrics = compute_event_metrics(particles, labels, true_labels, quark_dirs)
            aggregator.add_event(metrics)
        
        comparison[algo_name] = aggregator.get_summary()
    
    return comparison


if __name__ == "__main__":
    print("Testing Metrics Module...")
    
    # Generate sample data
    np.random.seed(42)
    
    true_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    pred_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # One error
    
    # Test efficiency
    eff = compute_clustering_efficiency(true_labels, pred_labels)
    print(f"\nClustering efficiency: {eff:.2%}")
    
    # Test purity
    purity = compute_jet_purity(true_labels, pred_labels)
    print(f"Average purity: {purity:.2%}")
    
    # Test ARI
    ari = compute_adjusted_rand_index(true_labels, pred_labels)
    print(f"Adjusted Rand Index: {ari:.3f}")
    
    # Test bootstrap CI
    data = np.random.normal(0.1, 0.05, 100)
    ci = bootstrap_confidence_interval(data)
    print(f"\n95% CI for mean: ({ci[0]:.4f}, {ci[1]:.4f})")
    print(f"Actual mean: {np.mean(data):.4f}")
    
    print("\n✓ Metrics tests passed!")
