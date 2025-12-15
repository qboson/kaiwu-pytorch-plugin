"""
Paper Figure Reproduction Module
================================

This module provides functions to reproduce the exact visualizations
from the paper "A novel quantum realization of jet clustering in 
high-energy physics experiments" (Figure 2).

Figure 2 contains 5 sub-figures:
(a) QAOA depth comparison (k=6, depths 1,3,5)
(b) k-value comparison (depth=3, k=2,4,6,7,8)
(c) Algorithm comparison (QAOA vs e+e- kt vs k-Means)
(d) Quantum circuit diagram
(e) Quafu hardware comparison

All figures use the same y-axis metric:
"Avg. angle (rad): jet and quark"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple
import os


# Paper-accurate color scheme
COLORS = {
    'depth_1': '#3274A1',      # Blue
    'depth_3': '#E1812C',      # Orange  
    'depth_5': '#3A923A',      # Green
    'qaoa_sim': '#3274A1',     # Blue
    'kt_algorithm': '#E1812C', # Orange
    'kmeans': '#3A923A',       # Green
    'quafu': '#3274A1',        # Blue (hardware)
}


def setup_paper_style():
    """Configure matplotlib for paper-quality figures"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': False,
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
    })


def plot_figure2a(results: Dict[int, Dict], 
                  save_path: Optional[str] = None,
                  show: bool = True) -> plt.Figure:
    """
    Reproduce Figure 2(a): QAOA depth comparison
    
    Shows jet clustering performance with k=6 and QAOA depths 1, 3, 5.
    
    Args:
        results: {depth: {'avg_angle': float, 'error': float}}
        save_path: Path to save figure
        show: Whether to display figure
        
    Returns:
        matplotlib Figure object
    """
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    depths = [1, 3, 5]
    colors = [COLORS['depth_1'], COLORS['depth_3'], COLORS['depth_5']]
    
    x_positions = np.arange(len(depths))
    bar_width = 0.5
    
    angles = [results.get(d, {}).get('avg_angle', 0.1) for d in depths]
    errors = [results.get(d, {}).get('error', 0.01) for d in depths]
    
    bars = ax.bar(x_positions, angles, bar_width, 
                  color=colors, edgecolor='black', linewidth=0.8,
                  yerr=errors, capsize=4, error_kw={'linewidth': 1.2})
    
    ax.set_ylabel('Avg. angle (rad): jet and quark', fontsize=12)
    ax.set_xlabel('(a)', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    
    ax.set_ylim(0, 0.16)
    ax.set_yticks(np.arange(0, 0.16, 0.02))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['depth_1'], edgecolor='black', label='depth = 1'),
        mpatches.Patch(facecolor=COLORS['depth_3'], edgecolor='black', label='depth = 3'),
        mpatches.Patch(facecolor=COLORS['depth_5'], edgecolor='black', label='depth = 5'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig


def plot_figure2b(results: Dict[int, Dict],
                  save_path: Optional[str] = None,
                  show: bool = True) -> plt.Figure:
    """
    Reproduce Figure 2(b): k-value comparison
    
    Shows performance with QAOA depth=3 and k values 2, 4, 6, 7, 8.
    
    Args:
        results: {k: {'avg_angle': float, 'error': float}}
    """
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    k_values = [2, 4, 6, 7, 8]
    colors = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E', '#9372B2']
    
    x_positions = np.arange(len(k_values))
    bar_width = 0.6
    
    angles = [results.get(k, {}).get('avg_angle', 0.1) for k in k_values]
    errors = [results.get(k, {}).get('error', 0.01) for k in k_values]
    
    bars = ax.bar(x_positions, angles, bar_width,
                  color=colors, edgecolor='black', linewidth=0.8,
                  yerr=errors, capsize=4, error_kw={'linewidth': 1.2})
    
    ax.set_ylabel('Avg. angle (rad): jet and quark', fontsize=12)
    ax.set_xlabel('(b)', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    
    ax.set_ylim(0, 0.18)
    ax.set_yticks(np.arange(0, 0.18, 0.02))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[i], edgecolor='black', label=f'k = {k}')
        for i, k in enumerate(k_values)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, ncol=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig


def plot_figure2c(results: Dict[str, Dict],
                  save_path: Optional[str] = None,
                  show: bool = True) -> plt.Figure:
    """
    Reproduce Figure 2(c): Algorithm comparison
    
    Compares QAOA simulation, e+e- kt algorithm, and k-Means.
    
    Args:
        results: {algorithm_name: {'avg_angle': float, 'error': float}}
    """
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    algorithms = ['QAOA simulation', 'e⁺e⁻ kₜ', 'k-Means']
    alg_keys = ['qaoa', 'kt', 'kmeans']
    colors = [COLORS['qaoa_sim'], COLORS['kt_algorithm'], COLORS['kmeans']]
    
    x_positions = np.arange(len(algorithms))
    bar_width = 0.5
    
    angles = [results.get(k, {}).get('avg_angle', 0.1) for k in alg_keys]
    errors = [results.get(k, {}).get('error', 0.01) for k in alg_keys]
    
    bars = ax.bar(x_positions, angles, bar_width,
                  color=colors, edgecolor='black', linewidth=0.8,
                  yerr=errors, capsize=4, error_kw={'linewidth': 1.2})
    
    ax.set_ylabel('Avg. angle (rad): jet and quark', fontsize=12)
    ax.set_xlabel('(c)', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    
    ax.set_ylim(0, 0.20)
    ax.set_yticks(np.arange(0, 0.20, 0.025))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[i], edgecolor='black', label=alg)
        for i, alg in enumerate(algorithms)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig


def plot_figure2e(results: Dict[str, Dict],
                  save_path: Optional[str] = None,
                  show: bool = True) -> plt.Figure:
    """
    Reproduce Figure 2(e): Quafu quantum hardware comparison
    
    Shows comparison on real quantum hardware (1217 events, 6 particles, depth=1, k=2).
    
    Args:
        results: {algorithm: {'avg_angle': float, 'error': float}}
    """
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Paper shows 4 bars: Quafu, QAOA sim, e+e- kt, k-Means
    algorithms = ['Quafu quantum hardware', 'QAOA simulation', 'e⁺e⁻ kₜ', 'k-Means']
    alg_keys = ['quafu', 'qaoa', 'kt', 'kmeans']
    colors = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E']
    
    x_positions = np.arange(len(algorithms))
    bar_width = 0.6
    
    angles = [results.get(k, {}).get('avg_angle', 0.05) for k in alg_keys]
    errors = [results.get(k, {}).get('error', 0.005) for k in alg_keys]
    
    bars = ax.bar(x_positions, angles, bar_width,
                  color=colors, edgecolor='black', linewidth=0.8,
                  yerr=errors, capsize=4, error_kw={'linewidth': 1.2})
    
    ax.set_ylabel('Avg. angle (rad): jet and quark', fontsize=12)
    ax.set_xlabel('(e)', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    
    ax.set_ylim(0, 0.16)
    ax.set_yticks(np.arange(0, 0.16, 0.02))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[i], edgecolor='black', label=alg)
        for i, alg in enumerate(algorithms)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig


def plot_combined_figure2(results_a: Dict, results_b: Dict, 
                          results_c: Dict, results_e: Dict,
                          save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
    """
    Create combined Figure 2 with subplots (a), (b), (c), (e)
    
    Matches the paper's layout with all comparison charts.
    """
    setup_paper_style()
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid for subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    
    # (a) Top-left
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_depth_comparison(ax_a, results_a)
    
    # (b) Top-center
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_k_comparison(ax_b, results_b)
    
    # (c) Top-right
    ax_c = fig.add_subplot(gs[0, 2])
    _plot_algorithm_comparison(ax_c, results_c)
    
    # (e) Bottom-right
    ax_e = fig.add_subplot(gs[1, 2])
    _plot_hardware_comparison(ax_e, results_e)
    
    # Note for (d) - circuit will be separate
    ax_note = fig.add_subplot(gs[1, :2])
    ax_note.text(0.5, 0.5, '(d) See circuit_visualization.py for quantum circuit diagram',
                 ha='center', va='center', fontsize=12, style='italic',
                 transform=ax_note.transAxes)
    ax_note.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig


def _plot_depth_comparison(ax, results):
    """Helper for subplot (a)"""
    depths = [1, 3, 5]
    colors = [COLORS['depth_1'], COLORS['depth_3'], COLORS['depth_5']]
    
    angles = [results.get(d, {}).get('avg_angle', 0.1) for d in depths]
    errors = [results.get(d, {}).get('error', 0.01) for d in depths]
    
    ax.bar(range(len(depths)), angles, color=colors, edgecolor='black',
           yerr=errors, capsize=4)
    ax.set_ylabel('Avg. angle (rad): jet and quark')
    ax.set_xlabel('(a)', fontweight='bold')
    ax.set_xticks([])
    ax.set_ylim(0, 0.16)
    
    legend = [mpatches.Patch(color=c, label=f'depth = {d}') 
              for c, d in zip(colors, depths)]
    ax.legend(handles=legend, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_k_comparison(ax, results):
    """Helper for subplot (b)"""
    k_values = [2, 4, 6, 7, 8]
    colors = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E', '#9372B2']
    
    angles = [results.get(k, {}).get('avg_angle', 0.1) for k in k_values]
    errors = [results.get(k, {}).get('error', 0.01) for k in k_values]
    
    ax.bar(range(len(k_values)), angles, color=colors, edgecolor='black',
           yerr=errors, capsize=4)
    ax.set_ylabel('Avg. angle (rad): jet and quark')
    ax.set_xlabel('(b)', fontweight='bold')
    ax.set_xticks([])
    ax.set_ylim(0, 0.18)
    
    legend = [mpatches.Patch(color=c, label=f'k = {k}') 
              for c, k in zip(colors, k_values)]
    ax.legend(handles=legend, loc='upper right', ncol=2, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_algorithm_comparison(ax, results):
    """Helper for subplot (c)"""
    algs = ['qaoa', 'kt', 'kmeans']
    labels = ['QAOA simulation', 'e⁺e⁻ kₜ', 'k-Means']
    colors = [COLORS['qaoa_sim'], COLORS['kt_algorithm'], COLORS['kmeans']]
    
    angles = [results.get(a, {}).get('avg_angle', 0.1) for a in algs]
    errors = [results.get(a, {}).get('error', 0.01) for a in algs]
    
    ax.bar(range(len(algs)), angles, color=colors, edgecolor='black',
           yerr=errors, capsize=4)
    ax.set_ylabel('Avg. angle (rad): jet and quark')
    ax.set_xlabel('(c)', fontweight='bold')
    ax.set_xticks([])
    ax.set_ylim(0, 0.20)
    
    legend = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_hardware_comparison(ax, results):
    """Helper for subplot (e)"""
    algs = ['quafu', 'qaoa', 'kt', 'kmeans']
    labels = ['Quafu quantum hardware', 'QAOA simulation', 'e⁺e⁻ kₜ', 'k-Means']
    colors = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E']
    
    angles = [results.get(a, {}).get('avg_angle', 0.05) for a in algs]
    errors = [results.get(a, {}).get('error', 0.005) for a in algs]
    
    ax.bar(range(len(algs)), angles, color=colors, edgecolor='black',
           yerr=errors, capsize=4)
    ax.set_ylabel('Avg. angle (rad): jet and quark')
    ax.set_xlabel('(e)', fontweight='bold')
    ax.set_xticks([])
    ax.set_ylim(0, 0.16)
    
    legend = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend, loc='upper right', fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_eta_phi_clustering(particles: np.ndarray,
                            labels: np.ndarray,
                            jet_properties: Optional[List[Dict]] = None,
                            title: str = 'Jet Clustering Result',
                            save_path: Optional[str] = None,
                            show: bool = True) -> plt.Figure:
    """
    Visualize clustering in η-φ plane
    
    Args:
        particles: Array of [pt, eta, phi, E]
        labels: Cluster assignment for each particle
        jet_properties: Optional jet info for annotation
        title: Plot title
        save_path: Save path
        show: Display plot
    """
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    n_jets = len(np.unique(labels[labels >= 0]))
    colors = plt.cm.tab10(np.linspace(0, 1, n_jets))
    
    for jet_idx in range(n_jets):
        mask = labels == jet_idx
        if not np.any(mask):
            continue
        
        jet_particles = particles[mask]
        
        # Size proportional to pT
        sizes = jet_particles[:, 0] * 3
        
        ax.scatter(
            jet_particles[:, 1],  # eta
            jet_particles[:, 2],  # phi
            c=[colors[jet_idx]],
            s=sizes,
            label=f'Jet {jet_idx + 1}',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Mark jet center if provided
        if jet_properties and jet_idx < len(jet_properties):
            props = jet_properties[jet_idx]
            ax.scatter(
                props.get('eta', 0),
                props.get('phi', 0),
                c=[colors[jet_idx]],
                s=200,
                marker='*',
                edgecolors='black',
                linewidth=1
            )
    
    # Mark noise particles
    noise_mask = labels < 0
    if np.any(noise_mask):
        noise_particles = particles[noise_mask]
        ax.scatter(
            noise_particles[:, 1],
            noise_particles[:, 2],
            c='gray',
            s=10,
            alpha=0.3,
            label='Noise'
        )
    
    ax.set_xlabel('η (Pseudorapidity)', fontsize=12)
    ax.set_ylabel('φ (Azimuthal angle)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-np.pi, np.pi)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig


def plot_convergence(history: List[float],
                     algorithm: str = 'QAOA',
                     save_path: Optional[str] = None,
                     show: bool = True) -> plt.Figure:
    """Plot optimization convergence history"""
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(history, 'b-', linewidth=1.5, label=algorithm)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Energy / Objective', fontsize=12)
    ax.set_title(f'{algorithm} Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    print("Testing Paper Figure Reproduction...")
    
    # Generate sample results
    sample_results_a = {
        1: {'avg_angle': 0.12, 'error': 0.015},
        3: {'avg_angle': 0.08, 'error': 0.012},
        5: {'avg_angle': 0.05, 'error': 0.008},
    }
    
    sample_results_b = {
        2: {'avg_angle': 0.06, 'error': 0.008},
        4: {'avg_angle': 0.08, 'error': 0.010},
        6: {'avg_angle': 0.10, 'error': 0.012},
        7: {'avg_angle': 0.12, 'error': 0.014},
        8: {'avg_angle': 0.14, 'error': 0.016},
    }
    
    sample_results_c = {
        'qaoa': {'avg_angle': 0.08, 'error': 0.010},
        'kt': {'avg_angle': 0.12, 'error': 0.015},
        'kmeans': {'avg_angle': 0.16, 'error': 0.020},
    }
    
    # Test individual figures
    print("Generating Figure 2(a)...")
    plot_figure2a(sample_results_a, show=False)
    print("✓ Figure 2(a) generated")
    
    print("Generating Figure 2(b)...")
    plot_figure2b(sample_results_b, show=False)
    print("✓ Figure 2(b) generated")
    
    print("Generating Figure 2(c)...")
    plot_figure2c(sample_results_c, show=False)
    print("✓ Figure 2(c) generated")
    
    plt.close('all')
    print("\n✓ Paper figure reproduction tests passed!")
