"""
Quantum Circuit Visualization Module
===================================

This module provides visualization of QAOA quantum circuits,
reproducing Figure 2(d) from the paper which shows the compiled
quantum circuit for QAOA-based jet clustering.

The circuit diagram shows:
- Initial Hadamard gates on all qubits
- Alternating cost and mixer layers
- CNOT gates for entanglement
- Single-qubit rotation gates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrow
from matplotlib.lines import Line2D
from typing import List, Dict, Optional, Tuple
import matplotlib.patches as mpatches


class QuantumCircuitDrawer:
    """
    Draw quantum circuit diagrams for QAOA
    
    Creates publication-quality circuit diagrams matching
    the style used in the paper (Figure 2d).
    """
    
    def __init__(self,
                 n_qubits: int,
                 figsize: Tuple[float, float] = (14, 8),
                 wire_spacing: float = 1.0,
                 gate_spacing: float = 0.8):
        """
        Initialize circuit drawer
        
        Args:
            n_qubits: Number of qubits
            figsize: Figure size
            wire_spacing: Vertical spacing between qubit wires
            gate_spacing: Horizontal spacing between gates
        """
        self.n_qubits = n_qubits
        self.figsize = figsize
        self.wire_spacing = wire_spacing
        self.gate_spacing = gate_spacing
        
        self.fig = None
        self.ax = None
        self.current_x = 0.5
        
        # Gate styles
        self.gate_width = 0.4
        self.gate_height = 0.6
        
        # Colors
        self.colors = {
            'H': '#A8D8EA',      # Light blue
            'RX': '#FFB6B6',     # Light red
            'RZ': '#B5E8B5',     # Light green
            'CNOT': '#FFE5B4',   # Light orange
            'wire': '#333333',
            'text': 'black'
        }
    
    def _setup_figure(self, width: float):
        """Initialize figure and axes"""
        # Adjust figure width based on circuit depth
        actual_width = max(self.figsize[0], width + 2)
        height = self.n_qubits * self.wire_spacing + 2
        
        self.fig, self.ax = plt.subplots(figsize=(actual_width, height))
        self.ax.set_xlim(-0.5, width + 1)
        self.ax.set_ylim(-0.5, self.n_qubits * self.wire_spacing + 0.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
    
    def _draw_wires(self, length: float):
        """Draw qubit wires"""
        for i in range(self.n_qubits):
            y = i * self.wire_spacing
            self.ax.axhline(y, xmin=0, xmax=1, color=self.colors['wire'],
                           linewidth=1, zorder=0)
            
            # Qubit labels
            self.ax.text(-0.3, y, f'|q{i}⟩', ha='right', va='center',
                        fontsize=10, fontfamily='serif')
    
    def _draw_single_gate(self, qubit: int, x: float, 
                          label: str, color: str):
        """Draw a single-qubit gate"""
        y = qubit * self.wire_spacing
        
        rect = FancyBboxPatch(
            (x - self.gate_width/2, y - self.gate_height/2),
            self.gate_width, self.gate_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            zorder=10
        )
        self.ax.add_patch(rect)
        
        self.ax.text(x, y, label, ha='center', va='center',
                    fontsize=9, fontweight='bold', zorder=11)
    
    def _draw_cnot(self, control: int, target: int, x: float):
        """Draw a CNOT gate"""
        y_ctrl = control * self.wire_spacing
        y_targ = target * self.wire_spacing
        
        # Vertical line
        self.ax.plot([x, x], [y_ctrl, y_targ], 'k-', linewidth=1.5, zorder=5)
        
        # Control dot
        self.ax.plot(x, y_ctrl, 'ko', markersize=8, zorder=10)
        
        # Target circle with plus
        circle = Circle((x, y_targ), 0.15, facecolor='white',
                        edgecolor='black', linewidth=1.5, zorder=10)
        self.ax.add_patch(circle)
        
        # Plus sign
        self.ax.plot([x-0.1, x+0.1], [y_targ, y_targ], 'k-', linewidth=1.5, zorder=11)
        self.ax.plot([x, x], [y_targ-0.1, y_targ+0.1], 'k-', linewidth=1.5, zorder=11)
    
    def _draw_barrier(self, x: float):
        """Draw a dashed barrier line"""
        for i in range(self.n_qubits):
            y = i * self.wire_spacing
            self.ax.plot([x, x], [y - 0.3, y + 0.3], 'k--',
                        linewidth=1, alpha=0.5, zorder=5)
    
    def draw_qaoa_circuit(self, depth: int = 1,
                          n_cost_terms: int = 5,
                          save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
        """
        Draw a QAOA circuit diagram
        
        Reproduces the style of Figure 2(d) from the paper.
        
        Args:
            depth: QAOA depth (number of layers)
            n_cost_terms: Number of ZZ interactions in cost layer
            save_path: Path to save figure
            show: Display figure
        """
        # Estimate circuit width
        gates_per_layer = self.n_qubits + n_cost_terms + self.n_qubits
        total_width = 2 + depth * (gates_per_layer * self.gate_spacing + 1)
        
        self._setup_figure(total_width)
        
        # Draw wires first
        for i in range(self.n_qubits):
            y = i * self.wire_spacing
            line = Line2D([-0.1, total_width + 0.5], [y, y],
                         color=self.colors['wire'], linewidth=1, zorder=0)
            self.ax.add_line(line)
            
            # Input state labels
            self.ax.text(-0.3, y, f'|q{self.n_qubits - 1 - i}⟩', 
                        ha='right', va='center', fontsize=10)
            
            # Output measurement symbols (simplified)
            self.ax.text(total_width + 0.3, y, f'c{self.n_qubits - 1 - i}',
                        ha='left', va='center', fontsize=9, color='gray')
        
        x = 0.5
        
        # Initial Hadamard layer
        for i in range(self.n_qubits):
            self._draw_single_gate(i, x, 'H', self.colors['H'])
        x += self.gate_spacing
        
        self._draw_barrier(x)
        x += 0.3
        
        # QAOA layers
        for layer in range(depth):
            # Cost layer (ZZ interactions)
            layer_label_y = (self.n_qubits - 0.5) * self.wire_spacing + 0.3
            self.ax.text(x + n_cost_terms * self.gate_spacing / 2, layer_label_y,
                        f'Cost Layer {layer + 1}', ha='center', fontsize=9,
                        style='italic', color='gray')
            
            # Draw RZ and CNOT gates for cost Hamiltonian
            for term in range(n_cost_terms):
                q1 = term % self.n_qubits
                q2 = (term + 1) % self.n_qubits
                
                # RZ gates
                self._draw_single_gate(q1, x, 'RZ', self.colors['RZ'])
                x += self.gate_spacing * 0.5
                
                # CNOT
                self._draw_cnot(q1, q2, x)
                x += self.gate_spacing * 0.5
                
                # Another RZ
                self._draw_single_gate(q2, x, 'RZ', self.colors['RZ'])
                x += self.gate_spacing
            
            self._draw_barrier(x)
            x += 0.3
            
            # Mixer layer
            self.ax.text(x + self.n_qubits * self.gate_spacing / 2, layer_label_y,
                        f'Mixer Layer {layer + 1}', ha='center', fontsize=9,
                        style='italic', color='gray')
            
            for i in range(self.n_qubits):
                self._draw_single_gate(i, x, 'RX', self.colors['RX'])
            x += self.gate_spacing
            
            if layer < depth - 1:
                self._draw_barrier(x)
                x += 0.3
        
        # Title and annotations
        self.ax.set_title(
            f'QAOA Circuit for Jet Clustering\n'
            f'{self.n_qubits} qubits, depth={depth}',
            fontsize=12, fontweight='bold', pad=20
        )
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.colors['H'], edgecolor='black', 
                          label='Hadamard'),
            mpatches.Patch(facecolor=self.colors['RZ'], edgecolor='black',
                          label='RZ (γ)'),
            mpatches.Patch(facecolor=self.colors['RX'], edgecolor='black',
                          label='RX (β)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                  markersize=8, label='CNOT'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right',
                      frameon=True, fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300,
                       facecolor='white')
        
        if show:
            plt.show()
        
        return self.fig


def draw_simplified_circuit(n_qubits: int = 6,
                            depth: int = 1,
                            save_path: Optional[str] = None,
                            show: bool = True) -> plt.Figure:
    """
    Draw a simplified QAOA circuit matching paper Figure 2(d)
    
    The paper shows a 6-qubit circuit with 34 CNOT gates,
    27 single-qubit gates, and depth 27.
    """
    drawer = QuantumCircuitDrawer(n_qubits=n_qubits)
    return drawer.draw_qaoa_circuit(depth=depth, n_cost_terms=5,
                                     save_path=save_path, show=show)


def plot_circuit_statistics(circuit_stats: Dict,
                            save_path: Optional[str] = None,
                            show: bool = True) -> plt.Figure:
    """
    Plot circuit statistics as a bar chart
    
    Shows gate counts, depth, etc.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Extract statistics
    metrics = ['n_qubits', 'depth', 'hadamard_gates', 'rzz_gates', 'rx_gates']
    labels = ['Qubits', 'Depth', 'H gates', 'RZZ gates', 'RX gates']
    values = [circuit_stats.get(m, 0) for m in metrics]
    
    colors = ['#3274A1', '#E1812C', '#A8D8EA', '#B5E8B5', '#FFB6B6']
    
    bars = ax.bar(labels, values, color=colors, edgecolor='black')
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('QAOA Circuit Statistics', fontsize=14)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(int(val)), ha='center', va='bottom', fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    print("Testing Quantum Circuit Visualization...")
    
    # Draw sample QAOA circuit
    print("\nDrawing 6-qubit QAOA circuit (depth=1)...")
    draw_simplified_circuit(n_qubits=6, depth=1, show=False)
    print("✓ Circuit diagram created")
    
    # Test statistics plot
    print("\nPlotting circuit statistics...")
    stats = {
        'n_qubits': 6,
        'depth': 27,
        'hadamard_gates': 6,
        'rzz_gates': 34,
        'rx_gates': 27
    }
    plot_circuit_statistics(stats, show=False)
    print("✓ Statistics plot created")
    
    plt.close('all')
    print("\n✓ Circuit visualization tests passed!")
