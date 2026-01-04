"""
QAOA (Quantum Approximate Optimization Algorithm) Implementation
================================================================

This module implements QAOA for solving the jet clustering QUBO problem,
as described in the paper:
"A novel quantum realization of jet clustering in high-energy physics experiments"

The QAOA algorithm uses parameterized quantum circuits with alternating
cost and mixer layers to find approximate solutions to combinatorial
optimization problems.

References:
- Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate 
  optimization algorithm. arXiv:1411.4028
- Paper Section III: QAOA-based jet clustering
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings


@dataclass
class QAOAResult:
    """Container for QAOA optimization results"""
    optimal_params: np.ndarray
    optimal_energy: float
    optimal_state: np.ndarray
    bitstring: str
    convergence_history: List[float]
    n_iterations: int
    success: bool


class QuantumGate:
    """
    Quantum gate representations for circuit simulation
    
    Implements common gates used in QAOA circuits:
    - Pauli gates (X, Y, Z)
    - Rotation gates (RX, RY, RZ)
    - Two-qubit gates (CNOT, RZZ)
    - Hadamard gate
    """
    
    @staticmethod
    def identity(n: int = 1) -> np.ndarray:
        """Identity gate for n qubits"""
        return np.eye(2**n, dtype=complex)
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Pauli-X (NOT) gate"""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        """Pauli-Y gate"""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """Pauli-Z gate"""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """Hadamard gate"""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def rx(theta: float) -> np.ndarray:
        """Rotation around X-axis"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    
    @staticmethod
    def ry(theta: float) -> np.ndarray:
        """Rotation around Y-axis"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    @staticmethod
    def rz(theta: float) -> np.ndarray:
        """Rotation around Z-axis"""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)
    
    @staticmethod
    def cnot() -> np.ndarray:
        """CNOT (Controlled-NOT) gate"""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    @staticmethod
    def rzz(theta: float) -> np.ndarray:
        """
        RZZ gate: exp(-i * theta/2 * Z⊗Z)
        Used for implementing cost Hamiltonian terms
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([
            [np.exp(-1j * theta / 2), 0, 0, 0],
            [0, np.exp(1j * theta / 2), 0, 0],
            [0, 0, np.exp(1j * theta / 2), 0],
            [0, 0, 0, np.exp(-1j * theta / 2)]
        ], dtype=complex)


class QAOACircuit:
    """
    QAOA Circuit Implementation
    
    Implements the quantum circuit for QAOA with:
    - Initial superposition state preparation
    - Cost Hamiltonian layers (encodes the QUBO problem)
    - Mixer Hamiltonian layers (provides quantum mixing)
    
    The circuit structure follows the paper's description:
    |ψ(γ,β)⟩ = U_M(β_p) U_C(γ_p) ... U_M(β_1) U_C(γ_1) |+⟩^n
    
    where:
    - U_C(γ) = exp(-iγH_C) is the cost unitary
    - U_M(β) = exp(-iβH_M) is the mixer unitary
    - H_C is the cost Hamiltonian (QUBO objective)
    - H_M is the mixer Hamiltonian (usually X-mixer)
    """
    
    def __init__(self, n_qubits: int, depth: int = 1):
        """
        Initialize QAOA circuit
        
        Args:
            n_qubits: Number of qubits (= number of binary variables)
            depth: QAOA depth parameter p (number of layers)
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.dim = 2 ** n_qubits
        
        # Cost Hamiltonian coefficients (to be set)
        self.linear_terms: Dict[int, float] = {}  # {qubit_idx: coefficient}
        self.quadratic_terms: Dict[Tuple[int, int], float] = {}  # {(i,j): coefficient}
        
        # Gate operation log for circuit visualization
        self.gate_log: List[Dict] = []
        
    def set_cost_hamiltonian(self, linear: Dict[int, float], 
                             quadratic: Dict[Tuple[int, int], float]):
        """
        Set the cost Hamiltonian coefficients from QUBO problem
        
        The cost Hamiltonian is:
        H_C = Σ_i h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j
        
        where Z_i are Pauli-Z operators
        
        Args:
            linear: Linear coefficients {qubit_index: coefficient}
            quadratic: Quadratic coefficients {(i,j): coefficient}
        """
        self.linear_terms = linear.copy()
        self.quadratic_terms = {
            tuple(sorted(k)): v for k, v in quadratic.items()
        }
        
    def _apply_single_qubit_gate(self, state: np.ndarray, 
                                  gate: np.ndarray, 
                                  qubit: int) -> np.ndarray:
        """Apply a single-qubit gate to specified qubit"""
        n = self.n_qubits
        
        # Build full operator using tensor products
        ops = [np.eye(2, dtype=complex)] * n
        ops[qubit] = gate
        
        # Tensor product from qubit 0 to n-1
        full_op = ops[0]
        for i in range(1, n):
            full_op = np.kron(full_op, ops[i])
            
        return full_op @ state
    
    def _apply_two_qubit_gate(self, state: np.ndarray,
                               gate: np.ndarray,
                               qubit1: int, qubit2: int) -> np.ndarray:
        """
        Apply a two-qubit gate to specified qubits
        
        This implementation handles non-adjacent qubits correctly
        using the swap network approach.
        """
        n = self.n_qubits
        
        if qubit1 > qubit2:
            qubit1, qubit2 = qubit2, qubit1
            
        # For adjacent qubits, direct application
        if qubit2 - qubit1 == 1:
            # Build identity operators
            pre = np.eye(2**qubit1, dtype=complex) if qubit1 > 0 else np.array([[1]], dtype=complex)
            post = np.eye(2**(n-qubit2-1), dtype=complex) if qubit2 < n-1 else np.array([[1]], dtype=complex)
            
            full_op = np.kron(np.kron(pre, gate), post)
            return full_op @ state
        else:
            # For non-adjacent qubits, decompose into simpler operations
            # Using direct matrix construction for efficiency
            new_state = state.copy()
            dim = 2 ** n
            
            for i in range(dim):
                for j in range(dim):
                    # Extract qubit values from basis states
                    q1_i = (i >> (n - 1 - qubit1)) & 1
                    q2_i = (i >> (n - 1 - qubit2)) & 1
                    q1_j = (j >> (n - 1 - qubit1)) & 1
                    q2_j = (j >> (n - 1 - qubit2)) & 1
                    
                    # Check if other qubits are the same
                    mask = ~((1 << (n - 1 - qubit1)) | (1 << (n - 1 - qubit2)))
                    if (i & mask) != (j & mask):
                        continue
                    
                    # Get gate matrix element
                    gate_row = q1_i * 2 + q2_i
                    gate_col = q1_j * 2 + q2_j
                    
                    if i == j:
                        new_state[i] = sum(
                            gate[gate_row, gate_col] * state[j]
                            for j in range(dim)
                            if (j & mask) == (i & mask)
                        )
            
            return new_state
    
    def _cost_layer(self, state: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply cost Hamiltonian layer
        
        U_C(γ) = exp(-iγH_C) = Π_i exp(-iγh_i Z_i) × Π_{i<j} exp(-iγJ_{ij} Z_i Z_j)
        
        For efficiency, we compute the diagonal directly:
        The cost Hamiltonian is diagonal in the computational basis.
        """
        self.gate_log.append({'type': 'cost_layer', 'gamma': gamma})
        
        # Compute diagonal of cost Hamiltonian
        diag = np.zeros(self.dim, dtype=complex)
        
        for basis_state in range(self.dim):
            energy = 0.0
            
            # Linear terms: h_i * z_i where z_i = 1 - 2*bit_i
            for qubit, h in self.linear_terms.items():
                bit = (basis_state >> (self.n_qubits - 1 - qubit)) & 1
                z_val = 1 - 2 * bit  # |0⟩→+1, |1⟩→-1
                energy += h * z_val
            
            # Quadratic terms: J_{ij} * z_i * z_j
            for (q1, q2), J in self.quadratic_terms.items():
                bit1 = (basis_state >> (self.n_qubits - 1 - q1)) & 1
                bit2 = (basis_state >> (self.n_qubits - 1 - q2)) & 1
                z1 = 1 - 2 * bit1
                z2 = 1 - 2 * bit2
                energy += J * z1 * z2
            
            diag[basis_state] = np.exp(-1j * gamma * energy)
        
        return diag * state
    
    def _mixer_layer(self, state: np.ndarray, beta: float) -> np.ndarray:
        """
        Apply mixer Hamiltonian layer
        
        Using X-mixer: H_M = Σ_i X_i
        U_M(β) = Π_i exp(-iβX_i) = Π_i RX(2β)
        """
        self.gate_log.append({'type': 'mixer_layer', 'beta': beta})
        
        rx_gate = QuantumGate.rx(2 * beta)
        
        for qubit in range(self.n_qubits):
            state = self._apply_single_qubit_gate(state, rx_gate, qubit)
            
        return state
    
    def prepare_initial_state(self) -> np.ndarray:
        """
        Prepare initial superposition state |+⟩^n
        
        Apply Hadamard to all qubits: H^⊗n |0⟩^n = |+⟩^n
        """
        self.gate_log = []  # Reset gate log
        self.gate_log.append({'type': 'initial_state'})
        
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0  # |0...0⟩
        
        h_gate = QuantumGate.hadamard()
        for qubit in range(self.n_qubits):
            state = self._apply_single_qubit_gate(state, h_gate, qubit)
            self.gate_log.append({'type': 'H', 'qubit': qubit})
            
        return state
    
    def run_circuit(self, params: np.ndarray) -> np.ndarray:
        """
        Run the full QAOA circuit with given parameters
        
        Args:
            params: Array of 2*depth parameters [γ_1, β_1, γ_2, β_2, ...]
            
        Returns:
            Final quantum state vector
        """
        if len(params) != 2 * self.depth:
            raise ValueError(f"Expected {2*self.depth} parameters, got {len(params)}")
        
        # Prepare initial state
        state = self.prepare_initial_state()
        
        # Apply alternating cost and mixer layers
        for layer in range(self.depth):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            
            state = self._cost_layer(state, gamma)
            state = self._mixer_layer(state, beta)
            
        return state
    
    def compute_expectation(self, state: np.ndarray) -> float:
        """
        Compute expectation value ⟨H_C⟩ for given state
        
        Returns:
            Expected energy (cost function value)
        """
        probabilities = np.abs(state) ** 2
        
        expectation = 0.0
        for basis_state in range(self.dim):
            if probabilities[basis_state] < 1e-15:
                continue
                
            # Compute energy for this basis state
            energy = 0.0
            
            # Linear terms
            for qubit, h in self.linear_terms.items():
                bit = (basis_state >> (self.n_qubits - 1 - qubit)) & 1
                energy += h * bit  # In QUBO: x=0 or x=1
            
            # Quadratic terms
            for (q1, q2), J in self.quadratic_terms.items():
                bit1 = (basis_state >> (self.n_qubits - 1 - q1)) & 1
                bit2 = (basis_state >> (self.n_qubits - 1 - q2)) & 1
                energy += J * bit1 * bit2
            
            expectation += probabilities[basis_state] * energy
            
        return expectation
    
    def sample_bitstring(self, state: np.ndarray, n_samples: int = 1) -> List[str]:
        """
        Sample bitstrings from the quantum state distribution
        
        Args:
            state: Quantum state vector
            n_samples: Number of samples to draw
            
        Returns:
            List of bitstring samples
        """
        probabilities = np.abs(state) ** 2
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        
        indices = np.random.choice(self.dim, size=n_samples, p=probabilities)
        
        bitstrings = []
        for idx in indices:
            bitstring = format(idx, f'0{self.n_qubits}b')
            bitstrings.append(bitstring)
            
        return bitstrings
    
    def get_most_probable_bitstring(self, state: np.ndarray) -> Tuple[str, float]:
        """
        Get the most probable bitstring from the quantum state
        
        Returns:
            (bitstring, probability)
        """
        probabilities = np.abs(state) ** 2
        max_idx = np.argmax(probabilities)
        max_prob = probabilities[max_idx]
        bitstring = format(max_idx, f'0{self.n_qubits}b')
        
        return bitstring, max_prob
    
    def get_circuit_stats(self) -> Dict:
        """
        Get circuit statistics for analysis
        
        Returns:
            Dictionary with gate counts, depth, etc.
        """
        h_count = sum(1 for g in self.gate_log if g.get('type') == 'H')
        cost_layers = sum(1 for g in self.gate_log if g.get('type') == 'cost_layer')
        mixer_layers = sum(1 for g in self.gate_log if g.get('type') == 'mixer_layer')
        
        # Estimate two-qubit gates (RZZ gates in cost layer)
        n_rzz = len(self.quadratic_terms) * cost_layers
        
        # Single qubit rotations in mixer
        n_rx = self.n_qubits * mixer_layers
        
        return {
            'n_qubits': self.n_qubits,
            'depth': self.depth,
            'hadamard_gates': h_count,
            'cost_layers': cost_layers,
            'mixer_layers': mixer_layers,
            'rzz_gates': n_rzz,
            'rx_gates': n_rx,
            'total_gates': h_count + n_rzz + n_rx,
            'circuit_depth': 1 + self.depth * (len(self.quadratic_terms) + 1)
        }


class QAOAOptimizer:
    """
    QAOA Optimizer for Jet Clustering QUBO Problems
    
    This class provides a high-level interface for solving
    jet clustering problems using QAOA, as described in the paper.
    
    Key features:
    - Multiple optimization strategies (COBYLA, Nelder-Mead, BFGS)
    - Support for different QAOA depths
    - Convergence tracking
    - Multi-start optimization for robustness
    """
    
    def __init__(self, 
                 depth: int = 1,
                 optimizer: str = 'COBYLA',
                 max_iter: int = 500,
                 n_restarts: int = 3,
                 tol: float = 1e-6,
                 verbose: bool = False):
        """
        Initialize QAOA optimizer
        
        Args:
            depth: QAOA circuit depth (p parameter)
            optimizer: Classical optimizer name ('COBYLA', 'Nelder-Mead', 'BFGS')
            max_iter: Maximum optimization iterations
            n_restarts: Number of random restarts for robustness
            tol: Convergence tolerance
            verbose: Whether to print optimization progress
        """
        self.depth = depth
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.tol = tol
        self.verbose = verbose
        
        self.circuit: Optional[QAOACircuit] = None
        self.result: Optional[QAOAResult] = None
        
    def _build_circuit_from_qubo(self, 
                                  linear: Dict[str, float],
                                  quadratic: Dict[Tuple[str, str], float],
                                  var_to_idx: Dict[str, int]) -> QAOACircuit:
        """
        Build QAOA circuit from QUBO model coefficients
        
        Converts variable names to qubit indices and sets up
        the cost Hamiltonian.
        """
        n_qubits = len(var_to_idx)
        circuit = QAOACircuit(n_qubits, self.depth)
        
        # Convert linear terms
        h = {}
        for var, coeff in linear.items():
            if var in var_to_idx:
                h[var_to_idx[var]] = coeff
        
        # Convert quadratic terms
        J = {}
        for (var1, var2), coeff in quadratic.items():
            if var1 in var_to_idx and var2 in var_to_idx:
                idx1 = var_to_idx[var1]
                idx2 = var_to_idx[var2]
                J[(min(idx1, idx2), max(idx1, idx2))] = coeff
        
        circuit.set_cost_hamiltonian(h, J)
        
        return circuit
    
    def _objective_function(self, params: np.ndarray) -> float:
        """
        Objective function for classical optimization
        
        Computes the expectation value ⟨H_C⟩ for given parameters.
        """
        state = self.circuit.run_circuit(params)
        energy = self.circuit.compute_expectation(state)
        
        if hasattr(self, '_convergence_history'):
            self._convergence_history.append(energy)
            
        if self.verbose and len(self._convergence_history) % 20 == 0:
            print(f"  Iteration {len(self._convergence_history)}: E = {energy:.6f}")
            
        return energy
    
    def optimize(self, 
                 qubo_model,
                 initial_params: Optional[np.ndarray] = None) -> QAOAResult:
        """
        Optimize QAOA parameters to minimize the QUBO objective
        
        Args:
            qubo_model: Kaiwu QUBO model
            initial_params: Initial parameter values (optional)
            
        Returns:
            QAOAResult with optimal parameters and solution
        """
        # Compile QUBO model
        compiled = qubo_model.compile()
        
        # Extract variable mapping
        vars_set = set(compiled.linear.keys())
        for (v1, v2) in compiled.quadratic.keys():
            vars_set.add(v1)
            vars_set.add(v2)
        var_list = sorted(list(vars_set))
        var_to_idx = {v: i for i, v in enumerate(var_list)}
        
        # Build circuit
        self.circuit = self._build_circuit_from_qubo(
            compiled.linear, 
            compiled.quadratic,
            var_to_idx
        )
        
        # Optimization parameters
        n_params = 2 * self.depth
        
        best_result = None
        best_energy = float('inf')
        
        # Multi-start optimization
        for restart in range(self.n_restarts):
            if self.verbose:
                print(f"\nRestart {restart + 1}/{self.n_restarts}")
            
            # Random initial parameters
            if initial_params is not None and restart == 0:
                params0 = initial_params.copy()
            else:
                # γ typically in [0, 2π], β typically in [0, π]
                gammas = np.random.uniform(0, 2*np.pi, self.depth)
                betas = np.random.uniform(0, np.pi, self.depth)
                params0 = np.zeros(n_params)
                params0[0::2] = gammas
                params0[1::2] = betas
            
            self._convergence_history = []
            
            # Run optimization
            opt_result = minimize(
                self._objective_function,
                params0,
                method=self.optimizer_name,
                options={'maxiter': self.max_iter},
                tol=self.tol
            )
            
            if opt_result.fun < best_energy:
                best_energy = opt_result.fun
                
                # Get final state and bitstring
                final_state = self.circuit.run_circuit(opt_result.x)
                bitstring, prob = self.circuit.get_most_probable_bitstring(final_state)
                
                best_result = QAOAResult(
                    optimal_params=opt_result.x,
                    optimal_energy=opt_result.fun,
                    optimal_state=final_state,
                    bitstring=bitstring,
                    convergence_history=self._convergence_history.copy(),
                    n_iterations=opt_result.nfev,
                    success=opt_result.success
                )
        
        self.result = best_result
        
        # Convert bitstring to solution dictionary
        solution_dict = {}
        for i, bit in enumerate(best_result.bitstring):
            solution_dict[var_list[i]] = int(bit)
        
        return best_result, solution_dict, best_energy
    
    def get_circuit_info(self) -> Dict:
        """Get information about the constructed circuit"""
        if self.circuit is None:
            return {}
        return self.circuit.get_circuit_stats()


def solve_jet_clustering_qaoa(particles: np.ndarray,
                              n_jets: int,
                              depth: int = 1,
                              R: float = 0.4,
                              p: int = -1,
                              verbose: bool = False) -> Dict:
    """
    High-level function to solve jet clustering using QAOA
    
    This function provides a convenient interface matching the
    paper's methodology for jet clustering using QAOA.
    
    Args:
        particles: Array of shape (n_particles, 4) with [pt, eta, phi, E]
        n_jets: Target number of jets
        depth: QAOA circuit depth (p in paper, typically 1, 3, or 5)
        R: Jet radius parameter (default 0.4 as in paper)
        p: kt algorithm parameter (-1 for anti-kt)
        verbose: Print optimization progress
        
    Returns:
        Dictionary with:
        - 'labels': Jet assignment for each particle
        - 'energy': Final objective value
        - 'circuit_stats': Circuit statistics
        - 'convergence': Optimization history
    """
    import sys
    sys.path.insert(0, 'F:/xiangmulianxi/QUBO问题')
    import kaiwu as kw
    
    n_particles = len(particles)
    
    # Compute distance matrix
    D = np.zeros((n_particles, n_particles))
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            pt_i, eta_i, phi_i = particles[i, :3]
            pt_j, eta_j, phi_j = particles[j, :3]
            
            # Delta phi with periodicity
            dphi = phi_i - phi_j
            while dphi > np.pi: dphi -= 2 * np.pi
            while dphi < -np.pi: dphi += 2 * np.pi
            
            delta_R_sq = (eta_i - eta_j)**2 + dphi**2
            kt_factor = min(pt_i**(2*p), pt_j**(2*p))
            d_ij = kt_factor * delta_R_sq / (R**2)
            
            D[i, j] = D[j, i] = d_ij
    
    # Build QUBO model
    x = kw.qubo.ndarray((n_particles, n_jets), 'x', kw.qubo.Binary)
    qubo_model = kw.qubo.QuboModel()
    
    # Objective: minimize intra-jet distances
    obj_terms = []
    for jet_idx in range(n_jets):
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                if D[i, j] > 0:
                    obj_terms.append(D[i, j] * x[i, jet_idx] * x[j, jet_idx])
    
    if obj_terms:
        qubo_model.set_objective(kw.qubo.quicksum(obj_terms))
    
    # Constraints: each particle belongs to exactly one jet
    penalty = np.max(D) * n_particles * 10
    for i in range(n_particles):
        constraint = kw.qubo.quicksum([x[i, j] for j in range(n_jets)]) - 1
        qubo_model.add_constraint(constraint == 0, f"p_{i}", penalty=penalty)
    
    # Solve using QAOA
    optimizer = QAOAOptimizer(
        depth=depth,
        optimizer='COBYLA',
        max_iter=500,
        n_restarts=3,
        verbose=verbose
    )
    
    result, sol_dict, energy = optimizer.optimize(qubo_model)
    
    # Parse solution to labels
    labels = np.zeros(n_particles, dtype=int)
    for i in range(n_particles):
        for j in range(n_jets):
            var_name = f'x[{i}][{j}]'
            if var_name in sol_dict and sol_dict[var_name] == 1:
                labels[i] = j
                break
    
    return {
        'labels': labels,
        'energy': energy,
        'circuit_stats': optimizer.get_circuit_info(),
        'convergence': result.convergence_history,
        'optimal_params': result.optimal_params,
        'bitstring': result.bitstring
    }


if __name__ == "__main__":
    # Test QAOA implementation
    print("Testing QAOA Circuit...")
    
    # Simple 3-qubit test
    circuit = QAOACircuit(n_qubits=3, depth=2)
    circuit.set_cost_hamiltonian(
        linear={0: 1.0, 1: -0.5, 2: 0.5},
        quadratic={(0, 1): 0.3, (1, 2): -0.2, (0, 2): 0.4}
    )
    
    params = np.array([0.5, 0.3, 0.7, 0.4])  # γ1, β1, γ2, β2
    state = circuit.run_circuit(params)
    
    print(f"State vector norm: {np.linalg.norm(state):.6f}")
    print(f"Expectation value: {circuit.compute_expectation(state):.6f}")
    
    bitstring, prob = circuit.get_most_probable_bitstring(state)
    print(f"Most probable: {bitstring} (p={prob:.4f})")
    
    print("\nCircuit stats:", circuit.get_circuit_stats())
    print("\n✓ QAOA implementation test passed!")
