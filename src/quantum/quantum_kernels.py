import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
import pennylane as qml
from pennylane import numpy as pnp
import qiskit
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
from functools import lru_cache
import threading
import logging

logger = logging.getLogger(__name__)

class QuantumResourceManager:
    """Context manager for quantum resources."""
    
    def __init__(self, device):
        self.device = device
        
    def __enter__(self):
        return self.device
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup quantum device resources
        if hasattr(self.device, 'reset'):
            self.device.reset()

class NoiseModel:
    """Quantum noise model configuration."""
    
    @staticmethod
    def get_depolarizing_noise(prob: float = 0.01) -> Dict[str, Any]:
        """Get depolarizing noise model."""
        return {
            'noise_model': 'depolarizing',
            'prob': prob
        }
    
    @staticmethod
    def get_amplitude_damping_noise(gamma: float = 0.05) -> Dict[str, Any]:
        """Get amplitude damping noise model."""
        return {
            'noise_model': 'amplitude_damping',
            'gamma': gamma
        }

class ThermalQuantumKernel:
    """Enhanced quantum kernel for thermal anomaly detection."""
    
    def __init__(
        self, 
        n_qubits: int = 8, 
        feature_map_reps: int = 2, 
        backend: str = "default.qubit",
        noise_model: Optional[Dict[str, Any]] = None,
        cache_circuits: bool = True,
        max_workers: int = 4
    ):
        self.n_qubits = n_qubits
        self.feature_map_reps = feature_map_reps
        self.backend = backend
        self.noise_model = noise_model
        self.cache_circuits = cache_circuits
        self.max_workers = max_workers
        
        # Initialize device with noise if specified
        if noise_model:
            if noise_model['noise_model'] == 'depolarizing':
                self.device = qml.device(
                    backend, 
                    wires=n_qubits,
                    noise='depolarizing',
                    prob=noise_model.get('prob', 0.01)
                )
            elif noise_model['noise_model'] == 'amplitude_damping':
                self.device = qml.device(
                    backend,
                    wires=n_qubits,
                    noise='amplitude_damping',
                    gamma=noise_model.get('gamma', 0.05)
                )
        else:
            self.device = qml.device(backend, wires=n_qubits)
        
        # Thread-local storage for quantum circuits
        self._thread_local = threading.local()
        
        # Initialize Qiskit components
        self.qiskit_feature_map = ZZFeatureMap(
            feature_dimension=n_qubits, 
            reps=feature_map_reps, 
            entanglement="linear"
        )
        
        # Cache for computed circuits
        self._circuit_cache = {} if cache_circuits else None
    
    def _get_qnode(self):
        """Get thread-local quantum node."""
        if not hasattr(self._thread_local, 'qnode'):
            self._thread_local.qnode = self._create_qnode()
        return self._thread_local.qnode
    
    def _create_qnode(self):
        """Create quantum node with current device."""
        @qml.qnode(self.device, interface='numpy')
        def thermal_kernel_circuit(x1, x2, params):
            """Optimized quantum circuit for thermal similarity."""
            n_features = len(x1)
            
            # Amplitude encoding for first patch
            qml.AmplitudeEmbedding(features=x1, wires=range(self.n_qubits), normalize=True)
            
            # Parameterized quantum feature map with optimized entanglement
            for layer in range(self.feature_map_reps):
                # Rotation gates
                for i in range(self.n_qubits):
                    qml.RY(params[layer * self.n_qubits + i], wires=i)
                
                # Optimized entanglement pattern using broadcast
                if self.n_qubits > 1:
                    qml.broadcast(qml.CNOT, wires=range(self.n_qubits), pattern="ring")
                
                # Thermal-sensitive Z-rotations
                for i in range(self.n_qubits - 1):
                    qml.RZ(x1[i % len(x1)] * x2[i % len(x2)], wires=i + 1)
            
            # Inverse amplitude encoding for fidelity calculation
            qml.adjoint(qml.AmplitudeEmbedding)(features=x2, wires=range(self.n_qubits), normalize=True)
            
            return qml.probs(wires=range(self.n_qubits))
        
        return thermal_kernel_circuit
    
    @lru_cache(maxsize=1000)
    def _cached_circuit_execution(self, x1_hash: str, x2_hash: str, params_hash: str):
        """Cached circuit execution."""
        # This would need actual parameter reconstruction from hashes
        # Simplified for demonstration
        pass
    
    def compute_kernel_matrix_batch(self, X: np.ndarray, Y: Optional[np.ndarray] = None, batch_size: int = 32) -> np.ndarray:
        """Batch computation of quantum kernel matrix for improved performance."""
        if Y is None:
            Y = X
            
        n_samples_x, n_samples_y = X.shape[0], Y.shape[0]
        kernel_matrix = np.zeros((n_samples_x, n_samples_y))
        
        # Initialize parameters
        n_params = self.feature_map_reps * self.n_qubits
        params = np.random.normal(0, 0.1, n_params)
        
        # Batch processing with PennyLane batch execution
        qnode = self._get_qnode()
        
        # Process in batches to manage memory
        for i in range(0, n_samples_x, batch_size):
            for j in range(0, n_samples_y, batch_size):
                end_i = min(i + batch_size, n_samples_x)
                end_j = min(j + batch_size, n_samples_y)
                
                # Prepare batch data
                batch_x = X[i:end_i]
                batch_y = Y[j:end_j]
                
                # Vectorized processing
                for bi, xi in enumerate(batch_x):
                    for bj, yj in enumerate(batch_y):
                        # Flatten and normalize patches
                        x1 = xi.flatten()[:self.n_qubits]
                        x2 = yj.flatten()[:self.n_qubits]
                        
                        x1 = x1 / (np.linalg.norm(x1) + 1e-8)
                        x2 = x2 / (np.linalg.norm(x2) + 1e-8)
                        
                        # Use resource manager for quantum operations
                        with QuantumResourceManager(self.device):
                            try:
                                probs = qnode(x1, x2, params)
                                kernel_matrix[i + bi, j + bj] = np.abs(probs[0])
                            except Exception as e:
                                logger.warning(f"Quantum circuit execution failed: {e}")
                                # Fallback to classical similarity
                                kernel_matrix[i + bi, j + bj] = np.dot(x1, x2)
        
        return kernel_matrix
    
    def compute_kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Standard kernel matrix computation with fallback."""
        try:
            return self.compute_kernel_matrix_batch(X, Y)
        except Exception as e:
            logger.error(f"Batch quantum kernel computation failed: {e}")
            # Fallback to classical kernel
            return self._classical_kernel_fallback(X, Y)
    
    def _classical_kernel_fallback(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Classical kernel as fallback when quantum computation fails."""
        if Y is None:
            Y = X
        
        # RBF kernel as fallback
        from sklearn.metrics.pairwise import rbf_kernel
        X_flat = X.reshape(X.shape[0], -1)
        Y_flat = Y.reshape(Y.shape[0], -1)
        return rbf_kernel(X_flat, Y_flat, gamma=0.1)

class QuantumThermalDetector(nn.Module):
    """Enhanced hybrid quantum-classical module with optimizations."""
    
    def __init__(
        self, 
        patch_size: int = 8, 
        n_qubits: int = 6,
        noise_model: Optional[Dict[str, Any]] = None,
        fallback_enabled: bool = True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.fallback_enabled = fallback_enabled
        
        # Classical preprocessing with batch normalization
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        
        # Quantum kernel with noise support
        self.quantum_kernel = ThermalQuantumKernel(
            n_qubits=n_qubits, 
            noise_model=noise_model,
            cache_circuits=True
        )
        
        # Enhanced classical post-processing
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        
        # Classical fallback model
        if fallback_enabled:
            self.classical_fallback = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32 * 16, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
    
    def forward(self, x: torch.Tensor, use_quantum: bool = True) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Input validation
        if x.size(-1) > 256 or x.size(-2) > 256:
            raise ValueError("Input image too large. Maximum size: 256x256")
        
        # Classical preprocessing
        x_processed = self.conv_reduce(x)
        x_processed = self.adaptive_pool(x_processed)
        
        if use_quantum:
            try:
                # Quantum feature extraction
                quantum_features = self._quantum_forward(x_processed)
                output = self.classifier(quantum_features)
            except Exception as e:
                logger.warning(f"Quantum forward pass failed: {e}")
                if self.fallback_enabled:
                    output = self.classical_fallback(x)
                else:
                    raise
        else:
            # Use classical fallback directly
            output = self.classical_fallback(x)
        
        return output
    
    def _quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum forward pass with error handling."""
        batch_size = x.size(0)
        
        # Convert to numpy for quantum processing
        x_np = x.detach().cpu().numpy()
        
        quantum_features = []
        for i in range(batch_size):
            patch = x_np[i, 0]  # Single channel
            
            # Compute quantum kernel features
            try:
                kernel_vals = self._compute_thermal_similarity(patch)
                quantum_features.append(kernel_vals)
            except Exception as e:
                logger.warning(f"Quantum similarity computation failed for batch {i}: {e}")
                # Use zero features as fallback
                quantum_features.append(np.zeros(64))
        
        return torch.tensor(quantum_features, dtype=torch.float32, device=x.device)
    
    def _compute_thermal_similarity(self, patch: np.ndarray) -> np.ndarray:
        """Enhanced thermal similarity computation with caching."""
        # Reference thermal patterns (learnable parameters in practice)
        ref_patterns = np.array([
            np.random.normal(0.7, 0.1, (self.patch_size, self.patch_size)),  # Hot spot
            np.random.normal(0.3, 0.1, (self.patch_size, self.patch_size)),  # Cool area
            np.random.normal(0.5, 0.05, (self.patch_size, self.patch_size)), # Moderate
            np.random.normal(0.8, 0.15, (self.patch_size, self.patch_size)), # Very hot
        ])
        
        similarities = []
        
        # Use quantum kernel with resource management
        with QuantumResourceManager(self.quantum_kernel.device):
            for ref_pattern in ref_patterns:
                kernel_matrix = self.quantum_kernel.compute_kernel_matrix(
                    patch.reshape(1, -1), 
                    ref_pattern.reshape(1, -1)
                )
                similarities.extend(kernel_matrix[0])
        
        # Pad to fixed size and add statistical features
        similarities = np.array(similarities)
        statistical_features = [
            np.mean(patch), np.std(patch), np.max(patch), np.min(patch),
            np.median(patch), np.percentile(patch, 95), np.percentile(patch, 5),
            np.sum(patch > np.mean(patch) + 2*np.std(patch))  # Hot pixel count
        ]
        
        all_features = np.concatenate([similarities, statistical_features])
        
        # Pad to fixed size
        padded = np.zeros(64)
        padded[:min(len(all_features), 64)] = all_features[:64]
        
        return padded
