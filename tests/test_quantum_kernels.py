import pytest
import numpy as np
import torch
from src.quantum.quantum_kernels import ThermalQuantumKernel, QuantumThermalDetector

class TestQuantumKernels:
    
    def test_thermal_kernel_initialization(self):
        """Test quantum kernel initialization."""
        kernel = ThermalQuantumKernel(n_qubits=4, feature_map_reps=2)
        assert kernel.n_qubits == 4
        assert kernel.feature_map_reps == 2
    
    def test_kernel_matrix_computation(self):
        """Test quantum kernel matrix computation."""
        kernel = ThermalQuantumKernel(n_qubits=4, feature_map_reps=1)
        
        # Create sample data
        X = np.random.rand(5, 4, 4)  # 5 patches of 4x4
        
        # Compute kernel matrix
        K = kernel.compute_kernel_matrix(X)
        
        # Check properties
        assert K.shape == (5, 5)
        assert np.allclose(K, K.T, atol=1e-6)  # Should be symmetric
        assert np.all(K >= 0)  # Should be positive
    
    def test_quantum_thermal_detector(self):
        """Test quantum thermal detector."""
        detector = QuantumThermalDetector(patch_size=8, n_qubits=4)
        
        # Create sample input
        batch_size = 2
        x = torch.randn(batch_size, 1, 16, 16)  # Batch of thermal images
        
        # Forward pass
        output = detector(x)
        
        # Check output shape
        assert output.shape == (batch_size, 2)  # Binary classification
        
        # Check output range (logits)
        assert torch.all(torch.isfinite(output))

    @pytest.mark.parametrize("n_qubits,patch_size", [(4, 8), (6, 16)])
    def test_different_configurations(self, n_qubits, patch_size):
        """Test different quantum kernel configurations."""
        detector = QuantumThermalDetector(patch_size=patch_size, n_qubits=n_qubits)
        
        x = torch.randn(1, 1, patch_size*2, patch_size*2)
        output = detector(x)
        
        assert output.shape == (1, 2)
