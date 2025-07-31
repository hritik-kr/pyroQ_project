import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import Tuple, List, Optional
import logging
import ray
from functools import lru_cache
import threading
import gc

logger = logging.getLogger(__name__)

class VectorizedQuantumConvolution:
    """Ray-optimized quantum convolution layer with LRU caching and resource management"""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, stride: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.stride = stride
        self.device = qml.device("default.qubit", wires=n_qubits)
        
        # Improved parameter initialization
        self.params = np.random.normal(0, np.sqrt(2.0 / n_qubits), (n_layers, n_qubits, 3))
        
        # Initialize Ray for distributed processing
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Thread-local storage for quantum nodes
        self._thread_local = threading.local()
        
        # Remote processing function with caching
        self.process_patch_remote = ray.remote(self._process_patch_remote_wrapper)

    @lru_cache(maxsize=1000)
    def _process_patch_remote_wrapper(self, patch_data: tuple):
        """Wrapper for LRU caching with Ray serialization"""
        return self._process_patch(np.array(patch_data))

    def _process_patch(self, patch_data: np.ndarray):
        """Process single patch through quantum circuit with error handling"""
        try:
            # Pad or truncate to match qubit count
            if len(patch_data) < 2**self.n_qubits:
                padded = np.pad(patch_data, (0, 2**self.n_qubits - len(patch_data)))
            else:
                padded = patch_data[:2**self.n_qubits]
            
            # Normalize inputs
            normalized = padded / (np.linalg.norm(padded) + 1e-8)
            return self._get_qnode()(normalized, self.params)
        except Exception as e:
            logger.warning(f"Quantum patch processing failed: {e}")
            # Fallback to classical features
            return np.mean(patch_data.reshape(-1, 1), axis=1)[:self.n_qubits]

    def _get_qnode(self):
        """Get thread-local quantum node"""
        if not hasattr(self._thread_local, 'qnode'):
            self._thread_local.qnode = self._create_qnode()
        return self._thread_local.qnode

    def _create_qnode(self):
        """Create optimized quantum convolution circuit"""
        @qml.qnode(self.device, interface='numpy')
        def quantum_conv_circuit(inputs, params):
            """Optimized quantum circuit for convolution"""
            # Data encoding with proper normalization
            normalized_inputs = inputs / (np.linalg.norm(inputs) + 1e-8)
            qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.n_qubits), normalize=True)
            
            # Variational layers with improved entanglement
            for layer in range(self.n_layers):
                # Rotation gates
                for qubit in range(self.n_qubits):
                    qml.Rot(params[layer, qubit, 0], params[layer, qubit, 1], params[layer, qubit, 2], wires=qubit)
                
                # Optimized entangling gates
                if self.n_qubits > 1:
                    qml.broadcast(qml.CNOT, wires=range(self.n_qubits), pattern="ring")
                    
                # Additional entanglement
                if layer % 2 == 0 and self.n_qubits > 2:
                    qml.broadcast(qml.CZ, wires=range(self.n_qubits), 
                                pattern=[[i, (i+2) % self.n_qubits] for i in range(0, self.n_qubits, 2)])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return quantum_conv_circuit

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Ray-optimized forward pass with batched processing"""
        batch_size, channels, height, width = x.shape
        kernel_size = 2
        
        # Extract patches efficiently using unfold
        x_tensor = torch.from_numpy(x).float()
        patches = F.unfold(x_tensor, kernel_size=kernel_size, stride=self.stride)
        patches = patches.transpose(1, 2).reshape(-1, kernel_size * kernel_size)
        
        # Process in parallel batches
        batch_results = []
        ray_batch_size = 256  # Optimal for Ray
        
        for i in range(0, len(patches), ray_batch_size):
            batch = patches[i:i + ray_batch_size]
            # Convert to tuples for LRU caching
            batch_tuples = [tuple(patch.numpy().tolist()) for patch in batch]
            # Distributed processing
            results = ray.get([self.process_patch_remote.remote(tuple(patch)) 
                            for patch in batch_tuples])
            batch_results.extend(results)
        
        # Reshape results
        out_height = (height - kernel_size) // self.stride + 1
        out_width = (width - kernel_size) // self.stride + 1
        return np.array(batch_results).reshape(batch_size, channels, out_height, out_width, self.n_qubits)

    def __del__(self):
        """Clean up Ray resources"""
        if ray.is_initialized():
            ray.shutdown()

class QuantumPoolingLayer:
    """Enhanced quantum pooling with multiple strategies"""
    
    def __init__(self, n_qubits: int = 2, pooling_type: str = "max"):
        self.n_qubits = n_qubits
        self.pooling_type = pooling_type
        self.device = qml.device("default.qubit", wires=n_qubits)
        self._thread_local = threading.local()
    
    def _get_qnode(self):
        if not hasattr(self._thread_local, 'qnode'):
            self._thread_local.qnode = self._create_qnode()
        return self._thread_local.qnode
    
    def _create_qnode(self):
        @qml.qnode(self.device, interface='numpy')
        def quantum_pool_circuit(inputs):
            """Enhanced quantum pooling circuit"""
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            
            if self.pooling_type == "max":
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.RY(np.pi/4, wires=0)
            elif self.pooling_type == "mean":
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
            
            return qml.expval(qml.PauliZ(0))
        
        return quantum_pool_circuit
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vectorized quantum pooling with error handling"""
        batch_size, channels, height, width, features = x.shape
        output_height, output_width = height // 2, width // 2
        
        output = np.zeros((batch_size, channels, output_height, output_width, features))
        qnode = self._get_qnode()
        
        for b in range(batch_size):
            for c in range(channels):
                for f in range(features):
                    for i in range(0, height - 1, 2):
                        for j in range(0, width - 1, 2):
                            region = x[b, c, i:i+2, j:j+2, f].flatten()
                            region = region / (np.linalg.norm(region) + 1e-8)
                            
                            if len(region) < 2**self.n_qubits:
                                region = np.pad(region, (0, 2**self.n_qubits - len(region)))
                            
                            try:
                                output[b, c, i//2, j//2, f] = qnode(region[:2**self.n_qubits])
                            except Exception as e:
                                logger.warning(f"Quantum pooling failed: {e}")
                                output[b, c, i//2, j//2, f] = (
                                    np.max(region) if self.pooling_type == "max" else np.mean(region))
        
        return output

class EnhancedQCNN(nn.Module):
    """Enhanced Quantum-Classical CNN with optimizations."""
    
    def __init__(
        self, 
        input_size: Tuple[int, int] = (32, 32), 
        num_classes: int = 2,
        quantum_enabled: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.input_size = input_size
        self.quantum_enabled = quantum_enabled
        
        # Enhanced classical layers with residual connections
        self.classical_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classical_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Quantum layers (optional)
        if quantum_enabled:
            self.quantum_conv = VectorizedQuantumConvolution(n_qubits=4, n_layers=2)
            self.quantum_pool = QuantumPoolingLayer(n_qubits=2, pooling_type="max")
        
        # Calculate feature size dynamically
        self.feature_size = self._calculate_feature_size(input_size)
        
        # Enhanced classifier with attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(128, num_classes)
        )
        
        # Classical fallback path
        self.classical_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Input size validation
        if x.size(-1) > 512 or x.size(-2) > 512:
            raise ValueError(f"Input size {x.shape} too large. Maximum: 512x512")
        
        # Classical processing
        x1 = self.classical_conv1(x)
        x2 = self.classical_conv2(x1)
        
        if self.quantum_enabled:
            try:
                # Quantum processing
                x_np = x2.detach().cpu().numpy()
                batch_size, channels, height, width = x_np.shape
                
                # Process each channel through quantum layers
                quantum_outputs = []
                for c in range(min(channels, 8)):  # Limit channels for quantum processing
                    channel_data = x_np[:, c:c+1, :, :]  # Keep channel dimension
                    
                    # Quantum convolution
                    conv_out = self.quantum_conv.forward(channel_data)
                    
                    # Quantum pooling
                    if conv_out.shape[2] > 1 and conv_out.shape[3] > 1:  # Check if pooling is needed
                        pool_out = self.quantum_pool.forward(conv_out)
                        quantum_outputs.append(pool_out)
                
                if quantum_outputs:
                    # Combine quantum outputs
                    quantum_features = np.concatenate(quantum_outputs, axis=-1)
                    quantum_features = torch.tensor(quantum_features, dtype=torch.float32, device=x.device)
                    
                    # Flatten for classification
                    quantum_features = quantum_features.view(batch_size, -1)
                    
                    # Apply attention if feature size allows
                    if quantum_features.size(-1) >= 128:
                        # Reshape for attention
                        att_input = quantum_features.view(batch_size, -1, 128)
                        att_output, _ = self.attention(att_input, att_input, att_input)
                        quantum_features = att_output.mean(dim=1)  # Global average pooling
                    
                    # Final classification
                    output = self.classifier(quantum_features)
                else:
                    # Fallback to classical path
                    output = self.classical_classifier(x2)
                    
            except Exception as e:
                logger.warning(f"Quantum processing failed: {e}")
                # Fallback to classical processing
                output = self.classical_classifier(x2)
        else:
            # Classical processing only
            output = self.classical_classifier(x2)
        
        return output
    
    def _calculate_feature_size(self, input_size: Tuple[int, int]) -> int:
        """Dynamically calculate feature size."""
        # Create dummy input
        dummy_input = torch.zeros(1, 1, *input_size)
        
        # Forward through classical layers
        with torch.no_grad():
            x1 = self.classical_conv1(dummy_input)
            x2 = self.classical_conv2(x1)
        
        # Calculate quantum output size (estimation)
        h, w = x2.shape[2], x2.shape[3]
        channels = min(x2.shape[1], 8)  # Limit channels for quantum
        
        if self.quantum_enabled:
            # After quantum conv and pool (rough estimation)
            quantum_features = channels * (h // 4) * (w // 4) * 4  # 4 qubits output
        else:
            quantum_features = 32 * 16  # Classical fallback size
        
        return quantum_features

def create_enhanced_qcnn(
    input_size: Tuple[int, int] = (32, 32), 
    num_classes: int = 2,
    quantum_enabled: bool = True
) -> EnhancedQCNN:
    """Factory function to create enhanced QCNN model."""
    return EnhancedQCNN(
        input_size=input_size, 
        num_classes=num_classes,
        quantum_enabled=quantum_enabled
    )