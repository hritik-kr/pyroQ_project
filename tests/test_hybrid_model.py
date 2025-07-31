import pytest
import torch
import torch.nn as nn
from src.hybrid.hybrid_model import PyroQHybridModel
from src.quantum.qcnn import QCNN

class TestHybridModel:
    
    def test_model_initialization(self):
        """Test hybrid model initialization."""
        model = PyroQHybridModel(
            model_type="qcnn",
            input_size=(32, 32),
            num_classes=2
        )
        
        assert isinstance(model.model, QCNN)
        assert model.num_classes == 2
    
    def test_forward_pass(self):
        """Test forward pass through hybrid model."""
        model = PyroQHybridModel(
            model_type="qcnn",
            input_size=(32, 32),
            num_classes=2
        )
        
        # Create sample input
        batch_size = 4
        x = torch.randn(batch_size, 1, 32, 32)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 2)
        
        # Check that output is valid
        assert torch.all(torch.isfinite(output))
    
    def test_training_step(self):
        """Test training step."""
        model = PyroQHybridModel(
            model_type="qcnn",
            input_size=(32, 32),
            num_classes=2
        )
        
        # Create sample batch
        batch_size = 4
        x = torch.randn(batch_size, 1, 32, 32)
        y = torch.randint(0, 2, (batch_size,))
        batch = (x, y)
        
        # Training step
        loss = model.training_step(batch, 0)
        
        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert torch.isfinite(loss)

class TestQCNN:
    
    def test_qcnn_creation(self):
        """Test QCNN model creation."""
        model = QCNN(input_size=(32, 32), num_classes=2)
        
        # Check model structure
        assert hasattr(model, 'classical_conv1')
        assert hasattr(model, 'classical_conv2')
        assert hasattr(model, 'quantum_conv')
        assert hasattr(model, 'quantum_pool')
        assert hasattr(model, 'classifier')
    
    def test_qcnn_forward(self):
        """Test QCNN forward pass."""
        model = QCNN(input_size=(32, 32), num_classes=2)
        model.eval()
        
        # Create sample input
        x = torch.randn(1, 1, 32, 32)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output
        assert output.shape == (1, 2)
        assert torch.all(torch.isfinite(output))
