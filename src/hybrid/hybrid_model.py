import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings

from src.quantum.qcnn import QCNN
from src.quantum.quantum_kernels import QuantumThermalDetector

class PyroQHybridModel(pl.LightningModule):
    """PyTorch Lightning module for the PyroQ hybrid quantum-classical model."""
    
    def __init__(
        self,
        model_type: str = "qcnn",
        input_size: tuple = (32, 32),
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        quantum_noise: bool = False,
        onnx_export: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.onnx_export = onnx_export
        
        # Initialize model based on type
        if model_type == "qcnn":
            self.model = QCNN(input_size=input_size, num_classes=num_classes)
        elif model_type == "quantum_kernel":
            self.model = QuantumThermalDetector(patch_size=8, n_qubits=6)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.train_metrics = []
        self.val_metrics = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        
        return {
            'val_loss': loss,
            'preds': preds.cpu(),
            'targets': y.cpu(),
            'probs': probs.cpu()
        }
    
    def validation_epoch_end(self, outputs):
        # Aggregate predictions
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        all_probs = torch.cat([x['probs'] for x in outputs])
        
        # Calculate metrics
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = accuracy_score(all_targets.numpy(), all_preds.numpy())
        
        if self.num_classes == 2:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets.numpy(), all_preds.numpy(), average='binary'
            )
            try:
                auc = roc_auc_score(all_targets.numpy(), all_probs[:, 1].numpy())
                self.log('val_auc', auc)
            except ValueError:
                auc = 0.0
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets.numpy(), all_preds.numpy(), average='macro'
            )
            auc = 0.0
        
        # Log metrics
        self.log('val_loss', avg_loss)
        self.log('val_acc', acc)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def to_onnx(self, file_path: str, input_sample: Optional[torch.Tensor] = None, 
                opset_version: int = 15, verbose: bool = False) -> Tuple[bool, str]:
        """
        Export model to ONNX format with support for quantum operations.
        
        Args:
            file_path: Path to save the ONNX model
            input_sample: Sample input tensor for tracing
            opset_version: ONNX opset version to use
            verbose: Whether to print export details
            
        Returns:
            Tuple of (success, message)
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            if input_sample is None:
                # Create default input sample if not provided
                input_sample = torch.randn(1, 1, *self.hparams.input_size)
            
            # Set model to eval mode
            self.eval()
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                input_sample,
                file_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                opset_version=opset_version,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                verbose=verbose,
                do_constant_folding=True
            )
            
            # Validate ONNX model
            onnx_model = onnx.load(file_path)
            onnx.checker.check_model(onnx_model)
            
            # Optimize for inference
            optimized_model = optimizer.optimize_model(
                file_path,
                model_type='bert',
                num_heads=0,
                hidden_size=0,
                optimization_options=optimizer.OptimizationOptions(
                    enable_gelu=True,
                    enable_layer_norm=True,
                    enable_attention=True,
                    enable_skip_layer_norm=True,
                    enable_embed_layer_norm=True,
                    enable_bias_skip_layer_norm=True,
                    enable_bias_gelu=True,
                    enable_gelu_approximation=False
                )
            )
            optimized_model.save_model_to_file(file_path)
            
            return True, f"Successfully exported ONNX model to {file_path}"
            
        except Exception as e:
            warnings.warn(f"ONNX export failed: {str(e)}")
            return False, f"ONNX export failed: {str(e)}"

def train_hybrid_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str = "qcnn",
    input_size: tuple = (32, 32),
    max_epochs: int = 50,
    gpus: int = 0,
    export_onnx: bool = False,
    onnx_path: str = "models/pyroq.onnx"
) -> PyroQHybridModel:
    """Train the hybrid quantum-classical model."""
    
    # Initialize model
    model = PyroQHybridModel(
        model_type=model_type,
        input_size=input_size,
        num_classes=2,
        learning_rate=1e-3,
        onnx_export=export_onnx
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus if torch.cuda.is_available() else 0,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            ),
            pl.callbacks.ModelCheckpoint(
                monitor='val_f1',
                mode='max',
                save_top_k=1,
                filename='best-pyroq-{epoch:02d}-{val_f1:.3f}'
            )
        ],
        logger=pl.loggers.TensorBoardLogger('lightning_logs/', name='PyroQ')
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Export to ONNX if requested
    if export_onnx:
        sample_input = next(iter(val_loader))[0][0].unsqueeze(0)
        success, msg = model.to_onnx(onnx_path, sample_input)
        if not success:
            warnings.warn(msg)
        else:
            print(msg)
    
    return model

if __name__ == "__main__":
    import argparse
    from src.data.dataset import create_dataloaders
    
    parser = argparse.ArgumentParser(description='Train PyroQ Hybrid Model')
    parser.add_argument('--train', type=str, required=True, help='Path to training data')
    parser.add_argument('--val', type=str, help='Path to validation data')
    parser.add_argument('--model_type', type=str, default='qcnn', choices=['qcnn', 'quantum_kernel'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save', type=str, default='models/pyroq.pt', help='Path to save model')
    parser.add_argument('--export_onnx', action='store_true', help='Export to ONNX format')
    parser.add_argument('--onnx_path', type=str, default='models/pyroq.onnx', help='Path to save ONNX model')
    
    args = parser.parse_args()
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_path=args.train,
        val_path=args.val,
        batch_size=args.batch_size
    )
    
    # Train model
    model = train_hybrid_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model_type=args.model_type,
        max_epochs=args.epochs,
        export_onnx=args.export_onnx,
        onnx_path=args.onnx_path
    )
    
    # Save model
    torch.save(model.state_dict(), args.save)
    print(f"Model saved to {args.save}")