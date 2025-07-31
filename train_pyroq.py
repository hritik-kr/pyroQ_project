#!/usr/bin/env python3
"""
Main training script for PyroQ: Quantum-Enhanced Wildfire Detection System
"""

import argparse
import logging
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.data.dataset import FireDataModule
from src.hybrid.hybrid_model import PyroQHybridModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train PyroQ Wildfire Detection Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='qcnn',
                       choices=['qcnn', 'quantum_kernel'],
                       help='Type of model to train')
    parser.add_argument('--input_size', type=int, nargs=2, default=[32, 32],
                       help='Input image size [height, width]')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    
    # Hardware arguments
    parser.add_argument('--gpus', type=int, default=-1,
                       help='Number of GPUs to use (-1 for all available)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--experiment_name', type=str, default='pyroq_experiment',
                       help='Name of the experiment')
    
    # Quantum arguments
    parser.add_argument('--quantum_backend', type=str, default='default.qubit',
                       help='Quantum backend to use')
    parser.add_argument('--quantum_noise', action='store_true',
                       help='Enable quantum noise simulation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up data module
    logger.info("Setting up data module...")
    data_module = FireDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=tuple(args.input_size),
        val_split=args.val_split
    )
    
    # Get data loaders
    train_loader, val_loader = data_module.get_dataloaders()
    logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Set up model
    logger.info(f"Initializing {args.model_type} model...")
    model = PyroQHybridModel(
        model_type=args.model_type,
        input_size=tuple(args.input_size),
        num_classes=2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        quantum_noise=args.quantum_noise
    )
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min',
            verbose=True
        ),
        ModelCheckpoint(
            dirpath=output_path / 'checkpoints',
            filename='pyroq-{epoch:02d}-{val_f1:.3f}',
            monitor='val_f1',
            mode='max',
            save_top_k=3,
            save_last=True
        )
    ]
    
    # Set up logger
    tb_logger = TensorBoardLogger(
        save_dir=output_path / 'logs',
        name=args.experiment_name,
        version=None
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus if torch.cuda.is_available() and args.gpus != 0 else 0,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=10,
        val_check_interval=0.25,  # Check validation 4 times per epoch
        gradient_clip_val=1.0,
        deterministic=True,
        precision=16 if torch.cuda.is_available() else 32
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test best model
    logger.info("Testing best model...")
    trainer.test(model, val_loader)
    
    # Save final model
    final_model_path = output_path / 'pyroq_final.pt'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Export to ONNX
    try:
        from src.edge.edge_deployment import export_to_onnx
        onnx_path = output_path / 'pyroq_final.onnx'
        export_to_onnx(str(final_model_path), str(onnx_path), tuple(args.input_size))
        logger.info(f"ONNX model saved to {onnx_path}")
    except Exception as e:
        logger.warning(f"Failed to export ONNX model: {e}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
