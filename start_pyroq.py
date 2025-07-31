#!/usr/bin/env python3
"""
Enhanced startup script for PyroQ with configuration validation and health checks.
"""
import os
import sys
import logging
import asyncio
import signal
from pathlib import Path
from typing import Optional
import uvicorn
import redis
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings, create_env_template
from src.edge.edge_deployment import create_app

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PyroQServer:
    """Enhanced server manager with health checks and graceful shutdown."""
    
    def __init__(self):
        self.app = None
        self.server = None
        self.running = False
    
    def validate_environment(self) -> bool:
        """Validate environment and dependencies."""
        logger.info("Validating environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        # Check model files
        model_path = Path(settings.model_path)
        onnx_path = Path(settings.model_path.replace('.pt', '.onnx'))
        
        if not model_path.exists() and not onnx_path.exists():
            logger.error(f"Model file not found: {settings.model_path}")
            logger.info("Run training first: python train_pyroq.py --data_path data/patches")
            return False
        
        # Check Redis connection
        if settings.enable_caching:
            try:
                redis_client = redis.from_url(settings.redis_url)
                redis_client.ping()
                logger.info("Redis connection successful")
            except (redis.RedisError, ConnectionError) as e:
                logger.warning(f"Redis connection failed: {e}. Caching will use memory fallback.")
        
        # Check GPU availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.info("CUDA not available, using CPU")
        
        # Check quantum backends
        try:
            import pennylane as qml
            device = qml.device(settings.quantum_backend, wires=4)
            logger.info(f"Quantum backend '{settings.quantum_backend}' available")
        except Exception as e:
            logger.warning(f"Quantum backend issue: {e}")
        
        # Validate configuration
        try:
            settings._validate_settings()
            logger.info("Configuration validation passed")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            return False
        
        return True
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def create_directories(self):
        """Create necessary directories."""
        dirs = [
            Path("logs"),
            Path("tmp"),
            Path(settings.model_path).parent,
            Path("outputs")
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def start_server(
        self, 
        host: str = "0.0.0.0", 
        port: int = 8000,
        workers: int = 1,
        reload: bool = False
    ):
        """Start the server with enhanced configuration."""
        
        if not self.validate_environment():
            logger.error("Environment validation failed!")
            sys.exit(1)
        
        self.create_directories()
        self.setup_signal_handlers()
        
        # Create FastAPI app
        logger.info("Creating FastAPI application...")
        self.app = create_app()
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level=settings.log_level.lower(),
            access_log=True,
            workers=workers if not reload else 1,
            reload=reload,
            reload_dirs=[str(project_root / "src")] if reload else None
        )
        
        # Add SSL in production
        if settings.is_production:
            ssl_keyfile = os.getenv("SSL_KEYFILE")
            ssl_certfile = os.getenv("SSL_CERTFILE")
            
            if ssl_keyfile and ssl_certfile:
                config.ssl_keyfile = ssl_keyfile
                config.ssl_certfile = ssl_certfile
                logger.info("SSL enabled")
            else:
                logger.warning("Production mode but SSL certificates not found")
        
        # Start server
        server = uvicorn.Server(config)
        
        logger.info(f"Starting PyroQ API v2.1")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Host: {host}:{port}")
        logger.info(f"Model: {settings.model_path}")
        logger.info(f"Quantum enabled: {settings.quantum_enabled}")
        logger.info(f"Workers: {workers}")
        
        try:
            self.running = True
            server.run()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown."""
        if self.running:
            logger.info("Shutting down PyroQ server...")
            self.running = False
            # Additional cleanup can be added here
            logger.info("Server shutdown complete")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PyroQ Enhanced Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--create-env', action='store_true', help='Create .env template file')
    parser.add_argument('--validate', action='store_true', help='Only validate environment, do not start server')
    
    args = parser.parse_args()
    
    if args.create_env:
        create_env_template()
        return
    
    server = PyroQServer()
    
    if args.validate:
        success = server.validate_environment()
        print("✅ Environment validation passed" if success else "❌ Environment validation failed")
        sys.exit(0 if success else 1)
    
    # Start server
    server.start_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
