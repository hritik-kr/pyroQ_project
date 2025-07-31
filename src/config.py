from pydantic import BaseSettings, Field
from typing import Optional, List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration using Pydantic BaseSettings."""
    
    # Model settings
    model_path: str = Field(default="models/pyroq.pt", description="Path to the trained model")
    use_onnx: bool = Field(default=True, description="Use ONNX model if available")
    enable_caching: bool = Field(default=True, description="Enable inference caching")
    quantum_enabled: bool = Field(default=True, description="Enable quantum processing")
    
    # Redis settings
    redis_host: str = Field(default="localhost", description="Redis server host")
    redis_port: int = Field(default=6379, description="Redis server port")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-change-in-production", description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiration time")
    
    # API settings
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Maximum file size in bytes")
    max_image_width: int = Field(default=2048, description="Maximum image width")
    max_image_height: int = Field(default=2048, description="Maximum image height")
    
    # Rate limiting
    rate_limit_requests_per_hour: int = Field(default=100, description="Requests per hour limit")
    rate_limit_geotiff_per_hour: int = Field(default=20, description="GeoTIFF requests per hour limit")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    metrics_retention_hours: int = Field(default=24, description="Metrics retention in hours")
    
    # Environment
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment (development/production)")
    
    # Allowed origins for CORS
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"], 
        description="Allowed CORS origins"
    )
    
    # Trusted hosts
    trusted_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1", "*.your-domain.com"],
        description="Trusted host patterns"
    )
    
    # Quantum backend settings
    quantum_backend: str = Field(default="default.qubit", description="Quantum backend")
    qiskit_token: Optional[str] = Field(default=None, description="IBM Quantum token")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validate critical settings
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate critical configuration settings."""
        # Model path validation
        model_path = Path(self.model_path)
        if not model_path.parent.exists():
            raise ValueError(f"Model directory does not exist: {model_path.parent}")
        
        # Check for either .pt or .onnx model
        if not (model_path.exists() or model_path.with_suffix('.onnx').exists()):
            raise ValueError(f"Model file not found: {self.model_path}")

        # Production checks
        if self.environment == "production":
            if self.secret_key == "your-secret-key-change-in-production":
                raise ValueError("Secret key must be changed in production!")
            if self.debug:
                raise ValueError("Debug mode must be disabled in production!")
            if not all(host.startswith(('https://', 'localhost')) for host in self.allowed_origins):
                raise ValueError("Production must use HTTPS for all origins")

        # Quantum backend validation
        valid_backends = ['default.qubit', 'qiskit.aer', 'qiskit.ibmq']
        if self.quantum_backend not in valid_backends:
            raise ValueError(f"Invalid quantum backend. Must be one of: {valid_backends}")

        # Redis URL validation
        if self.enable_caching:
            try:
                import redis
                redis_client = redis.from_url(self.redis_url)
                redis_client.ping()
            except Exception as e:
                raise ValueError(f"Redis connection failed: {e}")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

# Global settings instance
settings = Settings()

# Environment file template
ENV_TEMPLATE = """
# PyroQ Configuration
# Copy this to .env and modify as needed

# Model Configuration
MODEL_PATH=models/pyroq.pt
USE_ONNX=true
ENABLE_CACHING=true
QUANTUM_ENABLED=true

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# Security Configuration (CHANGE IN PRODUCTION!)
SECRET_KEY=your-production-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Configuration
MAX_FILE_SIZE=52428800
MAX_IMAGE_WIDTH=2048
MAX_IMAGE_HEIGHT=2048

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_HOUR=100
RATE_LIMIT_GEOTIFF_PER_HOUR=20

# Environment
DEBUG=false
ENVIRONMENT=production

# CORS Origins (comma-separated)
ALLOWED_ORIGINS=https://your-domain.com,https://app.your-domain.com

# Trusted Hosts (comma-separated)
TRUSTED_HOSTS=your-domain.com,*.your-domain.com

# Quantum Backend
QUANTUM_BACKEND=default.qubit
QISKIT_TOKEN=your_ibm_quantum_token_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/pyroq.log
"""

def create_env_template():
    """Create .env template file."""
    with open(".env.template", "w") as f:
        f.write(ENV_TEMPLATE)
    print("Created .env.template - copy to .env and customize")

if __name__ == "__main__":
    create_env_template()