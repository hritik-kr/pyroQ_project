FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/outputs /app/logs

# Set environment variables
ENV PYTHONPATH=/app \
    GDAL_DATA=/usr/share/gdal \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Install NVIDIA runtime if available (for GPU support)
RUN if command -v nvidia-smi; then \
        pip install --no-cache-dir nvidia-cudnn-cu11==8.6.0.163 nvidia-cublas-cu11==11.11.3.6; \
    fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "src/edge/edge_deployment.py", "--serve", "--model", "models/pyroq.pt"]