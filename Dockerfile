FROM python:3.11-slim

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependencies first to leverage Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch to keep the image lightweight.
# Note: For GPU support, replace with the CUDA-compatible index URL.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure required directory structure exists
RUN mkdir -p data/processed outputs/figures outputs/results outputs/checkpoints pretrained_models

# Default entrypoint for interactive shell
CMD ["/bin/bash"]