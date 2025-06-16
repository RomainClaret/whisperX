#!/bin/bash

# WhisperX MLX Setup Script for Apple Silicon
# This script creates a new micromamba environment optimized for MLX WhisperX

echo "==================================="
echo "WhisperX MLX Setup for Apple Silicon"
echo "==================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is designed for macOS (Apple Silicon)"
    exit 1
fi

# Check if micromamba is installed
if ! command -v micromamba &> /dev/null; then
    echo "Error: micromamba is not installed. Please install it first."
    echo "Visit: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
    exit 1
fi

# Environment name
ENV_NAME="whisper"

# Check if environment already exists
if micromamba env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and create a fresh one? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        micromamba env remove -n ${ENV_NAME} -y
    else
        echo "Exiting without changes."
        exit 0
    fi
fi

# Create environment
echo "Creating micromamba environment '${ENV_NAME}' with Python 3.11..."
micromamba create -n ${ENV_NAME} python=3.11 -y -c conda-forge

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate ${ENV_NAME}

echo "Installing dependencies..."

# Core MLX dependencies
echo "Installing MLX framework..."
pip install --upgrade pip
pip install mlx>=0.5.0 mlx-whisper>=0.2.0

# WhisperX and dependencies
echo "Installing WhisperX..."
pip install whisperx

# PyAnnote without onnxruntime-gpu issues
echo "Installing PyAnnote (CPU version for stability)..."
pip install pyannote-audio==3.0.0

# PyTorch for alignment and diarization
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# Additional useful packages
echo "Installing additional packages..."
pip install numpy pandas matplotlib jupyter ipython rich

# Set environment variables
echo "Setting up environment variables..."
cat >> ~/.bashrc << 'EOL'

# WhisperX MLX Environment Variables
if [[ "$CONDA_DEFAULT_ENV" == "whisper" ]] || [[ "$MAMBA_DEFAULT_ENV" == "whisper" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    echo "WhisperX MLX environment activated with MPS fallback enabled"
fi
EOL

# Create a test script
echo "Creating test script..."
cat > test_mlx_setup.py << 'EOL'
#!/usr/bin/env python3
"""Test script to verify MLX WhisperX setup"""

import sys
print("Testing MLX WhisperX Setup...")
print("-" * 40)

# Test imports
try:
    import mlx
    print("✓ MLX imported successfully")
except ImportError as e:
    print(f"✗ MLX import failed: {e}")

try:
    import mlx_whisper
    print("✓ MLX Whisper imported successfully")
except ImportError as e:
    print(f"✗ MLX Whisper import failed: {e}")

try:
    import whisperx
    print("✓ WhisperX imported successfully")
except ImportError as e:
    print(f"✗ WhisperX import failed: {e}")

try:
    import torch
    print("✓ PyTorch imported successfully")
    
    # Check for MPS availability
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✓ MPS (Metal Performance Shaders) is available")
    else:
        print("✗ MPS is not available")
        
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import pyannote.audio
    print("✓ PyAnnote imported successfully")
except ImportError as e:
    print(f"✗ PyAnnote import failed: {e}")

print("-" * 40)
print("Setup test complete!")
EOL

chmod +x test_mlx_setup.py

# Final instructions
echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To use the environment:"
echo "  micromamba activate ${ENV_NAME}"
echo ""
echo "To test the setup:"
echo "  python test_mlx_setup.py"
echo ""
echo "Environment variables set:"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1 (enables CPU fallback for unsupported MPS ops)"
echo ""
echo "Example usage:"
echo "  whisperx audio.wav --backend mlx --model mlx-community/whisper-large-v3-mlx"
echo ""