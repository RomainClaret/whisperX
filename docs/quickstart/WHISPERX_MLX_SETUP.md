# WhisperX with MLX Backend - Complete Setup Guide

This guide provides comprehensive instructions for setting up WhisperX with Apple Silicon GPU acceleration using MLX (Apple's machine learning framework).

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)

## Overview

WhisperX is an enhanced version of OpenAI's Whisper that provides:
- Fast automatic speech recognition (70x realtime)
- Word-level timestamps
- Speaker diarization (who said what)
- Voice activity detection (VAD)

This implementation adds **MLX backend support** for Apple Silicon Macs (M1/M2/M3/M4), providing:
- 10x faster transcription using GPU acceleration
- Full compatibility with existing WhisperX features
- Optimized memory usage for Apple's unified memory architecture

## Prerequisites

### Hardware Requirements
- Apple Silicon Mac (M1, M2, M3, or M4)
- Recommended: 16GB+ unified memory
- Optimal: M4 Max with 128GB for large models and batch processing

### Software Requirements
- macOS 12.0 or later
- Python 3.11
- Micromamba or Conda
- Git

### Accounts Required
- [Hugging Face account](https://huggingface.co/) (for diarization models)

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/m-bain/whisperX.git
cd whisperX
```

### Step 2: Install Micromamba (if not installed)
```bash
# Install micromamba
curl micro.mamba.pm/install.sh | bash

# Restart your terminal or source the config
source ~/.bashrc  # or ~/.zshrc
```

### Step 3: Run the Setup Script
```bash
# Make the setup script executable
chmod +x setup_mlx_whisperx.sh

# Run the setup script
./setup_mlx_whisperx.sh
```

This script will:
1. Create a new environment called `whisper`
2. Install Python 3.11
3. Install all required dependencies including MLX
4. Set up environment variables
5. Create a test script

### Step 4: Activate the Environment
```bash
micromamba activate whisper
```

### Step 5: Verify Installation
```bash
# Run the test script
python test_mlx_setup.py
```

You should see all tests passing with checkmarks (✓).

## Configuration

### 1. Hugging Face Token Setup

To use speaker diarization, you need a Hugging Face token:

1. Create an account at [huggingface.co](https://huggingface.co/)
2. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept the user agreement for these models:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

4. Create a `.env` file in the repository root:
```bash
cp .env.example .env
```

5. Edit `.env` and add your token:
```
HF_TOKEN=your_huggingface_token_here
```

### 2. Environment Variables

The following environment variables are automatically set:
- `PYTORCH_ENABLE_MPS_FALLBACK=1` - Enables CPU fallback for unsupported MPS operations

## Usage

### Basic Transcription

```bash
# Using MLX backend (GPU accelerated)
whisperx audio.mp3 --backend mlx --model mlx-community/whisper-large-v3-mlx

# Using faster-whisper backend (CPU)
whisperx audio.mp3 --model large-v3
```

### Advanced Options

```bash
whisperx audio.mp3 \
  --backend mlx \
  --model mlx-community/whisper-large-v3-mlx \
  --batch_size 32 \              # Increase for better GPU utilization
  --chunk_size 60 \              # Larger chunks for M4 Max
  --output_dir outputs \         # Where to save results
  --output_format all \          # Export all formats
  --language en \                # Skip language detection
  --align_model WAV2VEC2_ASR_LARGE_LV60K_960H \  # Large alignment model
  --diarize \                    # Enable speaker diarization
  --min_speakers 2 \             # Minimum expected speakers
  --max_speakers 4 \             # Maximum expected speakers
  --verbose                      # Show progress
```

### Available MLX Models

- `mlx-community/whisper-tiny` - Fastest, least accurate
- `mlx-community/whisper-base` 
- `mlx-community/whisper-small-mlx`
- `mlx-community/whisper-medium-mlx`
- `mlx-community/whisper-large-v3-mlx` - Best accuracy
- `mlx-community/whisper-large-v3-turbo` - Faster large model

### Python API

```python
import whisperx
import gc

# Load model with MLX backend
model = whisperx.load_model(
    "mlx-community/whisper-large-v3-mlx",
    device="cpu",  # MLX handles GPU internally
    backend="mlx",
    compute_type="float16"
)

# Load and transcribe audio
audio = whisperx.load_audio("audio.mp3")
result = model.transcribe(audio, batch_size=32)

# Clean up
del model
gc.collect()

# Align whisper output (optional)
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], 
    device="cpu"
)
result = whisperx.align(
    result["segments"], 
    model_a, 
    metadata, 
    audio, 
    device="cpu"
)

# Assign speaker labels (optional)
diarize_model = whisperx.DiarizationPipeline(device="cpu")
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

# Print results
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
```

## Development

### Project Structure

```
whisperX/
├── whisperx/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── asr.py               # ASR model loading (modified for MLX)
│   ├── mlx_asr.py           # MLX backend implementation (new)
│   ├── transcribe.py        # Main transcription logic
│   ├── alignment.py         # Word-level alignment
│   ├── diarize.py           # Speaker diarization
│   └── audio.py             # Audio processing utilities
├── setup_mlx_whisperx.sh    # Setup script
├── test_mlx_integration.py  # Integration tests
├── .env.example             # Environment template
└── MLX_INTEGRATION_PLAN.md  # Technical documentation
```

### Key Modifications for MLX Support

1. **`whisperx/mlx_asr.py`** - New file implementing MLX backend
   - `MLXWhisperModel` class wrapping mlx_whisper
   - `MLXWhisperPipeline` implementing WhisperX pipeline interface
   - Compatible segment output format

2. **`whisperx/asr.py`** - Modified to support backend selection
   - Added `backend` parameter to `load_model()`
   - Conditional import and routing to MLX backend

3. **`whisperx/__main__.py`** - CLI modifications
   - Added `--backend` argument with choices ["faster-whisper", "mlx"]

4. **`whisperx/transcribe.py`** - Environment variable support
   - Auto-loads `.env` file if present
   - Reads `HF_TOKEN` from environment

### Adding New Features

To extend the MLX backend:

1. Modify `whisperx/mlx_asr.py`:
```python
class MLXWhisperModel:
    def your_new_method(self):
        # Implementation
        pass
```

2. Update the pipeline if needed:
```python
class MLXWhisperPipeline(Pipeline):
    def transcribe(self, audio, your_new_param=None):
        # Extended implementation
        pass
```

3. Add tests in `test_mlx_integration.py`

### Running Tests

```bash
# Run all tests
python test_mlx_integration.py

# Test specific functionality
python -m pytest tests/test_mlx.py -v
```

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'mlx'
```bash
# Solution: Install MLX
pip install mlx mlx-whisper
```

#### 2. NumPy Version Conflicts
```bash
# WhisperX wants numpy>=2.0, but pyannote needs <2.0
# Solution: Use numpy 1.26.4
pip install "numpy<2.0"
```

#### 3. MPS Fallback Warnings
```bash
# Set environment variable
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### 4. Hugging Face Token Issues
```bash
# Check if token is set
echo $HF_TOKEN

# Or use --hf_token flag
whisperx audio.mp3 --hf_token your_token_here --diarize
```

#### 5. Out of Memory
- Reduce batch_size: `--batch_size 8`
- Use smaller model: `--model mlx-community/whisper-base`
- Reduce chunk_size: `--chunk_size 30`

### Performance Tuning

For Apple Silicon optimization:

1. **M1/M2 Base Models** (8-16GB RAM):
```bash
--batch_size 8 --chunk_size 30 --model mlx-community/whisper-base
```

2. **M1/M2 Pro/Max** (16-32GB RAM):
```bash
--batch_size 16 --chunk_size 45 --model mlx-community/whisper-medium-mlx
```

3. **M3/M4 Max** (64-128GB RAM):
```bash
--batch_size 32 --chunk_size 60 --model mlx-community/whisper-large-v3-mlx
```

## Architecture

### Pipeline Overview

```
Audio Input
    ↓
[VAD Processing] → Silence removal, segment detection
    ↓
[MLX Transcription] → GPU-accelerated speech-to-text
    ↓
[Alignment] → Word-level timestamp alignment
    ↓
[Diarization] → Speaker identification (optional)
    ↓
Output (JSON/SRT/VTT/TXT)
```

### Device Management

- **Transcription**: MLX (Apple GPU)
- **Alignment**: MPS when possible, CPU fallback
- **Diarization**: CPU (due to PyAnnote compatibility)
- **VAD**: CPU

### Memory Usage

Approximate memory requirements:
- Tiny model: ~1GB
- Base model: ~1.5GB
- Small model: ~2GB
- Medium model: ~5GB
- Large-v3 model: ~10GB

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

- GitHub Issues: [github.com/m-bain/whisperX/issues](https://github.com/m-bain/whisperX/issues)
- WhisperX Paper: [arxiv.org/abs/2303.00747](https://arxiv.org/abs/2303.00747)
- MLX Documentation: [ml-explore.github.io/mlx](https://ml-explore.github.io/mlx)

## License

This MLX integration maintains the same BSD-2-Clause license as WhisperX.