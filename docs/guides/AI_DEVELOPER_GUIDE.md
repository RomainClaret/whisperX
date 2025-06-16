# AI Developer Guide for WhisperX MLX Integration

This guide is specifically designed for AI assistants (like Claude, GPT-4, etc.) working on the WhisperX project with MLX backend support.

## Quick Context

**What is this project?**
- WhisperX: Enhanced Whisper ASR with word-level timestamps and speaker diarization
- MLX Integration: Apple Silicon GPU acceleration for 10x faster transcription
- Target Hardware: Apple Silicon Macs (M1/M2/M3/M4)

## Essential Information

### Repository Structure
```
whisperX/
├── whisperx/               # Main package directory
│   ├── mlx_asr.py         # MLX backend (NEW - GPU acceleration)
│   ├── asr.py             # Model loading (MODIFIED for backend selection)
│   ├── transcribe.py      # CLI task handler (MODIFIED for env vars)
│   └── __main__.py        # CLI entry (MODIFIED for --backend flag)
├── .env                   # Environment variables (GITIGNORED)
├── .env.example           # Template for users
└── setup_mlx_whisperx.sh  # One-click setup script
```

### Key Modifications Made

1. **Backend Selection**: Added `--backend` flag to choose between "faster-whisper" (CPU) and "mlx" (GPU)
2. **Environment Variables**: Auto-loads HF_TOKEN from .env file
3. **MLX Pipeline**: New MLXWhisperPipeline class that mimics FasterWhisperPipeline interface

### Critical Files to Understand

1. **`whisperx/mlx_asr.py`** - Contains:
   - `MLXWhisperModel`: Wrapper around mlx_whisper
   - `MLXWhisperPipeline`: Pipeline implementation
   - `load_mlx_model()`: Factory function

2. **`whisperx/asr.py`** - Modified `load_model()` at line 301:
   ```python
   if backend == "mlx":
       from whisperx.mlx_asr import load_mlx_model
       return load_mlx_model(...)
   ```

## Common Tasks

### Adding a New Feature to MLX Backend

1. Check if the feature exists in the original WhisperX
2. Implement in `mlx_asr.py` following the same interface
3. Test with: `whisperx audio.mp3 --backend mlx`

### Fixing Bugs

Common issues and solutions:

**NumPy Version Conflict**
```bash
# WhisperX wants >=2.0, pyannote wants <2.0
pip install "numpy<2.0"
```

**MLX API Changes**
- MLX models are loaded with: `mlx_whisper.load_models.load_model()`
- Transcription: `mlx_whisper.transcribe(audio, path_or_hf_repo=model_name)`

**Device Management**
- MLX always uses Apple GPU (no device parameter needed)
- PyTorch operations should use "mps" or "cpu"
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for compatibility

### Testing Your Changes

1. **Unit Test**: 
   ```bash
   python test_mlx_integration.py
   ```

2. **Integration Test**:
   ```bash
   whisperx test_audio.mp3 --backend mlx --model mlx-community/whisper-tiny
   ```

3. **Full Pipeline Test**:
   ```bash
   whisperx test_audio.mp3 --backend mlx --model mlx-community/whisper-large-v3-mlx --diarize --align_model WAV2VEC2_ASR_LARGE_LV60K_960H
   ```

## Important Constraints

### Do NOT:
- Modify the output format (must match original WhisperX)
- Change the Pipeline interface methods
- Remove backward compatibility
- Commit .env files

### Always:
- Maintain the same CLI interface
- Keep output JSON structure identical
- Test both backends after changes
- Handle missing MLX gracefully (fallback to faster-whisper)

## Environment Setup for Development

```bash
# 1. Create environment
micromamba create -n whisper python=3.11
micromamba activate whisper

# 2. Install dependencies
pip install -e .  # Install WhisperX in editable mode
pip install mlx mlx-whisper
pip install python-dotenv

# 3. Set up token
echo "HF_TOKEN=your_token" > .env
```

## Key APIs and Patterns

### MLX Model Loading
```python
import mlx_whisper
model = mlx_whisper.load_models.load_model("mlx-community/whisper-large-v3-mlx")
```

### Transcription Format
```python
result = {
    "segments": [
        {
            "text": "Hello world",
            "start": 0.0,
            "end": 2.0,
            "words": [...]  # Optional, from alignment
        }
    ],
    "language": "en"
}
```

### Pipeline Pattern
All pipelines must implement:
- `transcribe()` - Main transcription method
- `detect_language()` - Language detection
- Abstract methods: `preprocess()`, `_forward()`, `postprocess()`, `_sanitize_parameters()`

## Performance Considerations

- MLX is optimized for Apple Silicon - always runs on GPU
- Batch processing is key: default batch_size=8, can go up to 64 on M4 Max
- VAD preprocessing happens on CPU before GPU transcription
- Alignment uses MPS (Metal) when possible, falls back to CPU

## Debugging Tips

1. **Enable verbose mode**: `--verbose` flag shows segment-by-segment progress
2. **Check device usage**: `Activity Monitor > GPU History`
3. **Memory issues**: Reduce batch_size or use smaller model
4. **Import errors**: Ensure you're in the `whisper` environment

## Related Documentation

- Original WhisperX: https://github.com/m-bain/whisperX
- MLX Framework: https://github.com/ml-explore/mlx
- MLX Whisper: https://github.com/ml-explore/mlx-examples/tree/main/whisper
- PyAnnote: https://github.com/pyannote/pyannote-audio

## Quick Fixes Cheatsheet

```bash
# Fix numpy issues
pip install "numpy<2.0"

# Fix MPS warnings
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Test MLX is working
python -c "import mlx; print('MLX available')"

# Run with small model for testing
whisperx test.mp3 --backend mlx --model mlx-community/whisper-tiny
```

## Contact for Issues

When creating GitHub issues, include:
1. Full error traceback
2. Command used
3. macOS version and chip (e.g., "macOS 14.5, M4 Max")
4. Output of `pip list | grep -E "(whisper|mlx|torch)"`