# WhisperX MLX Integration Plan for Apple Silicon M4 Max

## Executive Summary

This document outlines the integration of MLX (Apple's machine learning framework) with WhisperX to leverage GPU acceleration on Apple Silicon while maintaining full diarization capabilities. The goal is to create a hybrid pipeline that uses MLX for transcription (10x faster) while preserving WhisperX's alignment and speaker diarization features.

## Current State Analysis

### WhisperX Architecture
- **Transcription**: Uses faster-whisper (CTranslate2) - CPU only on Apple Silicon
- **Alignment**: Uses wav2vec2 models via PyTorch
- **Diarization**: Uses pyannote-audio for speaker identification
- **Pipeline**: VAD → Transcription → Alignment → Diarization

### Limitations on Apple Silicon
1. faster-whisper doesn't support Apple GPU (Metal/MPS)
2. PyAnnote has MPS compatibility issues (wrong timestamps)
3. Some PyTorch operations fall back to CPU from MPS

### MLX Advantages
- Native Apple Silicon GPU support
- 10x faster than CPU for Whisper inference
- Optimized memory usage
- Support for large-v3 model

## Proposed Solution

### Architecture Overview

```
Audio Input
    ↓
[VAD Processing] (WhisperX - CPU)
    ↓
[Transcription] (MLX - GPU)  ← NEW
    ↓
[Alignment] (WhisperX - MPS/CPU)
    ↓
[Diarization] (WhisperX - CPU)
    ↓
Output with timestamps & speakers
```

### Implementation Strategy

#### Phase 1: Environment Setup
```bash
# Create new environment
micromamba create -n whisper python=3.11 -y
micromamba activate whisper

# Install dependencies
pip install mlx>=0.5.0 mlx-whisper>=0.2.0
pip install whisperx
pip install pyannote-audio==3.0.0  # Version without onnxruntime-gpu
pip install torch torchvision torchaudio

# Environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### Phase 2: MLX Integration Module

**File: `whisperx/mlx_asr.py`**

```python
import mlx_whisper
import numpy as np
from typing import Optional, Union, List
from whisperx.types import TranscriptionResult, SingleSegment
from whisperx.vads import Vad

class MLXWhisperPipeline:
    """
    MLX-based Whisper pipeline compatible with WhisperX interface
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "mlx",  # Always MLX for this backend
        batch_size: int = 8,
        compute_type: str = "float16",
        options: dict = None,
        language: Optional[str] = None,
        vad_model: Optional[Vad] = None,
        vad_params: dict = None,
        suppress_numerals: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.language = language
        self.vad_model = vad_model
        self.vad_params = vad_params or {}
        self.suppress_numerals = suppress_numerals
        
        # Load MLX model
        self.model = self._load_mlx_model(model_name)
        
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size: int = 30,
        print_progress: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio using MLX Whisper with VAD preprocessing
        """
        # Implementation details...
```

#### Phase 3: Modify ASR Loader

**File: `whisperx/asr.py` (modifications)**

```python
def load_model(
    whisper_arch: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad] = None,
    vad_method: str = "pyannote",
    vad_options: Optional[dict] = None,
    backend: str = "faster-whisper",  # NEW parameter
    task: str = "transcribe",
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    threads: int = 4,
) -> Union[FasterWhisperPipeline, MLXWhisperPipeline]:
    """
    Load a Whisper model with specified backend
    """
    if backend == "mlx":
        from whisperx.mlx_asr import MLXWhisperPipeline
        return MLXWhisperPipeline(
            model_name=whisper_arch,
            batch_size=asr_options.get("batch_size", 8),
            compute_type=compute_type,
            options=asr_options,
            language=language,
            vad_model=vad_model,
            vad_params=vad_options,
        )
    else:
        # Existing faster-whisper implementation
        ...
```

#### Phase 4: CLI Integration

**File: `whisperx/__main__.py` (modifications)**

```python
parser.add_argument(
    "--backend", 
    default="faster-whisper", 
    choices=["faster-whisper", "mlx"],
    help="Backend to use for transcription"
)
parser.add_argument(
    "--mlx-model",
    default=None,
    help="MLX model path (e.g., mlx-community/whisper-large-v3-mlx)"
)
```

### Technical Considerations

#### 1. Segment Format Compatibility
MLX Whisper output must match WhisperX format:
```python
{
    "segments": [
        {
            "text": "transcribed text",
            "start": 0.0,
            "end": 2.5,
            "id": 0
        }
    ],
    "language": "en"
}
```

#### 2. Device Management
- Transcription: MLX (GPU)
- Alignment: MPS with CPU fallback
- Diarization: CPU (due to PyAnnote MPS issues)

#### 3. Memory Optimization
With 128GB unified memory:
- Batch size: 32-64 (vs default 8)
- Chunk size: 60s (vs default 30s)
- Model: large-v3 feasible

#### 4. Error Handling
- Graceful fallback to faster-whisper if MLX fails
- Clear error messages for missing MLX dependencies
- Device compatibility checks

### Performance Expectations

| Stage | Current (CPU) | With MLX | Speedup |
|-------|--------------|----------|---------|
| Transcription | 1x | 10x | 10x |
| Alignment | 1x | 1-1.5x | Minimal |
| Diarization | 1x | 1x | None |
| **Overall** | **1x** | **3-5x** | **Significant** |

### Testing Strategy

#### 1. Unit Tests
- MLX pipeline initialization
- Segment format conversion
- Device detection
- Error handling

#### 2. Integration Tests
- Full pipeline execution
- Accuracy comparison
- Performance benchmarks
- Memory usage profiling

#### 3. Example Usage
```bash
# Basic usage with MLX
whisperx audio.wav --backend mlx --model mlx-community/whisper-large-v3-mlx

# Full pipeline with optimization
whisperx audio.wav \
  --backend mlx \
  --model mlx-community/whisper-large-v3-mlx \
  --align_model WAV2VEC2_ASR_LARGE_LV60K_960H \
  --batch_size 32 \
  --chunk_size 60 \
  --compute_type float16 \
  --diarize \
  --hf_token YOUR_TOKEN
```

### Implementation Timeline

1. **Week 1**: Environment setup and MLX pipeline core
2. **Week 2**: Integration with WhisperX pipeline
3. **Week 3**: Testing and optimization
4. **Week 4**: Documentation and examples

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MLX API changes | High | Pin MLX version, monitor updates |
| Segment format mismatch | Medium | Comprehensive format conversion |
| Memory spikes | Low | Implement batch size auto-tuning |
| Accuracy differences | Medium | Extensive testing against baseline |

### Future Enhancements

1. **MLX-optimized alignment**: Port wav2vec2 to MLX
2. **MLX diarization**: When PyAnnote MLX support available
3. **Streaming support**: Real-time transcription
4. **Model quantization**: 4-bit models for even faster inference

## Conclusion

This integration will provide significant performance improvements for WhisperX on Apple Silicon while maintaining full functionality. The modular approach ensures backward compatibility and allows users to choose between backends based on their hardware and requirements.