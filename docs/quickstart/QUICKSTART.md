# WhisperX MLX Quick Start Guide

Get WhisperX running with Apple Silicon GPU acceleration in 5 minutes!

## 🚀 One-Line Install

```bash
git clone https://github.com/m-bain/whisperX.git && cd whisperX && ./setup_mlx_whisperx.sh
```

## 📋 Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 12.0+
- 16GB+ RAM recommended

## 🎯 Quick Usage

### 1. Activate Environment
```bash
micromamba activate whisper
```

### 2. Basic Transcription
```bash
# Fast transcription with GPU
whisperx audio.mp3 --backend mlx --model mlx-community/whisper-large-v3-mlx
```

### 3. With Speaker Detection
```bash
# First, set up your Hugging Face token
cp .env.example .env
# Edit .env and add your token

# Then run with diarization
whisperx audio.mp3 --backend mlx --model mlx-community/whisper-large-v3-mlx --diarize
```

## 🔧 Common Commands

### Transcribe with Timestamps
```bash
whisperx audio.mp3 --backend mlx --model mlx-community/whisper-large-v3-mlx --output_format srt
```

### Process Multiple Files
```bash
whisperx *.mp3 --backend mlx --model mlx-community/whisper-large-v3-mlx --output_dir transcripts/
```

### Optimize for Large Files (M4 Max)
```bash
whisperx large_audio.mp3 --backend mlx --model mlx-community/whisper-large-v3-mlx --batch_size 32 --chunk_size 60
```

## 🐍 Python Usage

```python
import whisperx

# Load model
model = whisperx.load_model("mlx-community/whisper-large-v3-mlx", device="cpu", backend="mlx")

# Transcribe
audio = whisperx.load_audio("audio.mp3")
result = model.transcribe(audio, batch_size=16)

# Print results
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s] {segment['text']}")
```

## 📊 Model Selection

| Model | Speed | Accuracy | RAM Usage | Use Case |
|-------|-------|----------|-----------|----------|
| `tiny` | ⚡⚡⚡⚡⚡ | ⭐⭐ | ~1GB | Real-time, drafts |
| `base` | ⚡⚡⚡⚡ | ⭐⭐⭐ | ~1.5GB | Quick transcription |
| `small-mlx` | ⚡⚡⚡ | ⭐⭐⭐⭐ | ~2GB | Balanced |
| `medium-mlx` | ⚡⚡ | ⭐⭐⭐⭐ | ~5GB | High quality |
| `large-v3-mlx` | ⚡ | ⭐⭐⭐⭐⭐ | ~10GB | Best quality |

## ❓ Troubleshooting

### MLX not found
```bash
pip install mlx mlx-whisper
```

### Token issues for diarization
```bash
# Set token in .env file
echo "HF_TOKEN=your_token_here" > .env
```

### Out of memory
```bash
# Use smaller model or reduce batch size
whisperx audio.mp3 --backend mlx --model mlx-community/whisper-small-mlx --batch_size 4
```

## 📚 More Info

- Full setup guide: [WHISPERX_MLX_SETUP.md](WHISPERX_MLX_SETUP.md)
- Developer guide: [AI_DEVELOPER_GUIDE.md](AI_DEVELOPER_GUIDE.md)
- Technical details: [MLX_INTEGRATION_PLAN.md](MLX_INTEGRATION_PLAN.md)

## 💡 Tips

1. **Language**: Add `--language en` to skip detection and save time
2. **Output**: Use `--output_format all` to get all formats at once
3. **Progress**: Add `--verbose` to see real-time progress
4. **Quality**: Use `--align_model WAV2VEC2_ASR_LARGE_LV60K_960H` for better word timestamps

Happy transcribing! 🎉