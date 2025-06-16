# WhisperX Documentation

Welcome to the WhisperX documentation! This folder contains all documentation for WhisperX, with special focus on Apple Silicon optimization using MLX.

## 📚 Documentation Structure

### 🚀 [/quickstart](./quickstart/)
Get up and running quickly with WhisperX.
- **[QUICKSTART.md](./quickstart/QUICKSTART.md)** - 5-minute setup guide for Apple Silicon
- **[WHISPERX_MLX_SETUP.md](./quickstart/WHISPERX_MLX_SETUP.md)** - Detailed installation instructions

### 🔧 [/technical](./technical/)
Technical details and implementation documentation.
- **[MLX_INTEGRATION_PLAN.md](./technical/MLX_INTEGRATION_PLAN.md)** - Architecture and design decisions
- **[CURRENT_LIMITATIONS_AND_FIXES.md](./technical/CURRENT_LIMITATIONS_AND_FIXES.md)** - Known issues and workarounds
- **[MLX_TEST_RESULTS.md](./technical/MLX_TEST_RESULTS.md)** - Initial test results
- **[README_MLX_INTEGRATION.md](./technical/README_MLX_INTEGRATION.md)** - What was changed and why

### 📖 [/guides](./guides/)
In-depth guides for developers and contributors.
- **[AI_DEVELOPER_GUIDE.md](./guides/AI_DEVELOPER_GUIDE.md)** - Guide for AI assistants and developers
- **[MLX_MODEL_CONVERSION_GUIDE.md](./guides/MLX_MODEL_CONVERSION_GUIDE.md)** - How to convert models to MLX
- **[APPLE_SILICON_OPTIMIZATION_ROADMAP.md](./guides/APPLE_SILICON_OPTIMIZATION_ROADMAP.md)** - Future optimization analysis

### 📊 [/benchmarks](./benchmarks/)
Performance testing and benchmarking tools.
- **[benchmark_m4_max.py](./benchmarks/benchmark_m4_max.py)** - Performance benchmark script for Apple Silicon

## 🎯 Quick Navigation

### "I want to..."

#### Get started with WhisperX on my Mac
→ Start with [quickstart/QUICKSTART.md](./quickstart/QUICKSTART.md)

#### Understand the technical implementation
→ Read [technical/MLX_INTEGRATION_PLAN.md](./technical/MLX_INTEGRATION_PLAN.md)

#### Fix an issue or contribute code
→ See [guides/AI_DEVELOPER_GUIDE.md](./guides/AI_DEVELOPER_GUIDE.md)

#### Know current limitations
→ Check [technical/CURRENT_LIMITATIONS_AND_FIXES.md](./technical/CURRENT_LIMITATIONS_AND_FIXES.md)

#### Convert my own models to MLX
→ Follow [guides/MLX_MODEL_CONVERSION_GUIDE.md](./guides/MLX_MODEL_CONVERSION_GUIDE.md)

#### Benchmark performance on my machine
→ Run `python docs/benchmarks/benchmark_m4_max.py`

## 📝 Key Documents at Root Level

These important documents remain at the repository root for easy access:
- **[EXECUTIVE_SUMMARY_APPLE_SILICON.md](../EXECUTIVE_SUMMARY_APPLE_SILICON.md)** - High-level overview for decision makers
- **[PROJECT_STATUS_SUMMARY.md](../PROJECT_STATUS_SUMMARY.md)** - Current project status and achievements
- **[DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md)** - Legacy index (use this README instead)

## 🚀 Quick Start Command

```bash
# One-line setup for Apple Silicon
./setup_mlx_whisperx.sh

# Run with MLX acceleration
whisperx audio.mp3 --backend mlx --model mlx-community/whisper-large-v3-mlx
```

## 🤝 Contributing

When adding new documentation:
1. Place it in the appropriate subfolder
2. Update this README with a link
3. Use clear, descriptive filenames
4. Include a table of contents for long documents

## 📊 Documentation Standards

- Use Markdown format
- Include code examples where relevant
- Add emoji sparingly for visual navigation
- Keep technical accuracy as the top priority
- Update dates when making significant changes

---

*Last updated: November 2024*