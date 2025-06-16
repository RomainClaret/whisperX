#!/usr/bin/env python3
"""
Benchmark script for WhisperX on Apple M4 Max
Shows real performance metrics and optimization opportunities
"""

import time
import os
import sys
import psutil
import whisperx
import numpy as np
from datetime import datetime
import json

class M4MaxBenchmark:
    def __init__(self):
        self.results = {
            "hardware": {
                "model": "Apple M4 Max",
                "memory": "128GB Unified",
                "cpu_cores": psutil.cpu_count(logical=False),
                "timestamp": datetime.now().isoformat()
            },
            "benchmarks": {}
        }
        
    def create_test_audio(self, duration_seconds=30, sample_rate=16000):
        """Create test audio file"""
        # Generate sine wave with speech-like variations
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        frequency = 440 + 100 * np.sin(0.1 * t)  # Varying frequency
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Add some silence periods (like real speech)
        for i in range(5):
            start = int(i * duration_seconds * sample_rate / 5)
            end = start + int(0.5 * sample_rate)  # 0.5s silence
            audio[start:end] = 0
            
        return audio.astype(np.float32)
    
    def benchmark_transcription(self, audio, backend="mlx", model_name="mlx-community/whisper-large-v3-mlx"):
        """Benchmark transcription performance"""
        print(f"\n{'='*60}")
        print(f"Benchmarking {backend} backend with {model_name}")
        print(f"{'='*60}")
        
        # Model loading time
        start = time.time()
        model = whisperx.load_model(
            model_name if backend == "mlx" else "large-v3",
            device="cpu",
            backend=backend,
            compute_type="float16" if backend == "mlx" else "int8"
        )
        load_time = time.time() - start
        print(f"Model load time: {load_time:.2f}s")
        
        # Test different batch sizes
        batch_sizes = [8, 16, 32, 48, 64] if backend == "mlx" else [1, 4, 8, 16]
        
        results = {
            "backend": backend,
            "model": model_name,
            "load_time": load_time,
            "batch_results": {}
        }
        
        for batch_size in batch_sizes:
            try:
                # Warm-up run
                _ = model.transcribe(audio[:16000], batch_size=batch_size, verbose=False)
                
                # Actual benchmark
                start = time.time()
                result = model.transcribe(audio, batch_size=batch_size, verbose=False)
                transcribe_time = time.time() - start
                
                audio_duration = len(audio) / 16000
                rtf = audio_duration / transcribe_time  # Real-time factor
                
                print(f"\nBatch size: {batch_size}")
                print(f"  Transcription time: {transcribe_time:.2f}s")
                print(f"  Real-time factor: {rtf:.1f}x")
                print(f"  Segments: {len(result['segments'])}")
                
                results["batch_results"][batch_size] = {
                    "time": transcribe_time,
                    "rtf": rtf,
                    "segments": len(result['segments'])
                }
                
            except Exception as e:
                print(f"  Failed with batch_size={batch_size}: {e}")
                results["batch_results"][batch_size] = {"error": str(e)}
        
        # Memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        results["peak_memory_gb"] = memory_info.rss / (1024**3)
        print(f"\nPeak memory usage: {results['peak_memory_gb']:.2f} GB")
        
        # Cleanup
        del model
        import gc
        gc.collect()
        
        return results
    
    def benchmark_alignment(self, audio, segments):
        """Benchmark alignment performance"""
        print(f"\n{'='*60}")
        print(f"Benchmarking Alignment")
        print(f"{'='*60}")
        
        devices = ["cpu", "mps"]
        results = {}
        
        for device in devices:
            try:
                print(f"\nTesting device: {device}")
                
                # Load alignment model
                start = time.time()
                model_a, metadata = whisperx.load_align_model(
                    language_code="en",
                    device=device,
                    model_name="WAV2VEC2_ASR_LARGE_LV60K_960H"
                )
                load_time = time.time() - start
                print(f"  Model load time: {load_time:.2f}s")
                
                # Perform alignment
                start = time.time()
                aligned = whisperx.align(
                    segments,
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False
                )
                align_time = time.time() - start
                print(f"  Alignment time: {align_time:.2f}s")
                
                results[device] = {
                    "load_time": load_time,
                    "align_time": align_time,
                    "success": True
                }
                
                # Cleanup
                del model_a
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"  Failed on {device}: {e}")
                results[device] = {"error": str(e), "success": False}
        
        return results
    
    def benchmark_full_pipeline(self, audio):
        """Benchmark complete pipeline"""
        print(f"\n{'='*60}")
        print(f"Benchmarking Full Pipeline")
        print(f"{'='*60}")
        
        start_total = time.time()
        
        # 1. Transcription
        start = time.time()
        model = whisperx.load_model(
            "mlx-community/whisper-large-v3-mlx",
            device="cpu",
            backend="mlx",
            compute_type="float16"
        )
        result = model.transcribe(audio, batch_size=48)
        transcribe_time = time.time() - start
        print(f"Transcription: {transcribe_time:.2f}s")
        
        # 2. Alignment
        start = time.time()
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device="cpu"
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            "cpu"
        )
        align_time = time.time() - start
        print(f"Alignment: {align_time:.2f}s")
        
        # 3. Diarization (if token available)
        diarize_time = 0
        if os.environ.get("HF_TOKEN"):
            start = time.time()
            diarize_model = whisperx.DiarizationPipeline(device="cpu")
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            diarize_time = time.time() - start
            print(f"Diarization: {diarize_time:.2f}s")
        else:
            print("Diarization: Skipped (no HF_TOKEN)")
        
        total_time = time.time() - start_total
        
        return {
            "transcribe_time": transcribe_time,
            "align_time": align_time,
            "diarize_time": diarize_time,
            "total_time": total_time,
            "transcribe_percent": (transcribe_time / total_time) * 100,
            "align_percent": (align_time / total_time) * 100,
            "diarize_percent": (diarize_time / total_time) * 100 if diarize_time > 0 else 0
        }
    
    def run_all_benchmarks(self, duration_seconds=30):
        """Run all benchmarks"""
        print("WhisperX M4 Max Benchmark Suite")
        print("================================")
        
        # Create test audio
        print(f"\nCreating {duration_seconds}s test audio...")
        audio = self.create_test_audio(duration_seconds)
        
        # Benchmark MLX backend
        mlx_results = self.benchmark_transcription(
            audio,
            backend="mlx",
            model_name="mlx-community/whisper-large-v3-mlx"
        )
        self.results["benchmarks"]["mlx_transcription"] = mlx_results
        
        # Get segments for alignment testing
        model = whisperx.load_model(
            "mlx-community/whisper-large-v3-mlx",
            device="cpu",
            backend="mlx"
        )
        result = model.transcribe(audio, batch_size=32, verbose=False)
        segments = result["segments"]
        
        # Benchmark alignment
        align_results = self.benchmark_alignment(audio, segments)
        self.results["benchmarks"]["alignment"] = align_results
        
        # Benchmark full pipeline
        pipeline_results = self.benchmark_full_pipeline(audio)
        self.results["benchmarks"]["full_pipeline"] = pipeline_results
        
        # Summary
        self.print_summary()
        
        # Save results
        with open("benchmark_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("\nResults saved to benchmark_results.json")
        
    def print_summary(self):
        """Print benchmark summary"""
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        # Best batch size for MLX
        mlx_batches = self.results["benchmarks"]["mlx_transcription"]["batch_results"]
        best_batch = max(mlx_batches.items(), 
                        key=lambda x: x[1].get("rtf", 0) if "rtf" in x[1] else 0)
        print(f"\nOptimal MLX batch size: {best_batch[0]}")
        print(f"  Real-time factor: {best_batch[1].get('rtf', 0):.1f}x")
        
        # Pipeline breakdown
        pipeline = self.results["benchmarks"]["full_pipeline"]
        print(f"\nPipeline time breakdown:")
        print(f"  Transcription: {pipeline['transcribe_percent']:.1f}%")
        print(f"  Alignment: {pipeline['align_percent']:.1f}%")
        print(f"  Diarization: {pipeline['diarize_percent']:.1f}%")
        
        # Memory usage
        mlx_memory = self.results["benchmarks"]["mlx_transcription"]["peak_memory_gb"]
        print(f"\nPeak memory usage: {mlx_memory:.2f} GB")
        print(f"Memory headroom: {128 - mlx_memory:.1f} GB")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS FOR YOUR M4 MAX")
        print(f"{'='*60}")
        print(f"1. Use batch_size={best_batch[0]} for optimal performance")
        print(f"2. You have {128 - mlx_memory:.0f}GB headroom for larger models/batches")
        print(f"3. Transcription is well optimized ({pipeline['transcribe_percent']:.0f}% of time)")
        
        if pipeline['align_percent'] > 30:
            print("4. Consider skipping alignment if word-level timestamps not needed")
        else:
            print("4. Alignment overhead is acceptable")


def main():
    # Check environment
    if "whisper" not in os.environ.get("CONDA_DEFAULT_ENV", ""):
        print("Warning: Not in whisper environment")
        print("Run: micromamba activate whisper")
        
    # Set optimal environment
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "8"
    
    # Run benchmarks
    benchmark = M4MaxBenchmark()
    
    # You can adjust duration for longer tests
    duration = 30  # seconds
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
        
    benchmark.run_all_benchmarks(duration)


if __name__ == "__main__":
    main()