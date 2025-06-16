"""
MLX-based ASR module for WhisperX
Provides GPU-accelerated transcription on Apple Silicon
"""
import os
import sys
import warnings
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, replace
import numpy as np

try:
    import mlx
    import mlx_whisper
except ImportError:
    raise ImportError(
        "MLX and mlx-whisper are required for MLX backend. "
        "Install with: pip install mlx mlx-whisper"
    )

import torch
from transformers import Pipeline

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio
from whisperx.types import SingleSegment, TranscriptionResult
from whisperx.vads import Vad, Silero, Pyannote
from whisperx.asr import WhisperModel


class MLXWhisperModel:
    """
    MLX Whisper model wrapper that mimics faster-whisper interface
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "mlx",
        compute_type: str = "float16",
        download_root: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        
        # Load MLX model
        print(f"Loading MLX Whisper model: {model_name}")
        self.model = mlx_whisper.load_models.load_model(model_name)
        
        # Model properties for compatibility
        self.is_multilingual = "large" in model_name or "medium" in model_name or "tiny" not in model_name
        
    def transcribe_segment(
        self,
        audio_segment: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = False,
        suppress_tokens: Optional[List[int]] = None,
        word_timestamps: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio segment using MLX Whisper
        """
        # MLX Whisper expects audio as float32 numpy array
        if audio_segment.dtype != np.float32:
            audio_segment = audio_segment.astype(np.float32)
            
        # Prepare options for MLX
        decode_options = {
            "language": language,
            "task": task,
            "temperature": temperature,
            "initial_prompt": initial_prompt,
            "suppress_tokens": suppress_tokens or [],
            "condition_on_previous_text": condition_on_previous_text,
            "word_timestamps": word_timestamps,
            "verbose": verbose,
        }
        
        # Remove None values
        decode_options = {k: v for k, v in decode_options.items() if v is not None}
        
        # Transcribe with MLX
        result = mlx_whisper.transcribe(
            audio_segment,
            path_or_hf_repo=self.model_name,
            **decode_options
        )
        
        return result
    
    def detect_language(self, audio: np.ndarray) -> str:
        """
        Detect language from audio sample
        """
        # Use first 30 seconds for language detection
        sample = audio[:N_SAMPLES] if len(audio) > N_SAMPLES else audio
        
        # MLX language detection
        result = mlx_whisper.transcribe(
            sample,
            path_or_hf_repo=self.model_name,
            task="transcribe",
            temperature=0.0,
            verbose=False,
        )
        
        language = result.get("language", "en")
        print(f"Detected language: {language}")
        return language


class MLXWhisperPipeline(Pipeline):
    """
    WhisperX-compatible pipeline using MLX backend
    """
    
    def __init__(
        self,
        model: MLXWhisperModel,
        vad,
        vad_params: dict,
        options: dict,
        tokenizer=None,
        device: Union[int, str, torch.device] = -1,
        framework="pt",
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        **kwargs,
    ):
        self.model = model
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", 8)
        self._num_workers = 1
        self.framework = framework
        
        # For compatibility with WhisperX
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device if device != "mlx" else "cpu")
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")
            
        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params
        
        # Set model properties for compatibility
        # MLX models have read-only properties, so we skip this step
            
    def _sanitize_parameters(self, **kwargs):
        """Required by Pipeline abstract class"""
        return {}, {}, {}
        
    def preprocess(self, inputs):
        """Required by Pipeline abstract class"""
        return inputs
        
    def _forward(self, model_inputs):
        """Required by Pipeline abstract class"""
        return model_inputs
        
    def postprocess(self, model_outputs):
        """Required by Pipeline abstract class"""
        return model_outputs
        
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio using MLX Whisper with VAD preprocessing
        """
        if isinstance(audio, str):
            audio = load_audio(audio)
            
        # Detect language if not specified
        if language is None and self.preset_language is None:
            language = self.model.detect_language(audio)
        else:
            language = language or self.preset_language
            
        task = task or "transcribe"
        
        # Process audio with VAD
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            # Fallback for pyannote VAD
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks
            
        # Get VAD segments
        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params.get("vad_onset", 0.5),
            offset=self._vad_params.get("vad_offset", 0.363),
        )
        
        # Transcribe each segment
        segments: List[SingleSegment] = []
        total_segments = len(vad_segments)
        
        for idx, vad_segment in enumerate(vad_segments):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
                
            # Extract audio segment
            start_sample = int(vad_segment['start'] * SAMPLE_RATE)
            end_sample = int(vad_segment['end'] * SAMPLE_RATE)
            audio_segment = audio[start_sample:end_sample]
            
            # Transcribe with MLX
            result = self.model.transcribe_segment(
                audio_segment,
                language=language,
                task=task,
                initial_prompt=self.options.get("initial_prompt"),
                temperature=self.options.get("temperatures", [0.0])[0],
                compression_ratio_threshold=self.options.get("compression_ratio_threshold", 2.4),
                log_prob_threshold=self.options.get("log_prob_threshold", -1.0),
                no_speech_threshold=self.options.get("no_speech_threshold", 0.6),
                condition_on_previous_text=self.options.get("condition_on_previous_text", False),
                suppress_tokens=self.options.get("suppress_tokens", []),
                verbose=verbose,
            )
            
            # Extract text from result
            text = result.get("text", "").strip()
            
            if verbose:
                print(f"Segment [{vad_segment['start']:.3f} --> {vad_segment['end']:.3f}]: {text}")
                
            # Add segment to results
            segments.append({
                "text": text,
                "start": round(vad_segment['start'], 3),
                "end": round(vad_segment['end'], 3),
            })
            
        return {
            "segments": segments,
            "language": language,
        }
        
    def detect_language(self, audio: np.ndarray) -> str:
        """
        Detect language from audio
        """
        return self.model.detect_language(audio)


def load_mlx_model(
    model_name: str,
    device: str = "mlx",
    compute_type: str = "float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad] = None,
    vad_method: str = "pyannote",
    vad_options: Optional[dict] = None,
    task: str = "transcribe",
    download_root: Optional[str] = None,
    suppress_numerals: bool = False,
) -> MLXWhisperPipeline:
    """
    Load MLX Whisper model for inference
    """
    # Load MLX model
    model = MLXWhisperModel(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )
    
    # Default ASR options
    default_asr_options = {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "suppress_tokens": [-1],
        "suppress_numerals": suppress_numerals,
    }
    
    if asr_options is not None:
        default_asr_options.update(asr_options)
        
    # Default VAD options
    default_vad_options = {
        "chunk_size": 30,
        "vad_onset": 0.500,
        "vad_offset": 0.363,
    }
    
    if vad_options is not None:
        default_vad_options.update(vad_options)
        
    # Initialize VAD model
    if vad_model is not None:
        print("Using provided VAD model")
    else:
        if vad_method == "silero":
            vad_model = Silero(**default_vad_options)
        elif vad_method == "pyannote":
            # Use CPU for PyAnnote due to MPS issues
            vad_device = torch.device("cpu")
            vad_model = Pyannote(vad_device, use_auth_token=None, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")
            
    # Create pipeline
    return MLXWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        vad_params=default_vad_options,
        language=language,
        suppress_numerals=suppress_numerals,
        device=device,
    )