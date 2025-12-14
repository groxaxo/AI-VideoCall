"""Parakeet ASR with NeMo for local CUDA inference."""

import logging
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


@dataclass
class ParakeetConfig:
    """Configuration for Parakeet ASR."""

    model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    device: str = "cuda:0"  # GPU 0 with interface
    use_amp: bool = True
    sample_rate: int = 16000


class ParakeetASR:
    """
    Parakeet ASR using NeMo for local CUDA-accelerated speech recognition.
    
    This provides local ASR without external API calls, running on GPU.
    """

    def __init__(self, cfg: ParakeetConfig):
        """
        Initialize Parakeet ASR.
        
        Args:
            cfg: ParakeetConfig configuration
        """
        self.cfg = cfg

        try:
            # Import NeMo ASR module
            import nemo.collections.asr as nemo_asr

            logger.info(
                f"Loading Parakeet model {cfg.model_name} on device {cfg.device}..."
            )

            # Load pretrained model
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=cfg.model_name
            )
            self.model = self.model.to(torch.device(cfg.device))
            self.model.eval()

            logger.info("Parakeet ASR initialized successfully")

        except ImportError as e:
            logger.error(
                f"NeMo ASR not available. Install with: pip install nemo_toolkit[asr]"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Parakeet ASR: {e}")
            raise

    def transcribe_pcm16(self, pcm16: bytes, sr: int = 16000) -> str:
        """
        Transcribe PCM16 audio data.
        
        Args:
            pcm16: PCM16 audio data as bytes
            sr: Sample rate of the audio (default: 16000)
            
        Returns:
            Transcribed text
        """
        try:
            # Convert PCM16 to float32
            audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample if needed
            if sr != self.cfg.sample_rate:
                # Simple linear interpolation resampling
                # Note: For production, consider using librosa.resample or 
                # scipy.signal.resample for higher quality resampling
                duration = len(audio) / sr
                target_length = int(duration * self.cfg.sample_rate)
                indices = np.linspace(0, len(audio) - 1, target_length)
                audio = np.interp(indices, np.arange(len(audio)), audio)
                sr = self.cfg.sample_rate

            return self.transcribe_audio(audio, sr)

        except Exception as e:
            logger.error(f"Error transcribing PCM16 audio: {e}")
            return ""

    def transcribe_audio(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        Transcribe float32 audio data.
        
        Args:
            audio: Audio as float32 numpy array
            sr: Sample rate of the audio (default: 16000)
            
        Returns:
            Transcribed text
        """
        try:
            # NeMo expects audio from file, so we write to temp file
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=True, mode="wb"
            ) as f:
                sf.write(f.name, audio, sr)

                # Transcribe with NeMo
                with torch.no_grad():
                    if self.cfg.use_amp:
                        with torch.cuda.amp.autocast():
                            out = self.model.transcribe([f.name])
                    else:
                        out = self.model.transcribe([f.name])

                # Extract text from result
                if out and len(out) > 0:
                    if hasattr(out[0], "text"):
                        return out[0].text.strip()
                    elif isinstance(out[0], str):
                        return out[0].strip()
                    else:
                        return str(out[0]).strip()

                return ""

        except Exception as e:
            logger.error(f"Error in Parakeet transcription: {e}")
            return ""

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "model"):
            del self.model
            # Note: Removed torch.cuda.empty_cache() as it's a performance footgun in destructors
