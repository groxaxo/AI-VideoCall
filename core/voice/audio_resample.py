"""Audio resampling utilities for voice processing."""

from typing import Optional

import numpy as np


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Input audio array (float32, mono)
        orig_sr: Original sample rate
        target_sr: Target sample rate (default: 16000 Hz)

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    # Simple linear interpolation for resampling
    # For production, consider using librosa.resample or scipy.signal.resample
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)

    # Linear interpolation
    indices = np.linspace(0, len(audio) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(audio)), audio)

    return resampled.astype(np.float32)


def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    """
    Convert PCM16 bytes to float32 numpy array.

    Args:
        pcm16: PCM16 audio data as bytes

    Returns:
        Float32 numpy array normalized to [-1, 1]
    """
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """
    Convert float32 numpy array to PCM16 bytes.

    Args:
        audio: Float32 audio array in range [-1, 1]

    Returns:
        PCM16 audio data as bytes
    """
    audio_clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio_clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes()
