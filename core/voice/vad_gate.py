"""Voice Activity Detection (VAD) gate for detecting speech pauses."""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class VADConfig:
    """Configuration for VAD gate."""

    sample_rate: int = 16000
    frame_duration_ms: int = 20  # 20ms frames
    energy_threshold: float = 0.001  # RMS energy threshold for speech
    min_speech_frames: int = 10  # Minimum frames to consider speech started
    min_silence_frames: int = 15  # Minimum silence frames to consider pause


class VADGate:
    """
    Simple energy-based Voice Activity Detection.
    Detects speech vs silence based on RMS energy.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        """
        Initialize VAD gate.
        
        Args:
            config: VAD configuration
        """
        self.config = config or VADConfig()
        self.frame_size = int(
            self.config.sample_rate * self.config.frame_duration_ms / 1000
        )

        self.is_speech_active = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0

    def process_frame(self, audio_frame: np.ndarray) -> tuple[bool, bool]:
        """
        Process a single audio frame.
        
        Args:
            audio_frame: Audio frame (float32, mono)
            
        Returns:
            Tuple of (is_speech, pause_detected)
            - is_speech: True if current frame contains speech or speech is active
            - pause_detected: True if a significant pause was detected
        """
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_frame**2))
        frame_has_speech = rms > self.config.energy_threshold

        pause_detected = False

        if frame_has_speech:
            self.speech_frame_count += 1
            self.silence_frame_count = 0

            # Start speech activity if we have enough consecutive speech frames
            if (
                not self.is_speech_active
                and self.speech_frame_count >= self.config.min_speech_frames
            ):
                self.is_speech_active = True

        else:
            self.silence_frame_count += 1
            self.speech_frame_count = 0

            # Detect pause if we were speaking and now have enough silence
            if (
                self.is_speech_active
                and self.silence_frame_count >= self.config.min_silence_frames
            ):
                pause_detected = True
                self.is_speech_active = False

        # Return True for is_speech if currently active OR frame contains speech
        # This ensures we don't miss speech at boundaries
        return self.is_speech_active or frame_has_speech, pause_detected

    def process_audio(self, audio: np.ndarray) -> list[tuple[bool, bool]]:
        """
        Process multiple frames of audio.
        
        Args:
            audio: Audio data (float32, mono)
            
        Returns:
            List of (is_speech, pause_detected) tuples for each frame
        """
        results = []
        num_frames = len(audio) // self.frame_size

        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio[start:end]
            results.append(self.process_frame(frame))

        return results

    def reset(self):
        """Reset VAD state."""
        self.is_speech_active = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0
