"""Voice session management for per-client state machine."""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .audio_resample import pcm16_to_float32
from .parakeet_nemo_cuda import ParakeetASR, ParakeetConfig
from .ring_buffer import RingBuffer
from .smart_turn_onnx_cuda import SmartTurnCuda, SmartTurnCudaConfig
from .vad_gate import VADConfig, VADGate

logger = logging.getLogger(__name__)


class VoiceSessionState(str, Enum):
    """Voice session states."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"


@dataclass
class VoiceSessionConfig:
    """Configuration for voice session."""

    sample_rate: int = 16000
    buffer_max_seconds: float = 30.0  # Maximum audio buffer
    turn_check_seconds: float = 2.0  # Seconds of audio to check for turn
    enable_smart_turn: bool = True  # Enable Smart Turn detection
    
    # Note: Component-specific configs (SmartTurnCudaConfig, ParakeetConfig, VADConfig) 
    # can be passed to VoiceSession constructor directly if customization is needed


class VoiceSession:
    """
    Per-client voice session managing the state machine:
    - Buffer incoming PCM16 audio
    - Run VAD to detect speech/pauses
    - On pause, run Smart Turn to check for end-of-turn
    - If end-of-turn detected, run Parakeet ASR for final transcript
    """

    def __init__(
        self,
        config: VoiceSessionConfig,
        smart_turn: Optional[SmartTurnCuda] = None,
        parakeet_asr: Optional[ParakeetASR] = None,
        vad_config: Optional[VADConfig] = None,
    ):
        """
        Initialize voice session.
        
        Args:
            config: VoiceSessionConfig
            smart_turn: Optional SmartTurnCuda instance (shared across sessions)
            parakeet_asr: Optional ParakeetASR instance (shared across sessions)
            vad_config: Optional VADConfig for customizing VAD behavior
        """
        self.config = config
        self.state = VoiceSessionState.IDLE

        # Audio processing components
        self.ring_buffer = RingBuffer(
            max_seconds=config.buffer_max_seconds,
            sample_rate=config.sample_rate,
        )
        self.vad = VADGate(vad_config or VADConfig())
        self.smart_turn = smart_turn
        self.parakeet_asr = parakeet_asr

        # Current turn buffer (for accumulating speech)
        self.current_turn_audio = []
        self.turn_start_time = None

        logger.info("VoiceSession initialized")

    def start(self):
        """Start voice session (listening mode)."""
        self.state = VoiceSessionState.LISTENING
        self.ring_buffer.clear()
        self.current_turn_audio = []
        self.vad.reset()
        self.turn_start_time = time.time()
        logger.info("Voice session started")

    def stop(self):
        """Stop voice session."""
        self.state = VoiceSessionState.IDLE
        self.current_turn_audio = []
        logger.info("Voice session stopped")

    async def push_pcm16(self, pcm16: bytes) -> Optional[str]:
        """
        Push PCM16 audio chunk and process.
        
        Args:
            pcm16: PCM16 audio data as bytes
            
        Returns:
            Final transcript if end-of-turn detected, None otherwise
        """
        if self.state != VoiceSessionState.LISTENING:
            return None

        try:
            # Convert to float32
            audio_f32 = pcm16_to_float32(pcm16)

            # Add to ring buffer
            self.ring_buffer.append(audio_f32)

            # Add to current turn buffer
            self.current_turn_audio.append(audio_f32)

            # Process with VAD
            is_speech, pause_detected = self._process_vad(audio_f32)

            # If pause detected, check for end of turn
            if pause_detected and len(self.current_turn_audio) > 0:
                logger.info("Pause detected, checking for end of turn...")
                return await self._check_end_of_turn()

            return None

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None

    def _process_vad(self, audio: np.ndarray) -> tuple[bool, bool]:
        """
        Process audio with VAD.
        
        Args:
            audio: Audio samples (float32)
            
        Returns:
            Tuple of (is_speech, pause_detected)
        """
        # Process full audio buffer
        results = self.vad.process_audio(audio)

        # Check if any frame detected a pause
        pause_detected = any(pause for _, pause in results)
        is_speech = any(speech for speech, _ in results)

        return is_speech, pause_detected

    async def _check_end_of_turn(self) -> Optional[str]:
        """
        Check if current audio represents end of turn.
        
        Returns:
            Final transcript if end-of-turn, None otherwise
        """
        try:
            # Get audio for Smart Turn check
            audio_buffer = np.concatenate(self.current_turn_audio)

            # Get last N seconds for Smart Turn
            check_duration = min(self.config.turn_check_seconds, len(audio_buffer) / self.config.sample_rate)
            check_samples = int(check_duration * self.config.sample_rate)
            audio_to_check = audio_buffer[-check_samples:] if len(audio_buffer) > check_samples else audio_buffer

            is_end_of_turn = False

            # Run Smart Turn if enabled and available
            if self.config.enable_smart_turn and self.smart_turn is not None:
                is_end_of_turn, confidence = self.smart_turn.is_end_of_turn(
                    audio_to_check
                )
                logger.info(
                    f"Smart Turn result: is_end={is_end_of_turn}, confidence={confidence:.3f}"
                )
            else:
                # Without Smart Turn, treat any pause as end of turn
                is_end_of_turn = True
                logger.info("Smart Turn disabled, treating pause as end of turn")

            if is_end_of_turn:
                return await self._finalize_turn(audio_buffer)
            else:
                # Not end of turn yet, continue listening
                return None

        except Exception as e:
            logger.error(f"Error checking end of turn: {e}")
            return None

    async def _finalize_turn(self, audio: np.ndarray) -> Optional[str]:
        """
        Finalize turn and get transcript.
        
        Args:
            audio: Full audio of the turn (float32)
            
        Returns:
            Final transcript
        """
        try:
            self.state = VoiceSessionState.PROCESSING

            # Run ASR if available
            if self.parakeet_asr is not None:
                # Convert to PCM16 for ASR
                audio_clipped = np.clip(audio, -1.0, 1.0)
                pcm16 = (audio_clipped * 32768.0).astype(np.int16).tobytes()

                # Transcribe
                transcript = self.parakeet_asr.transcribe_pcm16(
                    pcm16, sr=self.config.sample_rate
                )

                logger.info(f"Transcript: {transcript}")

                # Reset for next turn
                self.current_turn_audio = []
                self.state = VoiceSessionState.LISTENING
                self.turn_start_time = time.time()

                return transcript if transcript else None
            else:
                logger.warning("Parakeet ASR not available")
                self.current_turn_audio = []
                self.state = VoiceSessionState.LISTENING
                return None

        except Exception as e:
            logger.error(f"Error finalizing turn: {e}")
            self.current_turn_audio = []
            self.state = VoiceSessionState.LISTENING
            return None

    def get_state(self) -> VoiceSessionState:
        """Get current session state."""
        return self.state
