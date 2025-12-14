"""WebSocket audio protocol message types and helpers."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class VoiceMessageType(str, Enum):
    """Voice message types for WebSocket communication."""

    VOICE_START = "voice_start"
    VOICE_STOP = "voice_stop"
    VOICE_CHUNK = "voice_chunk"  # Binary PCM16 data
    VOICE_TRANSCRIPT = "voice_transcript"
    VOICE_ERROR = "voice_error"


@dataclass
class VoiceMessage:
    """Voice message structure."""

    type: VoiceMessageType
    data: Optional[bytes] = None
    text: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "text": self.text,
            "error": self.error,
            "timestamp": self.timestamp,
        }
