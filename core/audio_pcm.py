from __future__ import annotations

import base64

import numpy as np


def decode_pcm16_base64(value: str) -> np.ndarray:
    """Decode mono little-endian signed PCM16 base64 into float32 samples."""

    if not isinstance(value, str) or not value:
        raise ValueError("PCM16 payload must be a non-empty base64 string")
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("PCM16 payload is invalid base64") from exc
    if not raw:
        raise ValueError("PCM16 payload is empty")
    if len(raw) % 2:
        raise ValueError("PCM16 payload must contain an even number of bytes")

    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return np.ascontiguousarray(pcm / 32768.0)
