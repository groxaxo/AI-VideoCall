"""Ring buffer for storing recent audio data."""

import numpy as np
from typing import Optional


class RingBuffer:
    """
    Circular buffer for storing audio samples.
    Useful for keeping a sliding window of recent audio.
    """

    def __init__(self, max_seconds: float = 10.0, sample_rate: int = 16000):
        """
        Initialize ring buffer.
        
        Args:
            max_seconds: Maximum duration to store in seconds
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.size = 0

    def append(self, audio: np.ndarray):
        """
        Append audio samples to the ring buffer.
        
        Args:
            audio: Audio samples to append (float32 array)
        """
        n = len(audio)
        if n >= self.max_samples:
            # If new audio is longer than buffer, only keep the most recent part
            self.buffer[:] = audio[-self.max_samples :]
            self.write_pos = 0
            self.size = self.max_samples
        else:
            # Wrap around if necessary
            space_left = self.max_samples - self.write_pos
            if n <= space_left:
                self.buffer[self.write_pos : self.write_pos + n] = audio
                self.write_pos = (self.write_pos + n) % self.max_samples
            else:
                # Split write across wrap boundary
                self.buffer[self.write_pos :] = audio[:space_left]
                remaining = n - space_left
                self.buffer[:remaining] = audio[space_left:]
                self.write_pos = remaining

            self.size = min(self.size + n, self.max_samples)

    def get_recent(self, seconds: Optional[float] = None) -> np.ndarray:
        """
        Get the most recent audio samples.
        
        Args:
            seconds: Duration to retrieve in seconds. If None, returns all.
            
        Returns:
            Audio samples as float32 numpy array
        """
        if seconds is None:
            n = self.size
        else:
            n = min(int(seconds * self.sample_rate), self.size)

        if n == 0:
            return np.array([], dtype=np.float32)

        # Read from circular buffer
        start_pos = (self.write_pos - n) % self.max_samples
        if start_pos + n <= self.max_samples:
            # Contiguous read
            return self.buffer[start_pos : start_pos + n].copy()
        else:
            # Wrap-around read
            part1 = self.buffer[start_pos :]
            part2 = self.buffer[: n - len(part1)]
            return np.concatenate([part1, part2])

    def clear(self):
        """Clear the ring buffer."""
        self.buffer.fill(0)
        self.write_pos = 0
        self.size = 0

    def __len__(self) -> int:
        """Return number of samples currently in buffer."""
        return self.size
