from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameSnapshot:
    """Immutable metadata plus an owned copy of the latest frame."""

    frame: np.ndarray
    received_at: float
    sequence: int


class LatestFrameStore:
    """Thread-safe latest-value frame store.

    Video generation is allowed to be bursty. Consumers always need the newest
    frame, not an unbounded FIFO of stale frames, so this store intentionally
    retains a single value.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._received_at = 0.0
        self._sequence = 0

    @staticmethod
    def _normalize(frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must have shape HxWx3, got {frame.shape}")
        if frame.size == 0:
            raise ValueError("frame must not be empty")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def update(self, frame: np.ndarray, *, received_at: Optional[float] = None) -> int:
        normalized = self._normalize(frame)
        timestamp = time.monotonic() if received_at is None else float(received_at)
        with self._lock:
            self._frame = normalized.copy()
            self._received_at = timestamp
            self._sequence += 1
            return self._sequence

    def snapshot(
        self, *, max_age_seconds: Optional[float] = None
    ) -> Optional[FrameSnapshot]:
        now = time.monotonic()
        with self._lock:
            if self._frame is None:
                return None
            if max_age_seconds is not None and max_age_seconds >= 0:
                if now - self._received_at > max_age_seconds:
                    return None
            return FrameSnapshot(
                frame=self._frame.copy(),
                received_at=self._received_at,
                sequence=self._sequence,
            )


class ZmqLatestFrameSubscriber:
    """Receive JPEG/PNG frames over ZeroMQ and keep only the newest one.

    Supported publisher payloads:
      * one-part message containing encoded image bytes
      * multipart message where the final part contains encoded image bytes

    A receive high-water mark of one, plus CONFLATE for the default single-part
    protocol, prevents stale frame accumulation under load.
    """

    def __init__(
        self,
        endpoint: str,
        store: LatestFrameStore,
        *,
        topic: str = "",
        poll_timeout_ms: int = 250,
    ) -> None:
        if not endpoint:
            raise ValueError("ZeroMQ frame endpoint must not be empty")
        self.endpoint = endpoint
        self.store = store
        self.topic = topic.encode("utf-8")
        self.poll_timeout_ms = max(10, int(poll_timeout_ms))
        self._stop_event = threading.Event()
        self._started = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._socket = None

    @property
    def started(self) -> bool:
        return self._started.is_set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="wanmuse-zmq-frame-subscriber",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._started.clear()

    def _run(self) -> None:
        try:
            import zmq
        except ImportError:
            logger.exception(
                "pyzmq is required for VIDEO_BACKEND=ti2v5b_musetalk; "
                "install the project requirements"
            )
            return

        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        self._socket = socket
        socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        socket.setsockopt(zmq.RCVHWM, 1)
        if not self.topic:
            try:
                socket.setsockopt(zmq.CONFLATE, 1)
            except zmq.ZMQError:
                logger.warning(
                    "ZeroMQ CONFLATE unavailable; RCVHWM=1 remains enabled"
                )
        else:
            logger.info("ZeroMQ topic mode uses RCVHWM=1 without CONFLATE")
        socket.connect(self.endpoint)

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self._started.set()
        logger.info("WanMuse frame subscriber connected to %s", self.endpoint)

        try:
            while not self._stop_event.is_set():
                events = dict(poller.poll(self.poll_timeout_ms))
                if socket not in events:
                    continue
                parts = socket.recv_multipart()
                if not parts:
                    continue
                payload = parts[-1]
                encoded = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning(
                        "Dropped an undecodable TI2V frame (%d bytes)", len(payload)
                    )
                    continue
                self.store.update(frame)
        except Exception:
            if not self._stop_event.is_set():
                logger.exception("WanMuse frame subscriber failed")
        finally:
            try:
                poller.unregister(socket)
            except Exception:
                pass
            socket.close(linger=0)
            self._socket = None
            self._started.clear()
