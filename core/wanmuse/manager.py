from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import uuid
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from config.config import config as service_config
from core.utils import encode_image_to_base64

from .frame_source import LatestFrameStore, ZmqLatestFrameSubscriber
from .musetalk_client import MuseTalkSidecarClient
from .settings import parse_face_bbox

logger = logging.getLogger(__name__)


class WanMuseLipSyncManager:
    """Compatibility manager for Wan2.2 TI2V-5B + MuseTalk.

    It implements the subset of ``LipSyncManager`` used by ``RealVideoApp`` and
    ``ModelHandler`` while removing all torch.distributed and S2V audio-encoder
    coupling. The latest TI2V frame is read from ZeroMQ, then a local MuseTalk
    sidecar generates audio-conditioned mouth frames.
    """

    def __init__(self, vae_idle_event: asyncio.Event):
        cfg = service_config.wanmuse
        self.fps = cfg.output_fps
        self.vae_idle_event = vae_idle_event
        self.websocket = None
        self.frame_count = 0
        self._paused = False
        self._render_lock = asyncio.Lock()
        self._fallback_frame: Optional[np.ndarray] = None

        self.frame_store = LatestFrameStore()
        self.frame_source = ZmqLatestFrameSubscriber(
            endpoint=cfg.frame_endpoint,
            store=self.frame_store,
            topic=cfg.frame_topic,
            poll_timeout_ms=cfg.frame_poll_timeout_ms,
        )
        self.musetalk = MuseTalkSidecarClient(
            cfg.musetalk_url,
            timeout_seconds=cfg.musetalk_timeout_seconds,
            token=cfg.musetalk_token,
            max_response_bytes=cfg.max_response_bytes,
            max_response_frames=cfg.max_response_frames,
            jpeg_quality=cfg.jpeg_quality,
        )
        self.face_bbox = parse_face_bbox(cfg.face_bbox)
        self.strict = cfg.strict
        self.max_frame_age_seconds = cfg.max_frame_age_seconds

        # ModelHandler uses this event as backpressure. The WanMuse path does
        # not own a local VAE, so it stays set except during one render segment.
        self.vae_idle_event.set()
        logger.info(
            "Initialized video backend ti2v5b_musetalk: "
            "frame_endpoint=%s, musetalk=%s",
            cfg.frame_endpoint,
            cfg.musetalk_url,
        )

    async def connect_websocket(self, websocket) -> None:
        self.websocket = websocket
        self.frame_source.start()
        self.vae_idle_event.set()
        logger.info("WanMuse websocket connected")

    async def disconnect_websocket(self) -> None:
        websocket = self.websocket
        self.websocket = None
        self.frame_source.stop()
        self.vae_idle_event.set()
        if websocket is not None:
            try:
                await websocket.close()
            except Exception:
                logger.debug("WanMuse websocket was already closed", exc_info=True)
        logger.info("WanMuse websocket disconnected")

    async def process_control_message(self, message: dict) -> None:
        message_type = message.get("type")
        if message_type == "control":
            command = message.get("text")
            if command == "stop decode":
                self._paused = True
            elif command == "do decode":
                self._paused = False
            return

        if message_type != "image_config":
            return
        image_path = str(message.get("image_path") or "").strip()
        if not image_path:
            raise ValueError("image_config requires image_path")
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"avatar image not found: {path}")
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"unable to decode avatar image: {path}")
        self._fallback_frame = frame
        self.frame_store.update(frame)
        self.frame_count = 0
        await self._send_frame(
            image_base64=encode_image_to_base64(frame),
            audio_base64="",
            audio_length=0.0,
        )

    async def process_audio_chunk(self, audio_base64, decoded_audio) -> list:
        if self._paused:
            return []
        if self.websocket is None:
            logger.debug(
                "WanMuse dropped segment because no websocket is connected"
            )
            return []

        async with self._render_lock:
            self.vae_idle_event.clear()
            try:
                frame = self._current_frame()
                if audio_base64 is None and decoded_audio is None:
                    await self._send_frame(
                        image_base64=(
                            encode_image_to_base64(frame)
                            if service_config.video.enabled
                            else ""
                        ),
                        audio_base64="",
                        audio_length=0.0,
                    )
                    return []

                audio_length = self._audio_length_seconds(decoded_audio)
                if not service_config.video.enabled:
                    await self._send_frame(
                        image_base64="",
                        audio_base64=audio_base64 or "",
                        audio_length=audio_length,
                    )
                    return []

                frames: Sequence[str]
                request_id = str(uuid.uuid4())
                try:
                    segment = await self.musetalk.render(
                        audio_wav_base64=audio_base64,
                        frame=frame,
                        fps=self.fps,
                        face_bbox=self.face_bbox,
                        request_id=request_id,
                    )
                    frames = segment.frames
                    if segment.fps != self.fps:
                        logger.warning(
                            "MuseTalk returned %d FPS while configured output "
                            "is %d FPS",
                            segment.fps,
                            self.fps,
                        )
                except Exception as exc:
                    if self.strict:
                        raise
                    logger.exception(
                        "MuseTalk render failed; sending audio with the latest "
                        "TI2V frame: %s",
                        exc,
                    )
                    frames = (encode_image_to_base64(frame),)

                for index, image_base64 in enumerate(frames):
                    await self._send_frame(
                        image_base64=image_base64,
                        audio_base64=(audio_base64 or "") if index == 0 else "",
                        audio_length=audio_length if index == 0 else 0.0,
                    )
                    await asyncio.sleep(0)
                return list(frames)
            finally:
                self.vae_idle_event.set()

    async def stop(self) -> None:
        self._paused = True
        self.vae_idle_event.set()

    def _current_frame(self) -> np.ndarray:
        snapshot = self.frame_store.snapshot(
            max_age_seconds=self.max_frame_age_seconds
        )
        if snapshot is not None:
            return snapshot.frame
        if self._fallback_frame is not None:
            return self._fallback_frame.copy()
        return np.zeros(
            (
                service_config.video.frame_height,
                service_config.video.frame_width,
                3,
            ),
            dtype=np.uint8,
        )

    @staticmethod
    def _audio_length_seconds(decoded_audio) -> float:
        if decoded_audio is None or not hasattr(decoded_audio, "shape"):
            return 0.0
        samples = int(decoded_audio.shape[-1]) if decoded_audio.shape else 0
        return max(0.0, samples / float(service_config.audio.sample_rate))

    async def _send_frame(
        self,
        *,
        image_base64: str,
        audio_base64: str,
        audio_length: float,
    ) -> None:
        if self.websocket is None:
            return
        frame_idx = self.frame_count
        payload = {
            "type": "audio_image",
            "audio": audio_base64,
            "image": image_base64,
            "timestamp": datetime.datetime.now().isoformat(),
            "frame_index": frame_idx,
            "audio_finish_frame": math.ceil(
                frame_idx + audio_length * self.fps
            ),
            "total_frames": frame_idx + 1,
            "fps": self.fps,
            "video_backend": "ti2v5b_musetalk",
        }
        await self.websocket.send_text(json.dumps(payload))
        self.frame_count += 1
