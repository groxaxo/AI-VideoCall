from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


class MuseTalkClientError(RuntimeError):
    """Raised when the MuseTalk sidecar cannot produce a valid segment."""


@dataclass(frozen=True)
class RenderedSegment:
    request_id: str
    frames: tuple[str, ...]
    fps: int
    duration_seconds: float


def _validate_frames(frames: object, *, max_frames: int) -> tuple[str, ...]:
    if not isinstance(frames, list):
        raise MuseTalkClientError("MuseTalk response 'frames' must be a list")
    if not frames:
        raise MuseTalkClientError("MuseTalk response contained no frames")
    if len(frames) > max_frames:
        raise MuseTalkClientError(
            f"MuseTalk response exceeded frame limit: {len(frames)} > {max_frames}"
        )

    validated: list[str] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, str) or not frame:
            raise MuseTalkClientError(
                f"MuseTalk frame {index} is not non-empty base64"
            )
        try:
            base64.b64decode(frame, validate=True)
        except Exception as exc:
            raise MuseTalkClientError(
                f"MuseTalk frame {index} is invalid base64"
            ) from exc
        validated.append(frame)
    return tuple(validated)


def parse_render_response(
    payload: object, *, max_frames: int = 600
) -> RenderedSegment:
    if not isinstance(payload, dict):
        raise MuseTalkClientError("MuseTalk response must be a JSON object")

    frames = _validate_frames(payload.get("frames"), max_frames=max_frames)
    try:
        fps = int(payload.get("fps", 25))
    except (TypeError, ValueError) as exc:
        raise MuseTalkClientError("MuseTalk response fps must be an integer") from exc
    if fps < 1 or fps > 60:
        raise MuseTalkClientError(f"MuseTalk response fps out of range: {fps}")

    try:
        duration = float(payload.get("duration_seconds", len(frames) / fps))
    except (TypeError, ValueError) as exc:
        raise MuseTalkClientError(
            "MuseTalk duration_seconds must be numeric"
        ) from exc
    if duration < 0:
        raise MuseTalkClientError("MuseTalk duration_seconds must be non-negative")

    request_id = str(payload.get("request_id") or "")
    return RenderedSegment(
        request_id=request_id,
        frames=frames,
        fps=fps,
        duration_seconds=duration,
    )


class MuseTalkSidecarClient:
    """Small dependency-free HTTP client for the local MuseTalk sidecar."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 60.0,
        token: str = "",
        max_response_bytes: int = 96 * 1024 * 1024,
        max_response_frames: int = 600,
        jpeg_quality: int = 90,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("MuseTalk base URL must not be empty")
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.token = token
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.max_response_frames = max(1, int(max_response_frames))
        self.jpeg_quality = min(100, max(40, int(jpeg_quality)))

    async def render(
        self,
        *,
        audio_wav_base64: str,
        frame: np.ndarray,
        fps: int,
        face_bbox: Optional[Sequence[int]] = None,
        request_id: str = "",
    ) -> RenderedSegment:
        if not audio_wav_base64:
            raise MuseTalkClientError("audio_wav_base64 must not be empty")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise MuseTalkClientError(
                f"frame must have shape HxWx3, got {frame.shape}"
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(frame),
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise MuseTalkClientError("failed to JPEG-encode the TI2V frame")

        payload = {
            "request_id": request_id,
            "audio_wav_base64": audio_wav_base64,
            "frame_jpeg_base64": base64.b64encode(encoded.tobytes()).decode(
                "ascii"
            ),
            "fps": int(fps),
            "jpeg_quality": self.jpeg_quality,
            "face_bbox": list(face_bbox) if face_bbox is not None else None,
        }
        response = await asyncio.to_thread(self._post_json, "/v1/render", payload)
        return parse_render_response(
            response, max_frames=self.max_response_frames
        )

    def _post_json(self, path: str, payload: dict) -> object:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                raw = response.read(self.max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise MuseTalkClientError(
                f"MuseTalk sidecar returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise MuseTalkClientError(
                f"MuseTalk sidecar request failed: {exc}"
            ) from exc

        if len(raw) > self.max_response_bytes:
            raise MuseTalkClientError(
                "MuseTalk sidecar response exceeded byte limit"
            )
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MuseTalkClientError(
                "MuseTalk sidecar returned invalid JSON"
            ) from exc
