from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import os
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("musetalk_sidecar")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


class RenderRequest(BaseModel):
    request_id: str = ""
    audio_wav_base64: str = Field(min_length=4)
    frame_jpeg_base64: str = Field(min_length=4)
    fps: int = Field(default=25, ge=1, le=60)
    jpeg_quality: int = Field(default=90, ge=40, le=100)
    face_bbox: Optional[list[int]] = None

    @field_validator("face_bbox")
    @classmethod
    def validate_face_bbox(cls, value):
        if value is None:
            return value
        if len(value) != 4:
            raise ValueError("face_bbox must be [x1,y1,x2,y2]")
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("face_bbox must have x2>x1 and y2>y1")
        return value


class RenderResponse(BaseModel):
    request_id: str
    fps: int
    duration_seconds: float
    frames: list[str]


@dataclass
class EngineConfig:
    livetalking_root: Path
    whisper_path: str
    device: str
    batch_size: int
    max_audio_seconds: float
    default_face_bbox: Optional[tuple[int, int, int, int]]
    token: str

    @classmethod
    def from_env(cls) -> "EngineConfig":
        bbox_raw = os.getenv("MUSE_TALK_FACE_BBOX", "").strip()
        bbox = None
        if bbox_raw:
            values = tuple(int(part.strip()) for part in bbox_raw.split(","))
            if (
                len(values) != 4
                or values[2] <= values[0]
                or values[3] <= values[1]
            ):
                raise ValueError("MUSE_TALK_FACE_BBOX must be x1,y1,x2,y2")
            bbox = values
        return cls(
            livetalking_root=Path(
                os.getenv(
                    "MUSE_TALK_LIVETALKING_ROOT", "/home/op/LiveTalking"
                )
            ).expanduser(),
            whisper_path=os.getenv(
                "MUSE_TALK_WHISPER_PATH", "./models/whisper"
            ),
            device=os.getenv("MUSE_TALK_DEVICE", "cuda"),
            batch_size=max(
                1, int(os.getenv("MUSE_TALK_BATCH_SIZE", "8"))
            ),
            max_audio_seconds=max(
                0.25,
                float(os.getenv("MUSE_TALK_MAX_AUDIO_SECONDS", "15")),
            ),
            default_face_bbox=bbox,
            token=os.getenv("MUSE_TALK_TOKEN", ""),
        )


class MuseTalkEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.ready = False
        self.error = ""
        self._lock = asyncio.Lock()
        self.vae = None
        self.unet = None
        self.pe = None
        self.audio_processor = None
        self.timesteps = None
        self.device = torch.device(config.device)

    def load(self) -> None:
        root = self.config.livetalking_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"LiveTalking root does not exist: {root}"
            )

        # LiveTalking resolves MuseTalk checkpoints relative to its repository.
        os.chdir(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from avatars.musetalk.utils.utils import load_all_model
        from avatars.musetalk.whisper.audio2feature import Audio2Feature

        logger.info("Loading MuseTalk from %s on %s", root, self.device)
        self.vae, self.unet, self.pe = load_all_model(device=self.device)
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        self.audio_processor = Audio2Feature(
            model_path=self.config.whisper_path
        )
        self.timesteps = torch.tensor([0], device=self.device)
        self.ready = True
        self.error = ""
        logger.info("MuseTalk sidecar ready")

    async def render(self, request: RenderRequest) -> RenderResponse:
        if not self.ready:
            raise RuntimeError(self.error or "MuseTalk engine is not ready")
        async with self._lock:
            return await asyncio.to_thread(self._render_sync, request)

    def _render_sync(self, request: RenderRequest) -> RenderResponse:
        wav = self._decode_audio(request.audio_wav_base64)
        duration = wav.shape[0] / 16000.0
        if duration > self.config.max_audio_seconds:
            raise ValueError(
                f"audio segment is {duration:.2f}s; maximum is "
                f"{self.config.max_audio_seconds:.2f}s"
            )
        frame = self._decode_frame(request.frame_jpeg_base64)
        bbox = self._resolve_bbox(frame, request.face_bbox)
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        crop_256 = cv2.resize(
            crop, (256, 256), interpolation=cv2.INTER_LANCZOS4
        )

        base_latent = self.vae.get_latents_for_unet(crop_256)
        features = self.audio_processor.audio2feat(wav)
        frame_count = max(1, int(math.ceil(duration * request.fps)))
        chunks = self.audio_processor.feature2chunks(
            features,
            fps=request.fps,
            batch_size=frame_count,
        )
        if not chunks:
            chunks = [np.zeros((50, 384), dtype=np.float32)]

        output_frames: list[str] = []
        for start in range(0, len(chunks), self.config.batch_size):
            batch_chunks = chunks[start : start + self.config.batch_size]
            whisper_batch = torch.from_numpy(np.stack(batch_chunks)).to(
                device=self.device,
                dtype=self.unet.model.dtype,
            )
            audio_features = self.pe(whisper_batch)
            latent_batch = torch.cat(
                [base_latent] * len(batch_chunks), dim=0
            ).to(
                device=self.device,
                dtype=self.unet.model.dtype,
            )
            with torch.inference_mode():
                predicted_latents = self.unet.model(
                    latent_batch,
                    self.timesteps,
                    encoder_hidden_states=audio_features,
                ).sample
                predicted_faces = self.vae.decode_latents(
                    predicted_latents
                )

            for predicted_face in predicted_faces:
                composited = self._blend_face(
                    frame, predicted_face, bbox
                )
                ok, encoded = cv2.imencode(
                    ".jpg",
                    composited,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        request.jpeg_quality,
                    ],
                )
                if not ok:
                    raise RuntimeError(
                        "failed to encode MuseTalk output frame"
                    )
                output_frames.append(
                    base64.b64encode(encoded.tobytes()).decode("ascii")
                )

        return RenderResponse(
            request_id=request.request_id or str(uuid.uuid4()),
            fps=request.fps,
            duration_seconds=duration,
            frames=output_frames,
        )

    @staticmethod
    def _decode_frame(value: str) -> np.ndarray:
        try:
            raw = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ValueError(
                "frame_jpeg_base64 is invalid base64"
            ) from exc
        encoded = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(
                "frame_jpeg_base64 is not a decodable image"
            )
        return frame

    @staticmethod
    def _decode_audio(value: str) -> np.ndarray:
        try:
            raw = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ValueError(
                "audio_wav_base64 is invalid base64"
            ) from exc
        try:
            waveform, sample_rate = sf.read(
                io.BytesIO(raw),
                dtype="float32",
                always_2d=False,
            )
        except Exception as exc:
            raise ValueError(
                "audio_wav_base64 is not a readable WAV"
            ) from exc
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if waveform.size == 0:
            raise ValueError("audio WAV is empty")
        if sample_rate != 16000:
            tensor = torch.from_numpy(waveform).unsqueeze(0)
            tensor = torchaudio.functional.resample(
                tensor, sample_rate, 16000
            )
            waveform = tensor.squeeze(0).numpy()
        return np.ascontiguousarray(waveform, dtype=np.float32)

    def _resolve_bbox(
        self,
        frame: np.ndarray,
        request_bbox: Optional[list[int]],
    ) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        bbox = (
            tuple(request_bbox)
            if request_bbox is not None
            else self.config.default_face_bbox
        )
        if bbox is None:
            # Safe center portrait heuristic. Production deployments should
            # set an explicit face ROI to eliminate detector jitter.
            bbox = (
                int(width * 0.25),
                int(height * 0.08),
                int(width * 0.75),
                int(height * 0.68),
            )
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(x1 + 1, min(int(x2), width))
        y2 = max(y1 + 1, min(int(y2), height))
        return x1, y1, x2, y2

    @staticmethod
    def _blend_face(
        frame: np.ndarray,
        predicted_face: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        output = frame.copy()
        generated = cv2.resize(
            predicted_face.astype(np.uint8),
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_LANCZOS4,
        ).astype(np.float32)
        original = output[y1:y2, x1:x2].astype(np.float32)

        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        mask[int(mask.shape[0] * 0.38) :, :] = 1.0
        kernel = max(9, (min(mask.shape[:2]) // 10) | 1)
        mask = cv2.GaussianBlur(
            mask, (kernel, kernel), 0
        )[..., None]
        blended = generated * mask + original * (1.0 - mask)
        output[y1:y2, x1:x2] = np.clip(
            blended, 0, 255
        ).astype(np.uint8)
        return output


engine_config = EngineConfig.from_env()
engine = MuseTalkEngine(engine_config)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await asyncio.to_thread(engine.load)
    except Exception as exc:
        engine.error = str(exc)
        logger.exception("MuseTalk sidecar failed to initialize")
    yield


app = FastAPI(
    title="AI VideoCall MuseTalk Sidecar", lifespan=lifespan
)


def _authorize(authorization: Optional[str]) -> None:
    expected = engine_config.token
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid bearer token")


@app.get("/healthz")
async def healthz():
    if not engine.ready:
        raise HTTPException(
            status_code=503,
            detail=engine.error or "not ready",
        )
    return {"status": "ok", "device": str(engine.device)}


@app.post("/v1/render", response_model=RenderResponse)
async def render(
    request: RenderRequest,
    authorization: Optional[str] = Header(default=None),
):
    _authorize(authorization)
    try:
        return await engine.render(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("MuseTalk render failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    uvicorn.run(
        "services.musetalk_sidecar:app",
        host=os.getenv("MUSE_TALK_HOST", "127.0.0.1"),
        port=int(os.getenv("MUSE_TALK_PORT", "8011")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
