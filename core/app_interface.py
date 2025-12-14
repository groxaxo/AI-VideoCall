import asyncio
import json
import logging
import os

import uvicorn
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

import logging
import time
import traceback
import uuid
from typing import Optional

from config.config import config

from .connection import ConnectionManager
from .model_handler import ModelHandler
from .voice_clone import clone, get_voice_list, upload_audio_file


class RealVideoApp:
    def __init__(self):
        self.app = FastAPI(title="RealVideo")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.model_handler = ModelHandler()
        self.connection_manager = ConnectionManager()
        self.lip_sync_manager = self.model_handler.lip_sync_manager

        self.upload_folder = "uploads"
        self.allowed_image_exts = {"png", "jpg", "jpeg"}
        self.allowed_audio_exts = {"mp3", "wav"}
        self.max_file_size = 10 * 1024 * 1024
        self.last_ws_message_time = None
        self.ws_lifecheck_task = None

        # Voice processing components (initialized lazily if enabled)
        self.voice_enabled = config.voice.enabled
        self.smart_turn = None
        self.parakeet_asr = None
        self.voice_sessions = {}  # client_id -> VoiceSession

        if self.voice_enabled:
            self._init_voice_components()

        self._setup_routes()

        logger.info("Initialization finished.")

    def _init_voice_components(self):
        """Initialize voice processing components if voice is enabled."""
        try:
            from .voice import (
                ParakeetASR,
                ParakeetConfig,
                SmartTurnCuda,
                SmartTurnCudaConfig,
            )

            logger.info("Initializing voice components...")

            # Initialize Smart Turn if enabled
            if config.voice.smart_turn_enabled and os.path.exists(
                config.voice.smart_turn_onnx_path
            ):
                smart_turn_cfg = SmartTurnCudaConfig(
                    onnx_path=config.voice.smart_turn_onnx_path,
                    device_id=config.voice.device_id,
                    threshold=config.voice.smart_turn_threshold,
                    sample_rate=config.voice.sample_rate,
                )
                self.smart_turn = SmartTurnCuda(smart_turn_cfg)
                logger.info("Smart Turn initialized")
            else:
                logger.warning(
                    f"Smart Turn disabled or ONNX model not found at {config.voice.smart_turn_onnx_path}"
                )

            # Initialize Parakeet ASR
            parakeet_cfg = ParakeetConfig(
                model_name=config.voice.parakeet_model_name,
                device=config.voice.parakeet_device,
                use_amp=config.voice.parakeet_use_amp,
                sample_rate=config.voice.sample_rate,
            )
            self.parakeet_asr = ParakeetASR(parakeet_cfg)
            logger.info("Parakeet ASR initialized")

        except Exception as e:
            logger.error(f"Failed to initialize voice components: {e}")
            logger.warning("Voice input will be disabled")
            self.voice_enabled = False

    def allowed_file(self, filename, file_type="img"):
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.allowed_image_exts
            if file_type == "img"
            else self.allowed_audio_exts
        )

    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def get_homepage():
            try:
                with open("templates/index.html", "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            except Exception as e:
                logger.exception(f"Failed to load homepage: {e}")
                return HTMLResponse(content="<h1>Server error</h1>")

        @self.app.get("/api/status")
        async def get_system_status():
            return {
                "status": "running",
                "connections": self.connection_manager.get_connection_count(),
                "timestamp": time.time(),
            }

        @self.app.post("/upload_image")
        async def upload_image(image: UploadFile = File(...)):
            os.makedirs(self.upload_folder, exist_ok=True)
            if not self.allowed_file(image.filename):
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported filetype. Available filetypes: "
                    + ", ".join(self.allowed_image_exts),
                )

            contents = await image.read()
            if len(contents) > self.max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File is too large, maximum allowed is {self.max_file_size // (1024 * 1024)}MB",
                )

            file_extension = image.filename.split(".")[-1]
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = os.path.join(self.upload_folder, unique_filename)

            with open(file_path, "wb") as f:
                f.write(contents)

            return JSONResponse(
                {
                    "success": True,
                    "message": "Image uploaded",
                    "image_path": file_path,
                    "filename": unique_filename,
                }
            )

        @self.app.post("/upload_audio")
        async def upload_audio(
            audio: UploadFile = File(...),
        ):  # Upload wav and clone voice
            os.makedirs(self.upload_folder, exist_ok=True)
            if not self.allowed_file(audio.filename, file_type="audio"):
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported filetype. Available filetypes: "
                    + ", ".join(self.allowed_audio_exts),
                )

            contents = await audio.read()
            if len(contents) > self.max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File is too large, maximum allowed is {self.max_file_size // (1024 * 1024)}MB",
                )

            file_extension = audio.filename.split(".")[-1]
            voice_name = os.path.splitext(os.path.basename(audio.filename))[0]
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = os.path.join(self.upload_folder, unique_filename)

            with open(file_path, "wb") as f:
                f.write(contents)
            logger.info(f"{file_path} saved")

            try:
                file_id = upload_audio_file(file_path)
                logger.info(f"file uploaded: {file_id}")
                clone_ret = clone(file_id, voice_name)
                logger.info(f"voice clone finished")

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error: {e}")

            voice_list = get_voice_list()
            return JSONResponse(
                {
                    "success": True,
                    "message": "Voice clone succeeded.",
                    "voice_list": voice_list,
                }
            )

        @self.app.get("/get_voice_list", response_class=JSONResponse)
        async def return_voice_list():
            voice_list = get_voice_list()
            return JSONResponse(
                {
                    "success": True,
                    "message": "Voice list fetched.",
                    "voice_list": voice_list,
                }
            )

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: int):
            logger.info(f"Connecting to websocket client {client_id}")

            try:
                if self.lip_sync_manager.websocket is not None:
                    await websocket.close()
                    logger.info(
                        "Active websocket exists, rejecting new websocket connection."
                    )
                    return

                else:
                    await self.connection_manager.connect(websocket, client_id)

                await self.lip_sync_manager.connect_websocket(websocket)
                await self.model_handler.start_jobs(websocket)

                self.model_handler.tts_pipeline.reset_status()

                if self.ws_lifecheck_task is None:
                    self.ws_lifecheck_task = asyncio.create_task(self.ws_lifecheck())
                    logger.info("WebSocket: ws_lifecheck task created")

                await self._handle_websocket_connection(websocket, client_id)

            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")

            except Exception as e:
                logger.exception(f"Exception in Client {client_id}: {e}")
                logger.exception(traceback.format_exc())
                
            finally:
                # Clean up voice session if exists
                if client_id in self.voice_sessions:
                    try:
                        voice_session = self.voice_sessions[client_id]
                        # Finalize to transcribe any buffered audio
                        transcript = await voice_session.finalize()
                        if transcript:
                            logger.info(f"Final transcript on disconnect for client {client_id}: {transcript}")
                        voice_session.stop()
                        del self.voice_sessions[client_id]
                        logger.info(f"Voice session cleaned up for client {client_id}")
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up voice session for client {client_id}: {cleanup_error}")
                
                # Clean up websocket connections
                await self.lip_sync_manager.disconnect_websocket()
                self.connection_manager.disconnect(client_id)

    async def _handle_websocket_connection(self, websocket: WebSocket, client_id: int):
        while True:
            try:
                # Receive message - can be text or binary
                msg = await websocket.receive()
                self.last_ws_message_time = time.time()

                # Handle binary messages (voice PCM16 data)
                if msg["type"] == "websocket.receive" and "bytes" in msg:
                    if self.voice_enabled:
                        await self._handle_voice_binary(msg["bytes"], websocket, client_id)
                    else:
                        logger.warning("Received binary voice data but voice is disabled")
                    continue

                # Handle text messages (JSON)
                if "text" not in msg:
                    continue

                data = msg["text"]
                message_data = json.loads(data)
                logger.debug(message_data)

                logger.debug(
                    f"Received message from client {client_id}: {message_data.get('type', 'unknown')}"
                )

                # Handle voice control messages (voice_start, voice_stop)
                # Note: Voice protocol uses VoiceMessageType enum for responses
                if message_data["type"] == "voice_start":
                    await self._handle_voice_start(message_data, websocket, client_id)
                elif message_data["type"] == "voice_stop":
                    await self._handle_voice_stop(message_data, websocket, client_id)
                # Handle existing message types
                elif message_data["type"] in {"text", "audio"}:
                    await self._handle_text_audio_message(
                        message_data, websocket, client_id
                    )
                elif message_data["type"] == "ping":
                    await self._handle_ping_message(websocket, client_id)
                elif message_data["type"] in {"control", "image_config"}:
                    await self._handle_control_message(
                        message_data, websocket, client_id
                    )
                else:
                    logger.warning(f"Unknown message type: {message_data['type']}")

            except Exception as e:
                logger.error(
                    f"Failed to process message in _handle_websocket_connection: {e}, {type(e)}"
                )
                print(traceback.format_exc(), flush=True)
                await self.lip_sync_manager.disconnect_websocket()
                raise

    async def _handle_text_audio_message(
        self, message_data: dict, websocket: WebSocket, client_id: int
    ):
        if message_data["type"] == "text":
            profile_content = message_data.get("profile", "")
            text_content = message_data.get("text", "")
            audio_content = None
            sample_rate = None
            voice_id = message_data.get("voice_id", None)
            logger.info(
                f"Text message from client {client_id}: profile: {profile_content}, text: {text_content}"
            )

        elif message_data["type"] == "audio":
            profile_content = None
            text_content = None
            audio_content = message_data.get("audio", None)
            sample_rate = message_data.get("sample_rate", None)
            voice_id = None
            if audio_content is not None:
                logger.info(
                    f"Audio message from client {client_id}: {len(audio_content)}"
                )
            else:
                logger.info(f"Empty audio message from client {client_id}")

        processing_data = {
            "type": "processing_status",
            "status": "processing",
            "message": "Processing message...",
            "timestamp": message_data.get("timestamp", ""),
        }
        await websocket.send_text(json.dumps(processing_data))

        await self.model_handler.process_message(
            profile_content=profile_content,
            text_content=text_content,
            audio_base64=audio_content,
            sample_rate=sample_rate,
            voice_id=voice_id,
            websocket=websocket,
        )

    async def _handle_voice_start(
        self, message_data: dict, websocket: WebSocket, client_id: int
    ):
        """Handle voice_start control message."""
        if not self.voice_enabled:
            error_data = {
                "type": "voice_error",
                "error": "Voice input is not enabled",
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(error_data))
            return

        try:
            from .voice import VoiceSession, VoiceSessionConfig

            # Create voice session for this client
            session_cfg = VoiceSessionConfig(
                sample_rate=config.voice.sample_rate,
                buffer_max_seconds=config.voice.buffer_max_seconds,
                turn_check_seconds=config.voice.turn_check_seconds,
                enable_smart_turn=config.voice.smart_turn_enabled,
            )

            voice_session = VoiceSession(
                config=session_cfg,
                smart_turn=self.smart_turn,
                parakeet_asr=self.parakeet_asr,
            )
            voice_session.start()

            self.voice_sessions[client_id] = voice_session

            logger.info(f"Voice session started for client {client_id}")

            # Send acknowledgment
            response_data = {
                "type": "voice_started",
                "message": "Voice recording started",
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(response_data))

        except Exception as e:
            logger.error(f"Error starting voice session: {e}")
            error_data = {
                "type": "voice_error",
                "error": f"Failed to start voice session: {str(e)}",
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(error_data))

    async def _handle_voice_stop(
        self, message_data: dict, websocket: WebSocket, client_id: int
    ):
        """Handle voice_stop control message."""
        try:
            if client_id in self.voice_sessions:
                voice_session = self.voice_sessions[client_id]
                
                # Finalize session to transcribe any buffered audio
                transcript = await voice_session.finalize()
                
                # Stop and cleanup session
                voice_session.stop()
                del self.voice_sessions[client_id]
                
                logger.info(f"Voice session stopped for client {client_id}")

                # If we got a final transcript, process it
                if transcript:
                    logger.info(f"Final voice transcript for client {client_id}: {transcript}")
                    
                    # Inject transcript as a text message into the existing pipeline
                    text_message = {
                        "type": "text",
                        "text": transcript,
                        "profile": "",
                        "timestamp": time.time(),
                    }

                    # Send transcript notification to client
                    transcript_data = {
                        "type": "voice_transcript",
                        "text": transcript,
                        "timestamp": time.time(),
                    }
                    await websocket.send_text(json.dumps(transcript_data))

                    # Process as normal text message
                    await self._handle_text_audio_message(
                        text_message, websocket, client_id
                    )

                # Send acknowledgment
                response_data = {
                    "type": "voice_stopped",
                    "message": "Voice recording stopped",
                    "timestamp": time.time(),
                }
                await websocket.send_text(json.dumps(response_data))

        except Exception as e:
            logger.error(f"Error stopping voice session: {e}")

    async def _handle_voice_binary(
        self, pcm16_data: bytes, websocket: WebSocket, client_id: int
    ):
        """Handle binary voice data (PCM16 audio chunks)."""
        try:
            if client_id not in self.voice_sessions:
                logger.warning(
                    f"Received voice data for client {client_id} without active session"
                )
                return

            voice_session = self.voice_sessions[client_id]

            # Process audio chunk and check for transcript
            transcript = await voice_session.push_pcm16(pcm16_data)

            if transcript:
                logger.info(f"Voice transcript for client {client_id}: {transcript}")

                # Inject transcript as a text message into the existing pipeline
                text_message = {
                    "type": "text",
                    "text": transcript,
                    "profile": "",
                    "timestamp": time.time(),
                }

                # Send transcript notification to client
                transcript_data = {
                    "type": "voice_transcript",
                    "text": transcript,
                    "timestamp": time.time(),
                }
                await websocket.send_text(json.dumps(transcript_data))

                # Process as normal text message
                await self._handle_text_audio_message(
                    text_message, websocket, client_id
                )

        except Exception as e:
            logger.error(f"Error handling voice binary data: {e}")
            error_data = {
                "type": "voice_error",
                "error": f"Failed to process voice data: {str(e)}",
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(error_data))

    async def _handle_ping_message(self, websocket: WebSocket, client_id: int):
        pong_data = {"type": "pong", "timestamp": time.time(), "client_id": client_id}
        await websocket.send_text(json.dumps(pong_data))

    async def _handle_control_message(
        self, message_data, websocket: WebSocket, client_id: int
    ):
        try:
            logger.info(f"Control message from client {client_id}: {message_data}")
            await self.lip_sync_manager.process_control_message(message_data)

        except Exception as e:
            logger.warning(
                f"Failed to process control message in _handle_control_message: {e}, {type(e)}"
            )
            error_data = {
                "type": "error",
                "message": f"Failed to process control message, {e}",
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(error_data))
            raise

    async def ws_lifecheck(self):
        logger.info("entering ws lifecheck")
        while True:
            try:
                await asyncio.sleep(20)
                logger.info("checking websocket life")
                if (
                    self.last_ws_message_time is not None
                    and time.time() - self.last_ws_message_time > 60
                    and self.lip_sync_manager.websocket is not None
                ):
                    logger.info(
                        "Disconnecting websocket due to long time inactive %.3fs."
                        % (time.time() - self.last_ws_message_time)
                    )
                    await self.lip_sync_manager.disconnect_websocket()
                    self.lip_sync_manager.websocket = None

                elif (
                    self.lip_sync_manager.websocket is not None
                    and self.last_ws_message_time is not None
                ):
                    logger.info(
                        "Websocket lifecheck passed, %.3fs"
                        % (time.time() - self.last_ws_message_time)
                    )

                else:
                    logger.info("Websocket lifechecking, no active websocket")

            except Exception as e:
                logger.exception(f"Exception in ws_lifecheck: {e}")

    def run(self):
        logger.info(
            f"Starting server: http://{config.server.host}:{config.server.port}"
        )
        logger.info(config)

        uvicorn.run(
            self.app, host=config.server.host, port=config.server.port, log_level="info"
        )


def main():
    app = RealVideoApp()
    app.run()


if __name__ == "__main__":
    main()
