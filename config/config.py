import os
from dataclasses import dataclass

from omegaconf import OmegaConf

PATH_TO_YOUR_MODEL = "zai-org/RealVideo/model.pt"  # Replace with your model path


@dataclass
class AudioConfig:
    sample_rate: int = 16000


@dataclass
class VoiceConfig:
    """Configuration for voice input processing."""

    enabled: bool = False  # Voice input disabled by default
    device_id: int = 0  # GPU 0 for voice (interface GPU)
    sample_rate: int = 16000

    # Smart Turn configuration
    smart_turn_enabled: bool = True
    smart_turn_onnx_path: str = "models/smart_turn/smart-turn-v3.1-gpu.onnx"
    smart_turn_threshold: float = 0.5

    # Parakeet ASR configuration
    parakeet_model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    parakeet_device: str = "cuda:0"
    parakeet_use_amp: bool = True

    # Voice session configuration
    buffer_max_seconds: float = 30.0
    turn_check_seconds: float = 8.0  # Increased from 2.0 for better Smart Turn context


@dataclass
class VideoConfig:
    enabled: bool = True  # Video generation enabled by default
    backend: str = "self_forcing_s2v"
    fps: int = 16

    frame_width: int = 480
    frame_height: int = 640

    speaking_prompt: str = "A character is talking."
    silence_prompt: str = "A character is looking at the camera."


@dataclass
class WanMuseConfig:
    """Wan2.2 TI2V-5B + MuseTalk sidecar configuration."""

    frame_endpoint: str = "tcp://127.0.0.1:5560"
    frame_topic: str = ""
    frame_poll_timeout_ms: int = 250
    max_frame_age_seconds: float = 10.0

    musetalk_url: str = "http://127.0.0.1:8011"
    musetalk_timeout_seconds: float = 60.0
    musetalk_token: str = ""
    output_fps: int = 25
    face_bbox: str = ""
    audio_segment_seconds: float = 1.0

    jpeg_quality: int = 90
    max_response_bytes: int = 96 * 1024 * 1024
    max_response_frames: int = 600
    strict: bool = False


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8003
    diffusion_socket_port: int = 9090
    app_socket_port: int = 9091
    app_ready_socket_port: int = 9092
    diffusion_ready_socket_port: int = 9093


@dataclass
class LipSyncConfig:
    fps: int = 16

    s2v_segment_latent_length = 80

    self_forcing_config_path: str = (
        "self_forcing/configs/sample_14B_s2v_sparse_nfb2.yaml"
    )
    # self_forcing_config_path: str = 'self_forcing/configs/sample_14B_s2v_sparse_nfb2_2steps.yaml'

    checkpoint_path: str = PATH_TO_YOUR_MODEL

    audio_padding_div = 16
    audio_padding_rem = 0
    audio_min_length = 16
    audio_segment_length = 80
    s2v_video_refresh_interval = 20
    compile = True
    profile = True
    fp8_quantize = False
    no_refresh_inference = True

    dit_config = OmegaConf.load(self_forcing_config_path)
    default_config = OmegaConf.load("self_forcing/configs/default_config.yaml")
    dit_config = OmegaConf.merge(default_config, dit_config)


class Config:
    VALID_VIDEO_BACKENDS = {"self_forcing_s2v", "ti2v5b_musetalk"}

    def __init__(self):
        self.audio = AudioConfig()
        self.video = VideoConfig()
        self.wanmuse = WanMuseConfig()
        self.server = ServerConfig()
        self.lip_sync = LipSyncConfig()
        self.voice = VoiceConfig()

        self._load_from_env()

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        return os.getenv(name, str(default)).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _load_from_env(self):
        self.api_key = os.getenv("ZHIPUAI_API_KEY")
        self.log_level = os.getenv("LOG_LEVEL", "DEBUG")
        self.self_focing_config_path = os.getenv("CONFIG_PATH", "")

        # Video backend selection must happen before audio chunk sizing because
        # the TI2V+MuseTalk path is not tied to DiT frame blocks.
        backend = os.getenv("VIDEO_BACKEND", self.video.backend).strip().lower()
        if backend not in self.VALID_VIDEO_BACKENDS:
            raise ValueError(
                f"Unsupported VIDEO_BACKEND={backend!r}; choose from "
                f"{sorted(self.VALID_VIDEO_BACKENDS)}"
            )
        self.video.backend = backend

        # Voice configuration from environment
        self.voice.enabled = self._env_bool("VOICE_ENABLED", False)
        self.voice.device_id = int(os.getenv("VOICE_GPU", "0"))
        self.voice.smart_turn_enabled = self._env_bool("SMART_TURN_ENABLED", True)
        self.voice.smart_turn_onnx_path = os.getenv(
            "SMART_TURN_ONNX_PATH", self.voice.smart_turn_onnx_path
        )
        self.voice.parakeet_device = f"cuda:{self.voice.device_id}"

        # Video configuration from environment
        self.video.enabled = self._env_bool("VIDEO_ENABLED", True)
        self.video.frame_width = int(
            os.getenv("VIDEO_FRAME_WIDTH", str(self.video.frame_width))
        )
        self.video.frame_height = int(
            os.getenv("VIDEO_FRAME_HEIGHT", str(self.video.frame_height))
        )

        # WanMuse sidecar and transport settings
        self.wanmuse.frame_endpoint = os.getenv(
            "WAN_FRAME_ENDPOINT", self.wanmuse.frame_endpoint
        )
        self.wanmuse.frame_topic = os.getenv("WAN_FRAME_TOPIC", self.wanmuse.frame_topic)
        self.wanmuse.frame_poll_timeout_ms = int(
            os.getenv(
                "WAN_FRAME_POLL_TIMEOUT_MS",
                str(self.wanmuse.frame_poll_timeout_ms),
            )
        )
        self.wanmuse.max_frame_age_seconds = float(
            os.getenv(
                "WAN_FRAME_MAX_AGE_SECONDS",
                str(self.wanmuse.max_frame_age_seconds),
            )
        )
        self.wanmuse.musetalk_url = os.getenv(
            "MUSETALK_URL", self.wanmuse.musetalk_url
        )
        self.wanmuse.musetalk_timeout_seconds = float(
            os.getenv(
                "MUSETALK_TIMEOUT_SECONDS",
                str(self.wanmuse.musetalk_timeout_seconds),
            )
        )
        self.wanmuse.musetalk_token = os.getenv("MUSE_TALK_TOKEN", "")
        self.wanmuse.output_fps = int(
            os.getenv("MUSETALK_OUTPUT_FPS", str(self.wanmuse.output_fps))
        )
        if not 1 <= self.wanmuse.output_fps <= 60:
            raise ValueError("MUSETALK_OUTPUT_FPS must be between 1 and 60")
        self.wanmuse.face_bbox = os.getenv("WANMUSE_FACE_BBOX", "")
        self.wanmuse.audio_segment_seconds = float(
            os.getenv(
                "WANMUSE_AUDIO_SEGMENT_SECONDS",
                str(self.wanmuse.audio_segment_seconds),
            )
        )
        if self.wanmuse.audio_segment_seconds <= 0:
            raise ValueError("WANMUSE_AUDIO_SEGMENT_SECONDS must be positive")
        self.wanmuse.jpeg_quality = int(
            os.getenv("WANMUSE_JPEG_QUALITY", str(self.wanmuse.jpeg_quality))
        )
        self.wanmuse.strict = self._env_bool("WANMUSE_STRICT", False)

        if self.video.backend == "ti2v5b_musetalk":
            self.video.fps = self.wanmuse.output_fps
            self.lip_sync.fps = self.wanmuse.output_fps
            desired_samples = max(
                1,
                round(self.audio.sample_rate * self.wanmuse.audio_segment_seconds),
            )
            # ModelHandler historically stores segment length in thousands of
            # samples. Preserve that contract while using a time-based setting.
            self.lip_sync.audio_segment_length = max(1, round(desired_samples / 1000))
            self.audio_samples_per_video_block = desired_samples
            self.lip_sync.audio_min_length = max(
                1, round(self.wanmuse.audio_segment_seconds * self.video.fps)
            )
        else:
            self.video.fps = int(os.getenv("VIDEO_FPS", str(self.video.fps)))
            self.lip_sync.fps = self.video.fps
            self.audio_samples_per_video_block = round(
                self.audio.sample_rate
                / self.video.fps
                * self.lip_sync.dit_config.num_frame_per_block
                * 4
            )  # in audio samples, (4 for vae)
            self.lip_sync.audio_min_length = (
                4 * self.lip_sync.dit_config.num_frame_per_block
            )  # in frames, (4 for vae)


config = Config()
