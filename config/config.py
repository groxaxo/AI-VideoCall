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
    fps: int = 16

    frame_width: int = 480
    frame_height: int = 640

    speaking_prompt: str = "A character is talking."
    silence_prompt: str = "A character is looking at the camera."


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
    def __init__(self):
        self.audio = AudioConfig()
        self.video = VideoConfig()
        self.server = ServerConfig()
        self.lip_sync = LipSyncConfig()
        self.voice = VoiceConfig()

        self._load_from_env()

    def _load_from_env(self):
        self.api_key = os.getenv("ZHIPUAI_API_KEY")
        self.log_level = os.getenv("LOG_LEVEL", "DEBUG")
        self.self_focing_config_path = os.getenv("CONFIG_PATH", "")
        self.audio_samples_per_video_block = round(
            self.audio.sample_rate
            / self.video.fps
            * self.lip_sync.dit_config.num_frame_per_block
            * 4
        )  # in audio samples, (4 for vae)
        self.lip_sync.audio_min_length = (
            4 * self.lip_sync.dit_config.num_frame_per_block
        )  # in frames, (4 for vae)

        # Voice configuration from environment
        self.voice.enabled = os.getenv("VOICE_ENABLED", "false").lower() == "true"
        self.voice.device_id = int(os.getenv("VOICE_GPU", "0"))
        self.voice.smart_turn_enabled = (
            os.getenv("SMART_TURN_ENABLED", "true").lower() == "true"
        )
        self.voice.smart_turn_onnx_path = os.getenv(
            "SMART_TURN_ONNX_PATH", self.voice.smart_turn_onnx_path
        )
        self.voice.parakeet_device = f"cuda:{self.voice.device_id}"

        # Video configuration from environment
        self.video.enabled = os.getenv("VIDEO_ENABLED", "true").lower() == "true"


config = Config()
