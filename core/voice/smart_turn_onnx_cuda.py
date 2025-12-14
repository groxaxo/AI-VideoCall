"""Smart Turn ONNX CUDA inference for end-of-turn detection."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class SmartTurnCudaConfig:
    """Configuration for Smart Turn CUDA inference."""

    onnx_path: str
    device_id: int = 0  # Default GPU 0 for voice processing
    threshold: float = 0.5
    sample_rate: int = 16000


class SmartTurnCuda:
    """
    Smart Turn end-of-turn detection using ONNX with CUDA acceleration.
    
    This implements the Smart Turn v3.x logic locally on GPU using ONNXRuntime
    with CUDA execution provider.
    """

    def __init__(self, cfg: SmartTurnCudaConfig):
        """
        Initialize Smart Turn CUDA inference.
        
        Args:
            cfg: SmartTurnCudaConfig configuration
        """
        self.cfg = cfg

        # Set up ONNX Runtime with CUDA provider
        providers = [
            ("CUDAExecutionProvider", {"device_id": cfg.device_id}),
            "CPUExecutionProvider",
        ]

        try:
            self.sess = ort.InferenceSession(cfg.onnx_path, providers=providers)
            self.in_name = self.sess.get_inputs()[0].name
            self.out_name = self.sess.get_outputs()[0].name

            # Get the actual provider used
            actual_providers = self.sess.get_providers()
            logger.info(f"Smart Turn initialized with providers: {actual_providers}")

            if "CUDAExecutionProvider" in actual_providers:
                logger.info(
                    f"Smart Turn using CUDA on device {cfg.device_id}"
                )
            else:
                logger.warning(
                    "Smart Turn: CUDA provider not available, falling back to CPU"
                )

            # Validate input/output shapes and types once on init
            self._validate_model_io()

        except Exception as e:
            logger.error(f"Failed to initialize Smart Turn ONNX model: {e}")
            raise

    def _validate_model_io(self):
        """Validate model input/output shapes and types."""
        try:
            input_info = self.sess.get_inputs()[0]
            output_info = self.sess.get_outputs()[0]
            
            logger.info(f"Smart Turn input: name={input_info.name}, shape={input_info.shape}, type={input_info.type}")
            logger.info(f"Smart Turn output: name={output_info.name}, shape={output_info.shape}, type={output_info.type}")
            
            # Store output shape for validation
            self.output_shape = output_info.shape
            
        except Exception as e:
            logger.warning(f"Could not validate Smart Turn model I/O: {e}")

    def is_end_of_turn(self, audio_f32_16k: np.ndarray) -> tuple[bool, float]:
        """
        Determine if the audio segment represents an end of turn.
        
        Args:
            audio_f32_16k: Audio samples as float32 at 16kHz
            
        Returns:
            Tuple of (is_end_of_turn, confidence_score)
        """
        try:
            # Reject audio that is too short (minimum 100ms)
            min_samples = int(0.1 * self.cfg.sample_rate)
            if len(audio_f32_16k) < min_samples:
                logger.debug(f"Audio too short ({len(audio_f32_16k)} samples), rejecting")
                return False, 0.0

            # Ensure audio is in correct format
            if audio_f32_16k.dtype != np.float32:
                audio_f32_16k = audio_f32_16k.astype(np.float32)

            # Add batch dimension if needed
            if audio_f32_16k.ndim == 1:
                x = audio_f32_16k[None, :]
            else:
                x = audio_f32_16k

            # Run inference
            y = self.sess.run([self.out_name], {self.in_name: x})[0]
            
            # Handle different output formats
            # If output has multiple dimensions, squeeze to get scalar
            if y.ndim > 0:
                y = np.squeeze(y)
            
            # Convert to probability
            # If output is logits (unbounded), apply sigmoid
            if y > 1.0 or y < 0.0:
                # Likely logits, apply sigmoid
                p = float(1.0 / (1.0 + np.exp(-y)))
            else:
                # Already a probability
                p = float(y)

            is_end = p >= self.cfg.threshold

            return is_end, p

        except Exception as e:
            logger.error(f"Error in Smart Turn inference: {e}")
            # In case of error, assume not end of turn
            return False, 0.0

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "sess"):
            del self.sess
