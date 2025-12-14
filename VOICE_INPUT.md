# Voice Input Documentation

This document describes the voice input feature implementation for local CUDA-based speech recognition and turn detection.

## Overview

The voice input system provides real-time speech-to-text transcription using:
- **Smart Turn v3.x** (ONNX CUDA): Local end-of-turn detection on GPU
- **Parakeet ASR** (NeMo): Local speech recognition on GPU
- **Energy-based VAD**: Simple voice activity detection for pause detection

## Architecture

### GPU Allocation (2x 3090 Setup)
- **GPU 0**: Interface server + Voice processing (VAD + Smart Turn + Parakeet ASR)
- **GPU 1**: DIT service

### Components

#### core/voice/ Package
- `ws_audio_protocol.py`: WebSocket message types for voice communication
- `audio_resample.py`: Audio format conversion utilities
- `ring_buffer.py`: Circular buffer for audio storage
- `vad_gate.py`: Energy-based voice activity detection
- `smart_turn_onnx_cuda.py`: Smart Turn ONNX GPU inference
- `parakeet_nemo_cuda.py`: Parakeet ASR with NeMo
- `voice_session.py`: Per-client state machine (VAD → pause → Smart Turn → ASR)

## Configuration

### Environment Variables

```bash
# Enable voice input
VOICE_ENABLED=true

# GPU configuration
VOICE_GPU=0  # GPU 0 for voice (with interface)

# Smart Turn settings
SMART_TURN_ENABLED=true
SMART_TURN_ONNX_PATH=models/smart_turn/smart-turn-v3.1-gpu.onnx

# Single GPU mode (optional)
WORLD_SIZE=1  # Runs everything on one GPU
```

### Config Options (config/config.py)

The `VoiceConfig` dataclass provides the following options:
- `enabled`: Enable/disable voice input (default: False)
- `device_id`: GPU device for voice processing (default: 0)
- `sample_rate`: Audio sample rate in Hz (default: 16000)
- `smart_turn_enabled`: Enable Smart Turn detection (default: True)
- `smart_turn_onnx_path`: Path to Smart Turn ONNX model
- `smart_turn_threshold`: Confidence threshold for end-of-turn (default: 0.5)
- `parakeet_model_name`: Parakeet model identifier
- `parakeet_device`: CUDA device for Parakeet
- `buffer_max_seconds`: Maximum audio buffer duration (default: 30.0)
- `turn_check_seconds`: Seconds of audio to check for turn (default: 2.0)

## Installation

### Required Dependencies

```bash
pip install onnxruntime-gpu nemo_toolkit[asr] soundfile
```

### Model Setup

1. **Smart Turn Model** (optional but recommended):
   - Download `smart-turn-v3.1-gpu.onnx` from official source
   - Place in `models/smart_turn/` directory

2. **Parakeet Model** (automatic):
   - Downloaded automatically by NeMo on first use
   - Cached in Hugging Face cache directory
   - Pre-download (optional):
     ```bash
     python -c "import nemo.collections.asr as nemo_asr; \
                nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')"
     ```

## WebSocket Protocol

### Client → Server Messages

#### Start Voice Recording
```json
{
  "type": "voice_start"
}
```

#### Binary Audio Chunks
- Send raw PCM16 audio data as binary WebSocket frames
- Format: 16kHz, mono, 16-bit PCM
- Recommended: 20ms frames (320 samples, 640 bytes)

#### Stop Voice Recording
```json
{
  "type": "voice_stop"
}
```

### Server → Client Messages

#### Voice Session Started
```json
{
  "type": "voice_started",
  "message": "Voice recording started",
  "timestamp": 1234567890.123
}
```

#### Voice Transcript
```json
{
  "type": "voice_transcript",
  "text": "transcribed text",
  "timestamp": 1234567890.123
}
```

#### Voice Session Stopped
```json
{
  "type": "voice_stopped",
  "message": "Voice recording stopped",
  "timestamp": 1234567890.123
}
```

#### Voice Error
```json
{
  "type": "voice_error",
  "error": "error message",
  "timestamp": 1234567890.123
}
```

## Usage Flow

1. **Client starts voice recording**:
   - Send `voice_start` JSON message
   - Server creates VoiceSession and responds with `voice_started`

2. **Client streams audio**:
   - Capture microphone audio at 16kHz, mono, PCM16
   - Send binary chunks (e.g., 20ms frames)
   - Server buffers audio and runs VAD

3. **Server detects end-of-turn**:
   - VAD detects pause in speech
   - Smart Turn confirms end-of-turn (if enabled)
   - Parakeet ASR transcribes accumulated audio
   - Server sends `voice_transcript` with text
   - Transcript is injected into normal text processing pipeline

4. **Client stops recording** (optional):
   - Send `voice_stop` JSON message
   - Server stops VoiceSession

## Frontend Integration

Example JavaScript code for client-side voice capture:

```javascript
// Start voice recording
websocket.send(JSON.stringify({ type: "voice_start" }));

// Capture microphone
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const audioContext = new AudioContext({ sampleRate: 16000 });
const source = audioContext.createMediaStreamSource(stream);
const processor = audioContext.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
  const audioData = e.inputBuffer.getChannelData(0);
  // Convert float32 to PCM16
  const pcm16 = new Int16Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) {
    pcm16[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
  }
  // Send as binary
  websocket.send(pcm16.buffer);
};

source.connect(processor);
processor.connect(audioContext.destination);

// Stop voice recording
websocket.send(JSON.stringify({ type: "voice_stop" }));
```

## Deployment Configurations

### 2x 3090 GPUs (Recommended)
```bash
WORLD_SIZE=2
VOICE_ENABLED=true
VOICE_GPU=0
```
- GPU 0: Interface + Voice
- GPU 1: DIT

### 1x 3090 GPU
```bash
WORLD_SIZE=1
VOICE_ENABLED=true
VOICE_GPU=0
```
- GPU 0: Everything (may require reduced video settings)

## Performance Considerations

- **Smart Turn**: ~5-10ms inference on GPU
- **Parakeet ASR**: ~100-500ms depending on audio length
- **VAD**: <1ms per frame (CPU)
- **Total latency**: Typically 200-800ms from speech end to transcript

## Troubleshooting

### Voice Not Working
1. Check `VOICE_ENABLED=true` in environment
2. Verify GPU availability: `nvidia-smi`
3. Check logs for initialization errors
4. Ensure models are downloaded/accessible

### CUDA Errors
- Ensure `onnxruntime-gpu` is installed (not just `onnxruntime`)
- Verify CUDA version compatibility with PyTorch
- Check GPU memory availability

### Poor Transcription Quality
- Verify audio format: 16kHz, mono, PCM16
- Check microphone quality
- Adjust VAD threshold in `VADConfig`
- Try different Smart Turn threshold values

## Future Enhancements

- Support for Silero VAD (more robust than energy-based)
- Streaming ASR for lower latency
- Multi-language support
- Speaker diarization
- Noise cancellation preprocessing
