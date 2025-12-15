# Audio-Only Mode Example

This example demonstrates how to run RealVideo in audio-only mode, which disables video generation and only processes audio.

## Benefits of Audio-Only Mode

- **Lower GPU memory usage**: Skips VAE video decoding
- **Faster processing**: No video frame generation overhead
- **Audio-only applications**: Perfect for voice assistants without visual requirements

## Usage

### Set the environment variable

```bash
export VIDEO_ENABLED=false
```

### Start the service

```bash
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/run_app.sh
```

### What to expect

1. The web interface will display "Audio-only mode (Video disabled)" instead of video frames
2. Audio responses will work normally
3. All text input and TTS functionality remains active
4. Video frame generation and VAE decoding will be skipped

## Re-enabling Video

To re-enable video generation:

```bash
export VIDEO_ENABLED=true
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/run_app.sh
```

Or simply omit the environment variable (video is enabled by default):

```bash
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/run_app.sh
```
