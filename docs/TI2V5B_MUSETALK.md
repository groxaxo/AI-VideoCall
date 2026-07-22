# Wan2.2 TI2V-5B + MuseTalk real-time backend

This backend decouples conversational audio from generative-video latency:

```text
Wan2.2 TI2V-5B / ABot-World (GPU 0)
        │ newest JPEG frame over ZeroMQ
        ▼
AI VideoCall interface (GPU 2 or CPU/cloud LLM)
        │ latest frame + each streamed WAV segment
        ▼
MuseTalk sidecar (GPU 1)
        │ audio-conditioned mouth frames
        ▼
existing AI VideoCall WebSocket protocol
```

The original `self_forcing_s2v` backend remains the default. Select the new
backend with `VIDEO_BACKEND=ti2v5b_musetalk`.

## Why MuseTalk remains in the pipeline

Wan2.2 TI2V-5B and ABot-World generate motion and scene continuity but do not
accept a phoneme timeline. MuseTalk consumes Whisper audio features and therefore
remains responsible for the synchronized mouth region. The two models run on
separate GPUs so a slow TI2V frame never stalls audio playback.

## Recommended RTX 3090 allocation

| GPU | Service |
|---|---|
| GPU 0 | ABot-World / Wan2.2 TI2V-5B at 832x480 |
| GPU 1 | MuseTalk sidecar |
| GPU 2 | AI VideoCall interface, Parakeet and/or local TTS |

For ABot-World on Ampere, start in BF16. If runtime quantization is needed, use
`fp8-per-block`; do not use its Blackwell-oriented `fp8-per-token` default on a
3090.

## 1. Install the additional dependencies

```bash
pip install 'pyzmq>=26,<28'
```

The normal project requirements include this package and a matching
`torchaudio` build for the pinned PyTorch version.

## 2. Prepare LiveTalking and MuseTalk

Clone and install LiveTalking in a separate environment. Download its MuseTalk
v1.5, SD VAE, positional encoder and Whisper model files according to its own
instructions.

The sidecar intentionally imports LiveTalking rather than copying its model
implementation into this repository. Run the sidecar with the LiveTalking
Python environment while keeping the working directory at AI VideoCall:

```bash
cd /home/op/AI-VideoCall

CUDA_VISIBLE_DEVICES=1 \
MUSE_TALK_LIVETALKING_ROOT=/home/op/LiveTalking \
MUSE_TALK_DEVICE=cuda \
MUSE_TALK_PORT=8011 \
MUSE_TALK_BATCH_SIZE=8 \
MUSE_TALK_FACE_BBOX='220,35,610,440' \
/home/op/LiveTalking/.venv/bin/python -m services.musetalk_sidecar
```

Health check:

```bash
curl --fail http://127.0.0.1:8011/healthz
```

`MUSE_TALK_FACE_BBOX` is strongly recommended. It is the face region in the
832x480 TI2V output, expressed as `x1,y1,x2,y2`. Without it, the sidecar uses a
central portrait heuristic.

To protect a non-loopback sidecar, set the same token in both processes:

```bash
export MUSE_TALK_TOKEN='replace-with-a-random-secret'
```

## 3. Publish TI2V frames

The subscriber accepts either a one-part message containing JPEG/PNG bytes or a
multipart message whose final part contains the encoded image. Single-part mode
uses ZeroMQ `CONFLATE`; all modes use `RCVHWM=1`, so stale generated frames are
discarded instead of accumulating.

A publisher can be inserted into ABot-World's decoded-frame loop:

```python
from services.ti2v_frame_publisher import LatestFramePublisher

publisher = LatestFramePublisher('tcp://127.0.0.1:5560')

# For each decoded BGR numpy frame:
publisher.publish(frame)
```

For a smoke test, publish an existing video in real time:

```bash
python -m services.ti2v_frame_publisher \
  /path/to/wan_avatar_loop.mp4 \
  --endpoint tcp://127.0.0.1:5560
```

The publisher must bind; AI VideoCall connects as the subscriber.

## 4. Start AI VideoCall

Run the interface on the third GPU. The new backend is single-process and does
not initialize torch.distributed or the 14B Self-Forcing DiT service. The
built-in browser currently schedules frames at 16 FPS, so keep
`MUSETALK_OUTPUT_FPS=16` for synchronized playback.

```bash
cd /home/op/AI-VideoCall

CUDA_VISIBLE_DEVICES=2 \
VIDEO_BACKEND=ti2v5b_musetalk \
VIDEO_ENABLED=true \
VOICE_ENABLED=true \
VOICE_GPU=0 \
WAN_FRAME_ENDPOINT=tcp://127.0.0.1:5560 \
MUSETALK_URL=http://127.0.0.1:8011 \
MUSETALK_OUTPUT_FPS=16 \
VIDEO_FRAME_WIDTH=832 \
VIDEO_FRAME_HEIGHT=480 \
WANMUSE_AUDIO_SEGMENT_SECONDS=1.0 \
WANMUSE_FACE_BBOX='220,35,610,440' \
bash scripts/run_app.sh
```

Because `CUDA_VISIBLE_DEVICES=2` exposes only one physical card, `VOICE_GPU=0`
correctly addresses that visible device.

Open `http://localhost:8003`, upload a stable portrait reference, connect the
WebSocket, and start a conversation.

A convenience launcher starts the MuseTalk sidecar and AI VideoCall while
leaving GPU 0 available for your Wan/ABot publisher:

```bash
MUSE_TALK_LIVETALKING_ROOT=/home/op/LiveTalking \
MUSE_TALK_PYTHON=/home/op/LiveTalking/.venv/bin/python \
WAN_GPU=0 MUSETALK_GPU=1 APP_GPU=2 \
bash scripts/run_wanmuse_stack.sh
```

## Environment variables

| Variable | Default | Purpose |
|---|---:|---|
| `VIDEO_BACKEND` | `self_forcing_s2v` | Set `ti2v5b_musetalk` for this backend |
| `WAN_FRAME_ENDPOINT` | `tcp://127.0.0.1:5560` | ZeroMQ TI2V frame publisher |
| `WAN_FRAME_TOPIC` | empty | Optional SUB topic |
| `WAN_FRAME_MAX_AGE_SECONDS` | `10` | Reject old publisher frames |
| `MUSETALK_URL` | `http://127.0.0.1:8011` | Local sidecar URL |
| `MUSETALK_OUTPUT_FPS` | `16` | Built-in browser and MuseTalk frame rate |
| `MUSETALK_TIMEOUT_SECONDS` | `60` | Per audio-segment timeout |
| `MUSE_TALK_TOKEN` | empty | Optional bearer token |
| `WANMUSE_FACE_BBOX` | empty | Fixed face ROI |
| `WANMUSE_AUDIO_SEGMENT_SECONDS` | `1.0` | TTS segment size sent to MuseTalk |
| `WANMUSE_STRICT` | `false` | Fail instead of falling back to a still frame |
| `MUSE_TALK_PYTHON` | LiveTalking `.venv` | Sidecar Python executable for launcher |

## Degraded behavior

If the TI2V publisher has not produced a frame, the uploaded reference image is
used. If MuseTalk is unavailable and `WANMUSE_STRICT=false`, audio is still sent
with the latest frame so conversation remains usable. Set `WANMUSE_STRICT=true`
when testing to surface sidecar failures immediately.

## Latency tuning

Start with one-second audio segments. Reduce
`WANMUSE_AUDIO_SEGMENT_SECONDS` to `0.5` only after the sidecar has enough
throughput, because shorter segments increase Whisper/VAE and HTTP overhead.
Keep TI2V outside the audio-critical path; the latest frame is sampled when each
segment begins.

## Tests

```bash
python -m unittest -v tests/test_wanmuse_components.py
python -m py_compile \
  core/wanmuse/*.py \
  core/lip_sync_factory.py \
  services/musetalk_sidecar.py \
  services/ti2v_frame_publisher.py
bash -n scripts/run_app.sh scripts/run_wanmuse_stack.sh
```
