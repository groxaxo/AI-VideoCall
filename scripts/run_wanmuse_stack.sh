#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIVETALKING_ROOT=${MUSE_TALK_LIVETALKING_ROOT:-/home/op/LiveTalking}
WAN_GPU=${WAN_GPU:-0}
MUSETALK_GPU=${MUSETALK_GPU:-1}
APP_GPU=${APP_GPU:-2}
MUSE_TALK_PYTHON=${MUSE_TALK_PYTHON:-$LIVETALKING_ROOT/.venv/bin/python}
if [ ! -x "$MUSE_TALK_PYTHON" ]; then
    MUSE_TALK_PYTHON=${PYTHON_BIN:-python}
fi

if [ ! -d "$LIVETALKING_ROOT" ]; then
    echo "LiveTalking not found at $LIVETALKING_ROOT" >&2
    exit 1
fi

cd "$ROOT_DIR"
mkdir -p logs

# The Wan/ABot process is intentionally not started here because deployments
# differ. It must bind WAN_FRAME_ENDPOINT and publish encoded frames.
echo "Expecting TI2V publisher at ${WAN_FRAME_ENDPOINT:-tcp://127.0.0.1:5560} on physical GPU $WAN_GPU"

CUDA_VISIBLE_DEVICES="$MUSETALK_GPU" \
MUSE_TALK_LIVETALKING_ROOT="$LIVETALKING_ROOT" \
MUSE_TALK_DEVICE=cuda \
"$MUSE_TALK_PYTHON" -m services.musetalk_sidecar \
    > logs/musetalk-sidecar.log 2>&1 &
MUSETALK_PID=$!

cleanup() {
    kill "$MUSETALK_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
    if curl -fsS "${MUSETALK_URL:-http://127.0.0.1:8011}/healthz" >/dev/null; then
        break
    fi
    sleep 1
done

curl -fsS "${MUSETALK_URL:-http://127.0.0.1:8011}/healthz" >/dev/null || {
    echo "MuseTalk sidecar failed health check; inspect logs/musetalk-sidecar.log" >&2
    exit 1
}

CUDA_VISIBLE_DEVICES="$APP_GPU" \
VIDEO_BACKEND=ti2v5b_musetalk \
exec bash scripts/run_app.sh
