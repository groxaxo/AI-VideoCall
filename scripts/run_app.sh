#! /bin/bash
set -euo pipefail

function get_gpu_count() {
    if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --list-gpus | wc -l
        else
            echo "0"
        fi
    elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "0"
    else
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l
    fi
}

export GLOO_SOCKET_IFNAME=${MLP_SOCKET_IFNAME:-}
export NCCL_SOCKET_IFNAME=${MLP_SOCKET_IFNAME:-}
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=./.inductor_cache
export TORCHDYNAMO_VERBOSE=${TORCHDYNAMO_VERBOSE:-1}
export GPU_COUNT=$(get_gpu_count)

export NCCL_PXN_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=4
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=25
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=VERSION
export NCCL_IB_TC=106

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

VIDEO_BACKEND=${VIDEO_BACKEND:-self_forcing_s2v}
if [ "$VIDEO_BACKEND" = "ti2v5b_musetalk" ]; then
    echo "Starting AI VideoCall with TI2V-5B + MuseTalk backend (single process)..."
    exec python app.py
fi

echo "Starting Standalone Torchrun on $GPU_COUNT GPUs..."
exec torchrun \
    --standalone \
    --nproc_per_node="$GPU_COUNT" \
    app.py
