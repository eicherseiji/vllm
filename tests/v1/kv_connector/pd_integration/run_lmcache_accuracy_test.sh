#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# LMCacheConnectorV1 accuracy test script
# Uses disagg_proxy_server.py from lmcache examples with YAML config files

set -xe

# Required for consistent hashing in P/D disaggregation
export PYTHONHASHSEED=0

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-0.6B"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.3}
GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(dirname "$0")

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr) 2>/dev/null; rm -f $PREFILL_CONFIG $DECODE_CONFIG 2>/dev/null' SIGINT SIGTERM EXIT

wait_for_server() {
  local port=$1
  timeout 600 bash -c "until curl -s localhost:${port}/v1/completions > /dev/null; do sleep 1; done"
}

# Cleanup
pkill -f "vllm serve" || true
sleep 2

# Create temp LMCache config files using the current lmcache config format
PREFILL_CONFIG=$(mktemp --suffix=.yaml)
DECODE_CONFIG=$(mktemp --suffix=.yaml)

cat > $PREFILL_CONFIG << 'EOF'
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL

enable_pd: True
pd_role: "sender"
pd_buffer_size: 1086324736
pd_buffer_device: "cuda"
pd_peer_host: "localhost"
pd_peer_init_port: [55555]
pd_peer_alloc_port: [55556]
pd_proxy_host: "localhost"
pd_proxy_port: 8192
transfer_channel: "nixl"
nixl_buffer_size: 1086324736
nixl_buffer_device: "cuda"
EOF

cat > $DECODE_CONFIG << 'EOF'
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL

enable_pd: True
pd_role: "receiver"
pd_buffer_size: 1086324736
pd_buffer_device: "cuda"
pd_peer_host: "localhost"
pd_peer_init_port: [55555]
pd_peer_alloc_port: [55556]
pd_proxy_host: "localhost"
pd_proxy_port: 8192
transfer_channel: "nixl"
nixl_buffer_size: 1086324736
nixl_buffer_device: "cuda"
EOF

# Prefill instance (kv_producer)
LMCACHE_CONFIG_FILE=$PREFILL_CONFIG \
LMCACHE_USE_EXPERIMENTAL=True \
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME \
  --port 8100 \
  --max-model-len 4096 \
  --enforce-eager \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer"}' &

# Decode instance (kv_consumer)
LMCACHE_CONFIG_FILE=$DECODE_CONFIG \
LMCACHE_USE_EXPERIMENTAL=True \
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
  --port 8200 \
  --max-model-len 4096 \
  --enforce-eager \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer"}' &

echo "Waiting for prefill instance on port 8100 to start..."
wait_for_server 8100
echo "Waiting for decode instance on port 8200 to start..."
wait_for_server 8200

# Use LMCache's disagg proxy server
python3 ${GIT_ROOT}/examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_proxy_server.py \
  --port 8192 \
  --prefiller-host localhost \
  --prefiller-port 8100 \
  --decoder-host localhost \
  --decoder-port 8200 &

sleep 5

# Run accuracy test
TEST_MODEL=$MODEL_NAME python3 -m pytest -s -x ${GIT_ROOT}/tests/v1/kv_connector/pd_integration/test_accuracy.py

# Cleanup temp configs
rm -f $PREFILL_CONFIG $DECODE_CONFIG
pkill -f "vllm serve" || true
echo "LMCache accuracy test completed!"
