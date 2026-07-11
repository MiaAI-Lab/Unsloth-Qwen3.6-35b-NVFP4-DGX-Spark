#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="unsloth/Qwen3.6-35B-A3B-NVFP4"
IMAGE="ghcr.io/miaai-lab/unsloth-qwen3.6-35b-nvfp4-fast-dgx-spark:latest"
CONTAINER_NAME="Qwen35-35b-a3b-nvfp4"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
CHAT_URL="http://127.0.0.1:${PORT}/v1/chat/completions"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}"

is_hf_model_id() {
  [[ "${1}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]
}

hf_cache_repo_dir() {
  echo "${HF_HOME}/hub/models--${1//\//--}"
}

model_is_cached() {
  local cache_dir snapshot
  cache_dir="$(hf_cache_repo_dir "${1}")"

  [[ -d "${cache_dir}/snapshots" ]] || return 1

  for snapshot in "${cache_dir}"/snapshots/*/; do
    [[ -d "${snapshot}" ]] || continue
    [[ -f "${snapshot}/config.json" ]] || continue
    if [[ -f "${snapshot}/model.safetensors" ]] \
      || [[ -f "${snapshot}/model.safetensors.index.json" ]] \
      || compgen -G "${snapshot}/model-"*.safetensors >/dev/null; then
      return 0
    fi
  done

  return 1
}

download_model() {
  local model_id="$1"
  echo "Downloading model ${model_id} to ${HF_HOME}"
  echo "This may take a while for large models..."

  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" HF_TOKEN="${HF_TOKEN:-}" \
      hf download "${model_id}" ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" HF_TOKEN="${HF_TOKEN:-}" \
      huggingface-cli download "${model_id}" ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi

  docker run --rm \
    --entrypoint python3 \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    "${IMAGE}" \
    -c "import os; from huggingface_hub import snapshot_download; snapshot_download('${model_id}', token=os.environ.get('HF_TOKEN') or None)"
}

ensure_model_available() {
  if is_hf_model_id "${MODEL_ID}"; then
    if model_is_cached "${MODEL_ID}"; then
      echo "Model ${MODEL_ID} is already cached in ${HF_HOME}"
    else
      download_model "${MODEL_ID}"
    fi
    return
  fi

  if [[ "${MODEL_ID}" == /* || "${MODEL_ID}" == ./* || "${MODEL_ID}" == ../* ]]; then
    if [[ ! -d "${MODEL_ID}" ]]; then
      echo "Local model directory not found: ${MODEL_ID}"
      exit 1
    fi
    echo "Using local model at ${MODEL_ID}"
  fi
}

ensure_model_available

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting vLLM container for ${MODEL_ID}"
echo "Image: ${IMAGE}"
echo "Listening on ${HOST}:${PORT}"
echo "Pulling image: ${IMAGE}"
docker pull "${IMAGE}" 2>&1 || { echo "Failed to pull image"; exit 1; }
echo

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container
EOF
# Override the entrypoint so we control serve flags and use the HF cache mount
# (the baked entrypoint expects a local path mount at /models/model).
docker run -d \
  --name "${CONTAINER_NAME}" \
  --user root \
  --network host \
  --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --cap-add=IPC_LOCK \
  --ipc host \
  --gpus all \
  --workdir /workspace \
  --entrypoint /usr/local/bin/vllm \
  -e VLLM_TARGET_DEVICE=cuda \
  -e CUTE_DSL_ARCH=sm_121a \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${WORK_DIR}:/workspace" \
  "${IMAGE}" \
  serve \
  "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --moe-backend auto \
    --gpu-memory-utilization 0.80 \
    --linear-backend flashinfer_b12x \
    --attention-backend flashinfer \
    --max-model-len 262144 \
    --max-num-seqs 24 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill \
    --async-scheduling \
    --kv-cache-dtype fp8 \
    --limit-mm-per-prompt '{"image":4}' \
    --allowed-media-domains '*' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]]; then
    kill "${log_follow_pid}" 2>/dev/null || true
    wait "${log_follow_pid}" 2>/dev/null || true
    log_follow_pid=""
  fi
}
trap cleanup EXIT INT TERM

echo "Waiting for HTTP readiness at ${READY_URL}"
echo "--- container logs ---"

docker logs -f "${CONTAINER_NAME}" 2>&1 &
log_follow_pid=$!

while ! curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo
    echo "vLLM container exited before becoming ready"
    exit 1
  fi
  sleep 2
done

cleanup

echo
echo "vLLM is ready!"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
