#!/usr/bin/env bash
# Build the linear-b12x soft-fallback image on DGX Spark (arm64 + sm_121).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/timothystewart6/vllm-gb10:latest}"
TAG="${TAG:-ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest}"

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: host is $(uname -m). Spark images should be built on arm64 (DGX Spark)."
  echo "Continue only if you know you are cross-building."
fi

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

echo "Pulling base: ${BASE_IMAGE}"
docker pull "${BASE_IMAGE}"

echo "Building ${TAG} from ${BASE_IMAGE}"
docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${TAG}" \
  -f "${ROOT}/Dockerfile" \
  "${ROOT}"

echo
echo "Built: ${TAG}"
echo "Verify (needs GPU):"
echo "  docker run --rm --gpus all --entrypoint python3 ${TAG} /usr/local/bin/verify_b12x.py"
echo
echo "Point start.sh at it:"
echo "  IMAGE=\"${TAG}\""
