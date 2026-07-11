#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="Qwen35-35b-a3b-nvfp4"
PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping container ${CONTAINER_NAME}"
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Removing container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

rm -f "${PID_FILE}"
echo "Stopped. Log preserved at ${LOG_FILE}"
