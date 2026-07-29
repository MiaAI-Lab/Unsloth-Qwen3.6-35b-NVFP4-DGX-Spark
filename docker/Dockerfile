# syntax=docker/dockerfile:1.7
#
# Bake a Spark/GB10 image that can take:
#   --linear-backend flashinfer_b12x
# on mixed FP8+NVFP4 models without aborting EngineCore.
#
# Base (timothystewart6/vllm-gb10) already has on sm_121:
#   b12x gemm True | b12x moe True
# Stock vLLM still hard-errors when a forced linear backend has no kernel for
# a given layer type (e.g. FP8 dense). This image soft-falls those layers back
# to auto selection so NVFP4 linears can use b12x while FP8 keeps CUTLASS/etc.
#
# Build ON the DGX Spark (linux/arm64):
#   ./docker/build.sh

ARG BASE_IMAGE=ghcr.io/timothystewart6/vllm-gb10:latest
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="mia-vllm-gb10-linear-b12x" \
      org.opencontainers.image.description="vllm-gb10 + soft --linear-backend=flashinfer_b12x for mixed NVFP4 models" \
      org.opencontainers.image.source="https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark"

ENV CUTE_DSL_ARCH=sm_121a \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a

COPY patch_linear_backend_soft_fallback.py /tmp/patch_linear_backend_soft_fallback.py
RUN python3 /tmp/patch_linear_backend_soft_fallback.py \
 && rm -f /tmp/patch_linear_backend_soft_fallback.py \
 && python3 - <<'PY'
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/__init__.py")
text = p.read_text()
assert "falling back to auto selection for this layer type" in text, "soft-fallback string missing"
assert text.count("falling back to auto selection for this layer type") >= 5, (
    "expected >=5 soft filter sites"
)
print("soft-fallback patch OK")
PY

COPY verify_b12x.py /usr/local/bin/verify_b12x.py
RUN chmod +x /usr/local/bin/verify_b12x.py

ENTRYPOINT []
CMD ["bash"]
