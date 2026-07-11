FROM ghcr.io/r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm:latest

LABEL org.opencontainers.image.title="Custom SM121 vLLM with flashinfer_b12x linear backend"
LABEL org.opencontainers.image.description="Extends the reference image: B12x linear kernel re-enabled, fallback kernels in b12x backend set"

COPY patch_b12x_backend.py /tmp/patch_b12x_backend.py
RUN python3 /tmp/patch_b12x_backend.py && rm /tmp/patch_b12x_backend.py
