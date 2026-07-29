# Qwen3.6-35B-A3B-NVFP4 — DGX Spark

[![GPU: GB10 / SM121](https://img.shields.io/badge/GPU-GB10%20%2F%20SM121-76B900)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![Model](https://img.shields.io/badge/model-unsloth%2FQwen3.6--35B--A3B--NVFP4-informational)](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4)
[![vLLM](https://img.shields.io/badge/vLLM-0.26%20(gb10)-5B8DEF)](https://github.com/vllm-project/vllm)
[![Image](https://img.shields.io/badge/GHCR-mia--vllm--gb10--linear--b12x-blue)](https://github.com/users/MiaAI-Lab/packages/container/package/mia-vllm-gb10-linear-b12x)

A vLLM deployment for **Unsloth Qwen3.6-35B-A3B-NVFP4** on NVIDIA DGX Spark (GB10) — mixed FP8 dense + NVFP4 MoE, FlashInfer B12X linear GEMM (with soft fallback for non-NVFP4 layers), FP8 KV cache, and MTP speculative decoding.

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>

---

## Key Features

| Feature | Detail |
|---|---|
| **Model** | `unsloth/Qwen3.6-35B-A3B-NVFP4` — mixed FP8 dense + NVFP4 expert MoE |
| **Docker image** | `ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest` |
| **Base stack** | [timothystewart6/vllm-gb10](https://github.com/timothystewart6/vllm-gb10) (ARM64 / SM121, CUDA 13, FlashInfer 0.6.14, vLLM ~0.26) |
| **Linear backend** | `--linear-backend flashinfer_b12x` — NVFP4 GEMM via B12X; non-NVFP4 layers soft-fall back to auto |
| **MoE backend** | `auto` (vLLM selects the best available path for this quant) |
| **Attention** | FlashInfer |
| **KV cache** | FP8 |
| **Speculative decoding** | MTP, 2 speculative tokens |
| **Context window** | Up to 262 144 tokens |
| **API** | OpenAI-compatible `/v1/chat/completions`, `/v1/models` |
| **Vision** | Multi-modal image input (up to 4 per request) |
| **Tool use** | Qwen3-coder tool-call parser, auto tool choice |
| **Reasoning** | Qwen3 CoT with thinking blocks (`enable_thinking` / `preserve_thinking`) |
| **Architecture** | ARM64 / SM121 native (NVIDIA GB10 / DGX Spark) |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **GPU** | NVIDIA GB10 (compute capability 12.1) / DGX Spark |
| **Host** | `linux/arm64` (Spark). Image is not for x86 / RTX 5090 |
| **Docker** | 24.0+ with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| **curl** | Readiness probes |
| **Disk** | ~50 GB for model weights + caches; extra space if rebuilding the image |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark.git
cd Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark
```

### 2. (Optional) Hugging Face token

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 3. Start the server

```bash
./start.sh
```

`start.sh` will:

1. Check for Docker and curl  
2. Use a **local** image if present; otherwise **`docker pull`** `ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest` (image is resolved **before** model download, since download can fall back to `docker run` with this image)  
3. Ensure the model is available in the local HF cache (download if needed)  
4. Launch with `--gpus all`, host network/IPC, large shared memory, `CUTE_DSL_ARCH=sm_121a`  
5. Stream logs and wait until `http://127.0.0.1:8888/v1/models` is ready  

**Example output:**

```text
Using local image: ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest
# (or: Pulling image: ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest)
Model unsloth/Qwen3.6-35B-A3B-NVFP4 is already cached in ...
Starting vLLM container for unsloth/Qwen3.6-35B-A3B-NVFP4
Image: ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest
Listening on 0.0.0.0:8888
...
vLLM is ready!
OpenAI base URL: http://0.0.0.0:8888/v1
```

First load can take several minutes (model + kernel warmup). Later restarts are much faster if weights are cached under `.cache/huggingface/`.

### 4. Test

```bash
curl -s http://127.0.0.1:8888/v1/models | jq

curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "What is 19 × 23?"}],
    "temperature": 0,
    "max_tokens": 100
  }' | jq
```

### 5. Stop

```bash
./stop.sh
```

---

## Docker image

| Property | Value |
|---|---|
| **Published image** | [`ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest`](https://github.com/users/MiaAI-Lab/packages/container/package/mia-vllm-gb10-linear-b12x) |
| **Base** | `ghcr.io/timothystewart6/vllm-gb10:latest` (SM121 / FlashInfer B12X GEMM + MoE available) |
| **What we add** | Soft-fallback so `--linear-backend flashinfer_b12x` does **not** abort on mixed FP8+NVFP4 layers |

### Why a custom image?

Stock vLLM treats `--linear-backend` as a hard filter: if a layer type has no kernel for that backend, EngineCore exits with:

```text
ValueError: --linear-backend=flashinfer_b12x was requested but no
'flashinfer_b12x' kernel exists for this layer type.
```

Unsloth Qwen3.6 NVFP4 is **mixed** (FP8 dense linears + NVFP4 experts/linears). B12X is the right path for **NVFP4 GEMM**, but FP8 (and other) linears need their own kernels.

This image patches linear selection so that:

| Layer type | With `--linear-backend flashinfer_b12x` |
|---|---|
| NVFP4 linears | Use FlashInfer B12X GEMM when available |
| FP8 / other linears | Soft-fall back to auto selection (log a warning once) |

Base image already reports `b12x gemm True` and `b12x moe True` on GB10; the patch only changes **forced-backend** behavior.

### Pull

```bash
docker pull ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest
```

The package is published on GHCR under MiaAI-Lab. If a future tag is private or rate-limited, log in:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin
```

### Rebuild on Spark (optional)

Build **on the DGX Spark** (`aarch64`), not on an x86 PC:

```bash
./docker/build.sh
# optional overrides:
# BASE_IMAGE=ghcr.io/timothystewart6/vllm-gb10:latest \
# TAG=ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest \
# ./docker/build.sh
```

Verify GPU + B12X + patch:

```bash
docker run --rm --gpus all --entrypoint python3 \
  ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest \
  /usr/local/bin/verify_b12x.py
```

Expected:

```text
cap (12, 1)
b12x gemm True | b12x moe True
linear soft-fallback patch True
OK: image ready for --linear-backend flashinfer_b12x on mixed NVFP4
```

### Image sources (`docker/`)

| File | Role |
|---|---|
| `Dockerfile` | `FROM` vllm-gb10 + apply soft-fallback patch |
| `patch_linear_backend_soft_fallback.py` | Converts hard `ValueError` on empty backend filter → warning + auto for that layer |
| `build.sh` | Pull base, build, tag |
| `verify_b12x.py` | Runtime probe for SM12x + B12X + patch |

---

## Configuration

Options live in [`start.sh`](start.sh).

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `unsloth/Qwen3.6-35B-A3B-NVFP4` | Hugging Face model id |
| `IMAGE` | `ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest` | Docker image (local hit skips pull) |
| `CONTAINER_NAME` | `Qwen35-35b-a3b-nvfp4` | Container name |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8888` | HTTP port |
| `HF_TOKEN` | — | Optional HF auth |

### vLLM flags (as launched)

| Flag | Value | Description |
|---|---|---|
| `--tensor-parallel-size` | `1` | Single GPU |
| `--trust-remote-code` | — | Required for Qwen |
| `--moe-backend` | `auto` | Best available MoE path for detected quant |
| `--linear-backend` | `flashinfer_b12x` | Prefer B12X for NVFP4 linears (soft fallback elsewhere) |
| `--attention-backend` | `flashinfer` | FlashInfer attention |
| `--kv-cache-dtype` | `fp8` | FP8 KV cache |
| `--gpu-memory-utilization` | `0.80` | 80% GPU memory |
| `--max-model-len` | `262144` | 256K context |
| `--max-num-seqs` | `24` | Max concurrent sequences |
| `--max-num-batched-tokens` | `32768` | Batch token cap |
| `--enable-chunked-prefill` | — | Chunked prefill |
| `--async-scheduling` | — | Async scheduling |
| `--speculative-config` | MTP, 2 tokens (`moe_backend: triton` for draft) | Speculative decode |
| `--reasoning-parser` | `qwen3` | Thinking / CoT parser |
| `--default-chat-template-kwargs` | `enable_thinking` + `preserve_thinking` | Thinking defaults |
| `--tool-call-parser` | `qwen3_coder` | Tool calls |
| `--enable-auto-tool-choice` | — | Auto tools |
| `--override-generation-config` | temp=0.6, top_p=0.95, top_k=20 | Default sampling |

### Runtime environment (container)

| Variable | Value |
|---|---|
| `CUTE_DSL_ARCH` | `sm_121a` |
| `VLLM_TARGET_DEVICE` | `cuda` |
| `HF_HOME` | `/root/.cache/huggingface` (host dir mounted) |

### Container runtime

| Property | Value |
|---|---|
| Network / IPC | `host` |
| GPUs | `--gpus all` |
| Shared memory | `--shm-size=32g` |
| ulimit | `memlock=-1:-1` |
| cap_add | `IPC_LOCK` |
| Volumes | HF cache + workspace |

---

## Linear B12X vs MoE B12X

| Flag | Affects | Notes |
|---|---|---|
| `--linear-backend flashinfer_b12x` | Dense **linear** GEMMs | NVFP4 linears use B12X; mixed FP8 layers need soft fallback (this image) |
| `--moe-backend flashinfer_b12x` | **MoE experts** | Only valid when vLLM classifies experts as **NVFP4 MoE**. If the engine reports **FP8 MoE**, forcing B12X fails — keep `auto` (current default) |

For tok/s on Spark, MoE backend choice usually matters more than linear B12X. Linear B12X still helps NVFP4 GEMM layers when available.

---

## Recommended client configuration (coding)

OpenAI-compatible API on port `8888`.

### Provider

```json
{
  "baseUrl": "http://localhost:8888/v1",
  "api": "openai-completions",
  "apiKey": "dummy",
  "compat": {
    "supportsDeveloperRole": false,
    "supportsReasoningEffort": false,
    "maxTokensField": "max_tokens"
  }
}
```

### Model entry

```json
{
  "id": "unsloth/Qwen3.6-35B-A3B-NVFP4",
  "name": "Unsloth Qwen3.6 35B A3B NVFP4",
  "reasoning": true,
  "input": ["text", "image"],
  "contextWindow": 262144,
  "maxTokens": 32000,
  "params": {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 0,
    "repetition_penalty": 1,
    "chat_template_kwargs": {
      "enable_thinking": true,
      "preserve_thinking": true
    }
  }
}
```

The `model` field in requests must match the served model id (defaults to the Hugging Face model id).

> **Sampling:** For precise coding (thinking mode), `temperature=0.6, top_p=0.95, top_k=20` is a good default. For general thinking tasks, Qwen often suggests higher temperature (e.g. `1.0`). See the [Unsloth Qwen3.6 guide](https://unsloth.ai/docs/models/qwen3.6).

---

## Performance notes

- **~80% GPU memory** (`--gpu-memory-utilization 0.80`); model weights on the order of ~22 GiB.  
- **24 concurrent sequences** and **32 768** batched tokens tuned for long context + batching.  
- **MTP** (2 tokens) improves decode; acceptance rate depends on workload.  
- **FlashInfer B12X** linear path for NVFP4 GEMM on SM121.  
- **FP8 KV cache** for memory efficiency.  
- **Decode (c1):** on the order of ~80 tok/s on GB10 for this recipe (workload-dependent).  
- **TTFT:** previously measured P50 ~103 ms / P95 ~107 ms at c1 (200-token prompt, 50-token gen) on the older stack — re-benchmark after stack changes if you need hard numbers.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `pull access denied` / package not found | Image private or missing: log in to `ghcr.io`, set package **Public**, or `./docker/build.sh` |
| `Failed to pull image (and no local image…)` | Build with `./docker/build.sh` or fix GHCR access |
| `docker is not on PATH` | Install Docker / NVIDIA Container Toolkit |
| `vLLM container exited before becoming ready` | Read streamed logs; check OOM, HF download, backend errors |
| `linear-backend=flashinfer_b12x … no kernel` | You are not on this image’s soft-fallback patch — pull/build `mia-vllm-gb10-linear-b12x` |
| `moe_backend='flashinfer_b12x' is not supported for FP8 MoE` | Experts classified as FP8; leave `--moe-backend auto` (default) |
| HF download / access errors | Set `HF_TOKEN`; check `.cache/huggingface/` |
| OOM | Lower `--gpu-memory-utilization` or `--max-num-seqs` / `--max-model-len` |
| Container stuck | `docker rm -f Qwen35-35b-a3b-nvfp4` then `./stop.sh` |

---

## License

- **Model weights:** see [Unsloth model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4)  
- **This codebase:** MIT License  
- **Base image:** [vllm-gb10](https://github.com/timothystewart6/vllm-gb10) (MIT) + upstream vLLM / FlashInfer licenses  

---

## Resources

- [Package on GHCR](https://github.com/users/MiaAI-Lab/packages/container/package/mia-vllm-gb10-linear-b12x)  
- [vllm-gb10 (base Spark image)](https://github.com/timothystewart6/vllm-gb10)  
- [Unsloth NVFP4 guide](https://unsloth.ai/docs/basics/nvfp4)  
- [Unsloth Qwen3.6](https://unsloth.ai/docs/models/qwen3.6)  
- [vLLM docs](https://docs.vllm.ai/)  
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)  
- [Model on Hugging Face](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4)  
