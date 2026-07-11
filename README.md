# Qwen3.6-35B-A3B-NVFP4-Fast — Self-Hosted Inference via vLLM

[![GPU: GB10 / SM121](https://img.shields.io/badge/GPU-GB10%20%2F%20SM121-76B900)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![Model](https://img.shields.io/badge/model-unsloth%2FQwen3.6--35B--A3B--NVFP4--Fast-informational)](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast)
[![vLLM](https://img.shields.io/badge/vLLM-0.24.1--dev-5B8DEF)](https://github.com/vllm-project/vllm)

A vLLM deployment for **Unsloth Qwen3.6-35B-A3B-NVFP4-Fast** on NVIDIA GB10 (DGX Spark) — mixed FP8/NVFP4 MoE with FlashInfer B12X kernels, FP8 KV cache, and MTP speculative decoding.

Built on the [reference SM121 runtime](https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm) with the `FlashInferB12xNvFp4LinearKernel` re-enabled and the `flashinfer_b12x` linear backend set extended with FP8/MXFP fallbacks.

---

## Key Features

| Feature | Detail |
|---|---|
| **Model** | `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` — mixed FP8 dense + NVFP4 expert MoE |
| **Inference Engine** | vLLM 0.24.1-dev with FlashInfer 0.6.13, CUDA 13.0 |
| **MoE Backend** | `flashinfer_b12x` (target experts) / `triton` (MTP draft) |
| **Linear Backend** | `flashinfer_b12x` — B12X NVFP4 GEMM + fallbacks for FP8/MXFP layers |
| **Attention** | FlashInfer |
| **KV Cache** | FP8 |
| **Speculative Decoding** | MTP, 2 speculative tokens |
| **Context Window** | Up to 262 144 tokens |
| **API** | OpenAI-compatible `/v1/chat/completions`, `/v1/models` |
| **Vision** | Multi-modal image input (up to 4 per request) |
| **Tool Use** | Qwen3-coder tool-call parser, auto tool choice enabled |
| **Reasoning** | Qwen3 CoT with `<thinking>` blocks, configurable |
| **Chunked Prefill** | Enabled |
| **Async Scheduling** | Enabled |
| **Architecture** | ARM64 / SM121 native (NVIDIA GB10 / DGX Spark) |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **GPU** | NVIDIA GB10 (compute capability 12.1) or compatible SM121 GPU |
| **CUDA** | 12.x / 13.x — NVIDIA driver ≥ 535 |
| **Docker** | 24.0+ with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| **curl** | Readiness probes |
| **Disk** | ~50 GB free for model weights + caches |

---

## Quick Start

### 1. (Optional) Set HuggingFace Token

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 2. Start the Server

```bash
./start.sh
```

This will:
1. Check for Docker and curl on PATH
2. Create the HuggingFace cache directory
3. Remove any stale container with the same name
4. Launch the container with `--gpus all`, `--shm-size=32g`, `--ulimit memlock=-1`, `--cap-add IPC_LOCK`
5. Stream container logs to your terminal
6. Poll `/v1/models` until the server is ready
7. Print the OpenAI base URL

**Expected output (after ~5–10 minutes for first load):**
```
Model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast is already cached in ...
Starting vLLM container for unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
Image: ghcr.io/miaai-lab/unsloth-qwen3.6-35b-nvfp4-fast-recipe:latest
Listening on 0.0.0.0:8888
Spawned container Qwen35-35b-a3b-nvfp4 (abc123...)
Waiting for HTTP readiness at http://127.0.0.1:8888/v1/models
--- container logs ---
...
Using FlashInferB12xNvFp4LinearKernel for NVFP4 GEMM
Using 'FLASHINFER_B12X' NvFp4 MoE backend
...
vLLM is ready!
OpenAI base URL: http://0.0.0.0:8888/v1
```

### 3. Test It

```bash
# Quick health check
curl http://0.0.0.0:8888/v1/models | jq

# Chat completion
curl -s http://0.0.0.0:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast",
    "messages": [{"role": "user", "content": "What is 19 × 23?"}],
    "temperature": 0,
    "max_tokens": 100
  }' | jq
```

### 4. Stop the Server

```bash
./stop.sh
```

---

## Configuration

All options in [`start.sh`](start.sh). Key variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` | HuggingFace model identifier |
| `IMAGE` | `ghcr.io/miaai-lab/unsloth-qwen3.6-35b-nvfp4-fast-recipe:latest` | Custom vLLM Docker image |
| `CONTAINER_NAME` | `Qwen35-35b-a3b-nvfp4` | Docker container name |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8888` | HTTP port |
| `HF_TOKEN` | — | HuggingFace auth token (optional) |

### vLLM Flags

| Flag | Value | Description |
|---|---|---|
| `--tensor-parallel-size` | `1` | Single-GPU |
| `--trust-remote-code` | — | Required by Qwen models |
| `--moe-backend` | `flashinfer_b12x` | FlashInfer B12X MoE kernel |
| `--linear-backend` | `flashinfer_b12x` | FlashInfer B12X NVFP4 GEMM + FP8/MXFP fallbacks |
| `--attention-backend` | `flashinfer` | FlashInfer attention |
| `--kv-cache-dtype` | `fp8` | FP8 KV cache |
| `--gpu-memory-utilization` | `0.90` | 90 % of GPU memory |
| `--max-model-len` | `262144` | 256K context window |
| `--max-num-seqs` | `27` | Max concurrent sequences |
| `--max-num-batched-tokens` | `32768` | Max tokens per batch |
| `--enable-chunked-prefill` | — | Improves throughput |
| `--async-scheduling` | — | Async scheduling |
| `--speculative-config` | MTP, 2 tokens | Multi-token prediction |
| `--reasoning-parser` | `qwen3` | Qwen3 CoT parser |
| `--default-chat-template-kwargs` | `{"enable_thinking":true,"preserve_thinking":true}` | Thinking block behaviour |
| `--tool-call-parser` | `qwen3_coder` | Qwen3 tool-call format |
| `--enable-auto-tool-choice` | — | Auto tool selection |
| `--override-generation-config` | temp=0.6, top_p=0.95, top_k=20 | Default sampling params |

---

## Custom Chat Template

[`chat_template.jinja`](chat_template.jinja) is a comprehensive Jinja2 template for Qwen3.6 with:

- **Multi-modal content** — image/video special tokens
- **Thinking / Reasoning Blocks** — `<thinking>...</thinking>` with `enable_thinking` / `preserve_thinking` kwargs
- **Tool Calling** — `<tool_call>` / `<tool_response>` format with JSON parameter rendering
- **Error Recovery** — Consecutive tool-call error detection with retry warnings
- **Auto-disabling Thinking** — Disables thinking when tools are active
- **Content Truncation** — `max_tool_arg_chars` / `max_tool_response_chars`

### Template Kwargs

| Kwarg | Type | Default | Description |
|---|---|---|---|
| `enable_thinking` | bool | `true` | Enable `<thinking>` blocks |
| `preserve_thinking` | bool | `true` | Preserve thinking from history |
| `auto_disable_thinking_with_tools` | bool | `false` | Disable thinking when tools are defined |
| `add_vision_id` | bool | `false` | Prepend "Picture N:" before vision tokens |
| `max_tool_arg_chars` | int | `0` | Truncate tool arguments (0 = no limit) |
| `max_tool_response_chars` | int | `0` | Truncate tool responses (0 = no limit) |

---

## Project Structure

```
qwen36-35b/
├── README.md               ← This file
├── start.sh                ← Launch script
├── stop.sh                 ← Stop & cleanup
├── chat_template.jinja     ← Custom Jinja chat template
├── .vllm.log               ← Container startup log
├── .vllm.pid               ← Container ID
└── .cache/huggingface/     ← Model weights cache
```

---

## Docker Details

| Property | Value |
|---|---|
| **Image** | `ghcr.io/miaai-lab/unsloth-qwen3.6-35b-nvfp4-fast-recipe:latest` |
| **Container Name** | `Qwen35-35b-a3b-nvfp4` |
| **Network** | `host` mode |
| **IPC** | `host` mode |
| **GPUs** | All (`--gpus all`) |
| **Shared Memory** | `--shm-size=32g` |
| **ulimit** | `memlock=-1:-1` |
| **cap_add** | `IPC_LOCK` |
| **Environment** | `CUTE_DSL_ARCH=sm_121a`, `VLLM_TARGET_DEVICE=cuda`, `HF_HOME` |
| **Volumes** | HF cache mount, working directory |

---

## Performance Notes

- **90 % GPU memory** allocated (`--gpu-memory-utilization 0.90`). Model loads ~22 GiB.
- **27 concurrent sequences** (`--max-num-seqs 27`) based on measured KV cache capacity of ~7.1M FP8 tokens at 262K context.
- **32768 batched tokens** (`--max-num-batched-tokens 32768`) for efficient batching.
- **Speculative decoding** (MTP, 2 tokens) improves decode speed; acceptance rate ~86 %.
- **FlashInfer B12X** MoE and linear kernels provide native SM121 NVFP4 execution.
- **FP8 KV cache** balances memory efficiency and accuracy.
- **Decode speed at c1:** ~80 tok/s on GB10. Higher concurrency increases total throughput (up to ~340 tok/s at c32 in published benchmarks).

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `docker is not on PATH` | Install Docker or add to `PATH` |
| `vLLM container exited before becoming ready` | Check container logs that streamed to terminal |
| `Error: cannot access '...'` | Set `HF_TOKEN` and re-run |
| OOM errors | Reduce `--gpu-memory-utilization` or `--max-num-seqs` |
| Model weights not downloading | Verify HF token and network; check `.cache/huggingface/` |
| Container won't stop | `docker rm -f Qwen35-35b-a3b-nvfp4` then `./stop.sh` |
| `'NoneType' object is not subscriptable` | Likely caused by experimental prefix caching with Mamba layers; ensure `--enable-prefix-caching` is not set |

---

## License

- **Model weights:** See [Unsloth model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast) for licensing
- **This codebase:** MIT License

---

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3.6 on HuggingFace](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast)
- [Reference SM121 runtime](https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
