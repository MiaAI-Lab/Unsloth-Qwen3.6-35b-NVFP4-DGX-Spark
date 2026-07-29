#!/usr/bin/env python3
"""Probe SM121 + FlashInfer b12x GEMM/MoE availability (must run with --gpus)."""

from __future__ import annotations

import sys


def main() -> int:
    import torch
    from importlib.metadata import PackageNotFoundError, version

    def pkg(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "missing"

    print("torch", torch.__version__, "cuda", torch.version.cuda)
    fi_ver = pkg("flashinfer")
    if fi_ver == "missing":
        fi_ver = pkg("flashinfer-python")
    print("vllm", pkg("vllm"), "flashinfer", fi_ver)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available in this container", file=sys.stderr)
        return 1

    cap = torch.cuda.get_device_capability()
    print("cap", cap)

    from vllm.utils.flashinfer import (
        has_flashinfer_b12x_gemm as has_gemm,
        has_flashinfer_b12x_moe as has_moe,
    )

    gemm, moe = has_gemm(), has_moe()
    print("b12x gemm", gemm, "| b12x moe", moe)

    # Confirm soft-fallback patch is present when using the baked image.
    try:
        import vllm.model_executor.kernels.linear as linear_mod
        from pathlib import Path

        src = Path(linear_mod.__file__).read_text()
        soft = "falling back to auto selection for this layer type" in src
        print("linear soft-fallback patch", soft)
    except Exception as exc:  # noqa: BLE001
        print("linear soft-fallback patch check failed:", exc)
        soft = False

    ok = cap[0] == 12 and gemm and moe and soft
    if not ok:
        print(
            "FAIL: need sm_12x + b12x gemm + b12x moe + soft-fallback patch",
            file=sys.stderr,
        )
        return 1
    print("OK: image ready for --linear-backend flashinfer_b12x on mixed NVFP4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
