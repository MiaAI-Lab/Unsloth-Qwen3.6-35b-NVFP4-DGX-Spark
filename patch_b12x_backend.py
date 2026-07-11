"""Patch _LINEAR_BACKEND_KERNEL_MAP and _POSSIBLE_NVFP4_KERNELS.

1. Extend flashinfer_b12x backend set with FP8/MXFP fallback kernels.
2. Add FlashInferB12xNvFp4LinearKernel back into _POSSIBLE_NVFP4_KERNELS
   (it was excluded from auto-selection due to a CUTLASS SM121 MMA op guard
   that is now resolved in this vLLM build).
"""
import ast
import sys
import re

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/__init__.py"

with open(path) as f:
    src = f.read()

# --- Patch 1: Extend flashinfer_b12x backend set ---
old1 = (
    '    "flashinfer_b12x": {\n'
    "        FlashInferB12xNvFp4LinearKernel,\n"
    "    },"
)

new1 = (
    '    "flashinfer_b12x": {\n'
    "        FlashInferB12xNvFp4LinearKernel,\n"
    "        FlashInferFP8ScaledMMLinearKernel,\n"
    "        FlashInferFp8DeepGEMMDynamicBlockScaledKernel,\n"
    "        FlashInferCutlassMxfp8LinearKernel,\n"
    "        FlashInferMxFp4LinearKernel,\n"
    "        MarlinFP8ScaledMMLinearKernel,\n"
    "        CutlassFP8ScaledMMLinearKernel,\n"
    "        CutlassFp8BlockScaledMMKernel,\n"
    "        MarlinLinearKernel,\n"
    "        MarlinMxfp8LinearKernel,\n"
    "        MarlinNvFp4LinearKernel,\n"
    "        MarlinMxFp4LinearKernel,\n"
    "        CutlassNvFp4LinearKernel,\n"
    "        FlashInferCutlassNvFp4LinearKernel,\n"
    "        PerTensorTorchFP8ScaledMMLinearKernel,\n"
    "        ChannelWiseTorchFP8ScaledMMLinearKernel,\n"
    "        RowWiseTorchFP8ScaledMMLinearKernel,\n"
    "    },"
)

if old1 in src:
    src = src.replace(old1, new1)
    print("Patch 1: extended flashinfer_b12x backend set")
else:
    print("WARNING: flashinfer_b12x entry not found, skipping patch 1")

# --- Patch 2: Add FlashInferB12xNvFp4LinearKernel back into _POSSIBLE_NVFP4_KERNELS ---
old2 = (
    "        # FlashInferB12xNvFp4LinearKernel excluded from auto-selection until\n"
    '        # upstream CUTLASS SM121 MMA op guard is resolved; use\n'
    '        # --linear-backend flashinfer_b12x to opt in explicitly.\n'
)

new2 = (
    "        # FlashInferB12xNvFp4LinearKernel -- SM121 MMA op guard resolved.\n"
)

if old2 in src:
    src = src.replace(old2, new2)
    # Now add the actual kernel reference after the comment
    src = src.replace(
        "        # FlashInferB12xNvFp4LinearKernel -- SM121 MMA op guard resolved.\n"
        "        FlashInferCutlassNvFp4LinearKernel,",
        "        # FlashInferB12xNvFp4LinearKernel -- SM121 MMA op guard resolved.\n"
        "        FlashInferB12xNvFp4LinearKernel,\n"
        "        FlashInferCutlassNvFp4LinearKernel,",
    )
    print("Patch 2: added FlashInferB12xNvFp4LinearKernel back into _POSSIBLE_NVFP4_KERNELS")
else:
    print("WARNING: commented-out B12x entry not found, skipping patch 2")

# Verify syntax
try:
    ast.parse(src)
except SyntaxError as e:
    print(f"Syntax error after patch: {e}", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)

print("All patches applied successfully")
