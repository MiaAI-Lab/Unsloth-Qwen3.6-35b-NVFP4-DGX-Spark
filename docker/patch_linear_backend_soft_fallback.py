#!/usr/bin/env python3
"""Make --linear-backend=flashinfer_b12x soft-fail on unsupported layer types.

vLLM's stock behavior: if you force a linear backend and the *current* layer
type has no matching kernel, raise ValueError and abort EngineCore.

Unsloth Qwen3.6 NVFP4 is mixed (FP8 dense + NVFP4 experts/linears). Forcing
flashinfer_b12x is correct for NVFP4 GEMM, but FP8 / other linears must keep
their own kernels (CUTLASS / Marlin / …).

This patch turns empty backend filters into a one-time warning + auto
selection for that layer only. NVFP4 layers still prefer b12x when present.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/__init__.py"
)

# Matches the five hard-fail sites in choose_* / init_* linear selectors.
PATTERN = re.compile(
    r"""(?P<indent>        )if not filtered:\n"""
    r"""            raise ValueError\(\n"""
    r"""                f"--linear-backend=\{linear_backend\} was requested but no "\n"""
    r"""                f"'\{linear_backend\}' kernel exists for (?P<kind>[^"]+)"\n"""
    r"""            \)\n"""
    r"""        (?P<var>platform_kernels|possible) = filtered""",
    re.MULTILINE,
)

REPLACEMENT = """\
\\g<indent>if filtered:
            \\g<var> = filtered
        else:
            logger.warning_once(
                "--linear-backend=%s has no kernel for %s; "
                "falling back to auto selection for this layer type.",
                linear_backend,
                "\\g<kind>",
            )"""


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    text = path.read_text()
    new_text, n = PATTERN.subn(REPLACEMENT, text)
    if n == 0:
        print(f"ERROR: no hard-fail sites patched in {path}", file=sys.stderr)
        return 1
    path.write_text(new_text)
    print(f"Patched {n} linear-backend hard-fail site(s) in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
