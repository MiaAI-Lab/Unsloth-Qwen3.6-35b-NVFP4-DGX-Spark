#!/usr/bin/env python3
"""Measure P95 TTFT (Time to First Token) against a vLLM OpenAI-compatible endpoint."""

import argparse
import statistics
import sys
import time

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P95 TTFT benchmark")
    parser.add_argument("--base-url", default="http://localhost:8888/v1",
                        help="OpenAI-compatible base URL (default: %(default)s)")
    parser.add_argument("--model", default="unsloth/Qwen3.6-35B-A3B-NVFP4",
                        help="Model name (default: %(default)s)")
    parser.add_argument("--num-requests", type=int, default=20,
                        help="Number of requests (default: %(default)s)")
    parser.add_argument("--prompt-length", type=int, default=256,
                        help="Approximate prompt length in tokens (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max generation tokens (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup requests (default: %(default)s)")
    return parser.parse_args()


def ttft(args: argparse.Namespace) -> list[float]:
    """Return list of TTFT values (seconds)."""
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Build a prompt of roughly `args.prompt_length` tokens
    words = args.prompt_length // 2
    prompt = " ".join(["word"] * words)

    ttfts: list[float] = []
    total = args.warmup + args.num_requests

    for i in range(total):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
        }

        first = True
        start = time.monotonic()

        try:
            resp = requests.post(url, json=payload, headers=headers,
                                 stream=True, timeout=(60, 120))
            resp.raise_for_status()

            for chunk in resp.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                if chunk.startswith("data: [DONE]"):
                    break
                if first:
                    elapsed = time.monotonic() - start
                    ttfts.append(elapsed)
                    first = False
        except Exception as e:
            print(f"  [error] request {i + 1}: {e}", file=sys.stderr)
            continue

        kind = "warmup" if i < args.warmup else "bench"
        print(f"  {kind} {i + 1:>3d}: TTFT={elapsed * 1000:.0f}ms")

    # Return only the bench runs
    return ttfts[args.warmup:]


def main() -> None:
    args = parse_args()

    print(f"Model:      {args.model}")
    print(f"Endpoint:   {args.base_url}")
    print(f"Requests:   {args.num_requests} (+ {args.warmup} warmup)")
    print(f"Prompt:     ~{args.prompt_length} tokens")
    print(f"Max tokens: {args.max_tokens}")
    print()

    ttfts = ttft(args)

    if len(ttfts) < 2:
        print("\nNot enough successful requests to report stats.", file=sys.stderr)
        sys.exit(1)

    ttfts.sort()
    n = len(ttfts)
    p50 = ttfts[int(n * 0.50)]
    p90 = ttfts[int(n * 0.90)]
    p95 = ttfts[int(n * 0.95)]
    p99 = ttfts[int(n * 0.99)]

    print()
    print(f"Runs:        {n}")
    print(f"Mean TTFT:   {statistics.mean(ttfts) * 1000:.0f}ms")
    print(f"Median TTFT: {p50 * 1000:.0f}ms")
    print(f"P90  TTFT:   {p90 * 1000:.0f}ms")
    print(f"P95  TTFT:   {p95 * 1000:.0f}ms")
    print(f"P99  TTFT:   {p99 * 1000:.0f}ms")
    print(f"Min  TTFT:   {ttfts[0] * 1000:.0f}ms")
    print(f"Max  TTFT:   {ttfts[-1] * 1000:.0f}ms")


if __name__ == "__main__":
    main()
