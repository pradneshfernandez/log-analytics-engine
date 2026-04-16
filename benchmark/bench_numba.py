"""
Numba parser-only benchmark.

Measures just the parsing step (file I/O + Numba kernel) without any
aggregation.  Useful for isolating parser throughput from downstream JAX work.

The first call triggers JIT compilation; we run twice and report the second
time so results reflect steady-state throughput.
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

from src.parser import parse_file


def run_benchmark(path: str) -> dict:
    # Warm-up run: JIT compilation happens here (not counted in results)
    print("  Warming up Numba JIT...", end=" ", flush=True)
    parse_file(path)
    print("done")

    tracemalloc.start()
    t0     = time.perf_counter()

    parsed = parse_file(path)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_lines  = len(parsed["timestamps"])
    file_mb  = Path(path).stat().st_size / 1024 ** 2
    peak_mb  = peak / 1024 ** 2

    return {
        "engine":        "numba",
        "file_mb":       file_mb,
        "n_lines":       n_lines,
        "elapsed_sec":   elapsed,
        "lines_per_sec": n_lines / elapsed,
        "peak_mem_mb":   peak_mb,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.log"
    r    = run_benchmark(path)
    print(f"[numba]   {r['n_lines']:>10,} lines  "
          f"{r['elapsed_sec']:.2f}s  "
          f"{r['lines_per_sec']:>10,.0f} lines/s  "
          f"peak {r['peak_mem_mb']:.0f} MB")
