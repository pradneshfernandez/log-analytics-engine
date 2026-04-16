"""
Full pipeline benchmark — prints the comparison table.

Tests all three engines (pandas, numba-parse-only, full Numba+JAX pipeline)
on each log file and prints a formatted results table.

Usage
-----
    # Generate test files first:
    python data/generate_logs.py --output data/1gb.log  --size-gb 1
    python data/generate_logs.py --output data/5gb.log  --size-gb 5
    python data/generate_logs.py --output data/10gb.log --size-gb 10

    # Run benchmark:
    python -m benchmark.bench_full                        # auto-detect device
    JAX_PLATFORM_NAME=cpu python -m benchmark.bench_full  # force CPU
    JAX_PLATFORM_NAME=gpu python -m benchmark.bench_full  # force GPU

JAX_PLATFORM_NAME must be set *before* any jax import (i.e. before this
script runs).  On Colab GPU runtime JAX picks up the GPU automatically.
Pass --files to override the default file list.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from benchmark.bench_pandas import run_benchmark as bench_pandas
from benchmark.bench_numba  import run_benchmark as bench_numba
from src.aggregator import compute_all_metrics, compute_n_windows, top_k_endpoints
from src.anomaly    import detect_anomalies, extract_anomaly_events
from src.parser     import parse_file


def _bench_full_pipeline(path: str) -> dict:
    """End-to-end pipeline: parse + aggregate + anomaly detect."""
    print("  Warming up Numba+JAX JIT...", end=" ", flush=True)
    _parsed  = parse_file(path)
    _ts      = np.sort(_parsed["timestamps"])
    _n_wins  = compute_n_windows(_ts)
    _result  = compute_all_metrics(
        jnp.array(_ts), jnp.array(_parsed["status_codes"]),
        jnp.array(_parsed["response_sizes"]), jnp.array(_parsed["endpoint_hashes"]),
        n_windows=_n_wins,
    )
    _result.rpm.block_until_ready()
    _mask, _, _ = detect_anomalies(_result.rpm, int(_ts[0]))
    _mask.block_until_ready()
    print("done")

    tracemalloc.start()
    t0 = time.perf_counter()

    parsed      = parse_file(path)
    sort_idx    = np.argsort(parsed["timestamps"])
    timestamps  = parsed["timestamps"][sort_idx]
    status      = parsed["status_codes"][sort_idx]
    sizes       = parsed["response_sizes"][sort_idx]
    ep_hashes   = parsed["endpoint_hashes"][sort_idx]

    n_wins = compute_n_windows(timestamps)
    result = compute_all_metrics(
        jnp.array(timestamps), jnp.array(status),
        jnp.array(sizes), jnp.array(ep_hashes),
        n_windows=n_wins,
    )
    result.rpm.block_until_ready()

    mask, atimes, sev = detect_anomalies(result.rpm, int(timestamps[0]))
    mask.block_until_ready()

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_lines = len(timestamps)
    file_mb = Path(path).stat().st_size / 1024 ** 2
    peak_mb = peak / 1024 ** 2

    device = jax.default_backend()
    return {
        "engine":        f"numba+jax({device})",
        "file_mb":       file_mb,
        "n_lines":       n_lines,
        "elapsed_sec":   elapsed,
        "lines_per_sec": n_lines / elapsed,
        "peak_mem_mb":   peak_mb,
    }


def _print_table(results: list[dict]) -> None:
    col = "{:<12} {:>8} {:>12} {:>10} {:>12} {:>10}"
    sep = "-" * 68
    print("\n" + sep)
    print(col.format("Engine", "File(MB)", "Lines", "Time(s)", "Lines/s", "Mem(MB)"))
    print(sep)
    for r in results:
        print(col.format(
            r["engine"],
            f"{r['file_mb']:.0f}",
            f"{r['n_lines']:,}",
            f"{r['elapsed_sec']:.2f}",
            f"{r['lines_per_sec']:,.0f}",
            f"{r['peak_mem_mb']:.0f}",
        ))
    print(sep)

    # Speed-up rows grouped by file size
    by_file: dict[float, dict] = {}
    for r in results:
        by_file.setdefault(r["file_mb"], {})[r["engine"]] = r

    print("\nSpeed-up vs pandas baseline:")
    for file_mb, engines in sorted(by_file.items()):
        base = engines.get("pandas")
        if not base:
            continue
        for eng, r in engines.items():
            if eng == "pandas":
                continue
            speedup = r["lines_per_sec"] / base["lines_per_sec"]
            print(f"  {file_mb:.0f} MB  {eng:<12} {speedup:.1f}×")
    print()


def _save_results(results: list[dict], speedups: list[dict], out_path: str) -> None:
    payload = {
        "run_at":   datetime.now(tz=timezone.utc).isoformat(),
        "jax_backend": jax.default_backend(),
        "results":  results,
        "speedups": speedups,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nResults saved → {out_path}")


def _compute_speedups(results: list[dict]) -> list[dict]:
    by_file: dict[float, dict] = {}
    for r in results:
        by_file.setdefault(round(r["file_mb"]), {})[r["engine"]] = r

    speedups = []
    for file_mb, engines in sorted(by_file.items()):
        base = engines.get("pandas")
        if not base:
            continue
        for eng, r in engines.items():
            if eng == "pandas":
                continue
            speedups.append({
                "file_mb":   file_mb,
                "engine":    eng,
                "speedup_x": round(r["lines_per_sec"] / base["lines_per_sec"], 2),
                "baseline_lines_per_sec": round(base["lines_per_sec"]),
                "engine_lines_per_sec":   round(r["lines_per_sec"]),
            })
    return speedups


def main(files: list[str], out: str) -> None:
    device = jax.default_backend()
    print(f"\nJAX backend: {device.upper()}")
    all_results: list[dict] = []

    for path in files:
        if not Path(path).exists():
            print(f"  [skip] {path} not found — run generate_logs.py first")
            continue

        print(f"\n{'='*60}")
        print(f"  Benchmarking: {path}")
        print(f"{'='*60}")

        print("\n-- pandas --")
        all_results.append(bench_pandas(path))

        print("\n-- numba (parse only) --")
        all_results.append(bench_numba(path))

        print("\n-- full pipeline (numba + jax) --")
        all_results.append(_bench_full_pipeline(path))

    if all_results:
        speedups = _compute_speedups(all_results)
        _print_table(all_results)
        _save_results(all_results, speedups, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the full benchmark comparison")
    ap.add_argument(
        "--files", nargs="+",
        default=["data/1gb.log", "data/5gb.log", "data/10gb.log"],
        help="Log files to benchmark",
    )
    ap.add_argument(
        "--out", default="results/benchmark.json",
        help="Path to save results JSON (default: results/benchmark.json)",
    )
    args = ap.parse_args()
    main(args.files, args.out)
