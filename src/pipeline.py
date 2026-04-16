"""
Pipeline orchestrator.

Wires together parser → aggregator → anomaly detector and prints a summary
report.  This is the single entry point for production use.

The first call on a cold process will be slower than subsequent calls because
Numba and JAX JIT-compile their kernels on first invocation.  The benchmark
scripts account for this by discarding the first run.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.aggregator import compute_all_metrics, compute_n_windows, top_k_endpoints
from src.anomaly import detect_anomalies, extract_anomaly_events
from src.parser import parse_file


def run(filepath: str, *, window_seconds: int = 60, anomaly_window: int = 10,
        anomaly_threshold: float = 3.0, top_k: int = 10) -> dict:
    """
    Parse *filepath* and return a summary dict.  Also prints a human-readable
    report to stdout.

    Parameters
    ----------
    filepath          : path to an Apache CLF log file
    window_seconds    : aggregation bucket width (default 60 → req/minute)
    anomaly_window    : rolling window width for z-score (in buckets)
    anomaly_threshold : z-score cutoff for flagging anomalies
    top_k             : number of top endpoints to report
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(filepath)

    wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Parse
    # ------------------------------------------------------------------
    t0     = time.perf_counter()
    parsed = parse_file(filepath)
    parse_sec = time.perf_counter() - t0

    n_lines    = len(parsed["timestamps"])
    timestamps = parsed["timestamps"]
    sort_idx   = np.argsort(timestamps)
    timestamps = timestamps[sort_idx]

    status_codes    = parsed["status_codes"][sort_idx]
    response_sizes  = parsed["response_sizes"][sort_idx]
    endpoint_hashes = parsed["endpoint_hashes"][sort_idx]

    # ------------------------------------------------------------------
    # Step 2: Aggregate (single JAX pass)
    # ------------------------------------------------------------------
    t0       = time.perf_counter()
    n_wins   = compute_n_windows(timestamps, window_seconds)

    result   = compute_all_metrics(
        jnp.array(timestamps),
        jnp.array(status_codes),
        jnp.array(response_sizes),
        jnp.array(endpoint_hashes),
        window_seconds=window_seconds,
        n_windows=n_wins,
    )
    result.rpm.block_until_ready()     # ensure JIT is done before we measure
    agg_sec = time.perf_counter() - t0

    top_indices, top_counts = top_k_endpoints(result.top_ep_buckets, top_k=top_k)

    # ------------------------------------------------------------------
    # Step 3: Anomaly detection
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    t_start   = int(timestamps[0])
    mask, atimes, sev = detect_anomalies(
        result.rpm,
        t_start,
        window=anomaly_window,
        threshold=anomaly_threshold,
    )
    mask.block_until_ready()
    anomaly_sec = time.perf_counter() - t0

    events = extract_anomaly_events(
        np.asarray(mask), np.asarray(atimes), np.asarray(sev)
    )

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    total_sec  = time.perf_counter() - wall_start
    lines_sec  = n_lines / total_sec if total_sec > 0 else 0
    file_mb    = path.stat().st_size / 1024 ** 2
    rpm_arr    = np.asarray(result.rpm)
    err_arr    = np.asarray(result.error_rates)

    print("\n" + "=" * 60)
    print(f"  Log Analytics Pipeline — {path.name}")
    print("=" * 60)
    print(f"  File size          : {file_mb:>10.1f} MB")
    print(f"  Lines processed    : {n_lines:>10,}")
    print(f"  Total duration     : {total_sec:>10.3f} s")
    print(f"  Throughput         : {lines_sec:>10,.0f} lines/s")
    print(f"  └─ parse           : {parse_sec:>10.3f} s")
    print(f"  └─ aggregate (JAX) : {agg_sec:>10.3f} s")
    print(f"  └─ anomaly (JAX)   : {anomaly_sec:>10.3f} s")
    print()
    print(f"  Time range         : {datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat()}")
    print(f"  Peak RPM           : {int(rpm_arr.max()):>10,}")
    print(f"  Avg RPM            : {float(rpm_arr.mean()):>10.1f}")
    print(f"  Avg error rate     : {float(err_arr.mean())*100:>9.2f}%")
    print(f"  Latency p50/p95/p99: {int(result.p50):,} / {int(result.p95):,} / {int(result.p99):,} bytes")
    print()
    print(f"  Anomalies detected : {len(events):>10,}")
    if events:
        worst = max(events, key=lambda e: e["severity"])
        ts    = datetime.fromtimestamp(worst["timestamp"], tz=timezone.utc)
        print(f"  Worst anomaly      : {ts.isoformat()}  z={worst['severity']:.2f}")
    print()
    print(f"  Top {top_k} endpoint hash buckets by volume:")
    for rank, (idx, cnt) in enumerate(
        zip(np.asarray(top_indices).tolist(), np.asarray(top_counts).tolist()), 1
    ):
        print(f"    {rank:2d}. bucket={idx:5d}  requests={int(cnt):>8,}")
    print("=" * 60 + "\n")

    return {
        "n_lines":        n_lines,
        "total_sec":      total_sec,
        "lines_per_sec":  lines_sec,
        "anomaly_count":  len(events),
        "anomaly_events": events,
        "rpm":            rpm_arr,
        "error_rates":    err_arr,
        "p50":            int(result.p50),
        "p95":            int(result.p95),
        "p99":            int(result.p99),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the log analytics pipeline")
    ap.add_argument("filepath", help="Path to Apache CLF log file")
    ap.add_argument("--window-sec",  type=int,   default=60)
    ap.add_argument("--threshold",   type=float, default=3.0)
    args = ap.parse_args()
    run(args.filepath, window_seconds=args.window_sec, anomaly_threshold=args.threshold)
