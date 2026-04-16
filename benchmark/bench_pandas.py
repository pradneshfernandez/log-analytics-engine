"""
Pandas baseline benchmark.

Reads the log file with a regex-based approach (pandas read_csv doesn't map
cleanly to Apache CLF, so we use read_csv with a separator that requires a
post-processing groupby — the same workload a typical analyst script does).

This is the comparison baseline; it reflects real-world pandas usage, not a
strawman.
"""

from __future__ import annotations

import re
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd


_CLF_RE = re.compile(
    r'^(\S+) \S+ \S+ \[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} [+\-]\d{4})\] '
    r'"(\S+) (\S+) \S+" (\d+) (\d+)'
)


def _read_to_dataframe(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="ascii", errors="ignore") as fh:
        for line in fh:
            m = _CLF_RE.match(line)
            if m:
                ip, ts, method, endpoint, status, size = m.groups()
                rows.append((ip, ts, endpoint, int(status), int(size)))

    return pd.DataFrame(rows, columns=["ip", "timestamp", "endpoint", "status", "size"])


def run_benchmark(path: str) -> dict:
    tracemalloc.start()
    t0  = time.perf_counter()

    df  = _read_to_dataframe(path)

    # Typical analytics groupbys
    _rpm            = df.groupby("timestamp")["status"].count()
    _error_rate     = (df["status"] >= 400).mean()
    _p50, _p95, _p99= np.percentile(df["size"], [50, 95, 99])
    _top10          = df.groupby("endpoint")["status"].count().nlargest(10)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    file_mb  = Path(path).stat().st_size / 1024 ** 2
    n_lines  = len(df)
    peak_mb  = peak / 1024 ** 2

    return {
        "engine":       "pandas",
        "file_mb":      file_mb,
        "n_lines":      n_lines,
        "elapsed_sec":  elapsed,
        "lines_per_sec": n_lines / elapsed,
        "peak_mem_mb":  peak_mb,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.log"
    r    = run_benchmark(path)
    print(f"[pandas]  {r['n_lines']:>10,} lines  "
          f"{r['elapsed_sec']:.2f}s  "
          f"{r['lines_per_sec']:>10,.0f} lines/s  "
          f"peak {r['peak_mem_mb']:.0f} MB")
