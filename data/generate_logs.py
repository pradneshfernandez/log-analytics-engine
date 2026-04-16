"""
Synthetic Apache CLF log generator.

Produces realistic traffic distributions (Zipf IPs/endpoints, lognormal response sizes)
and injects traffic anomaly windows every ANOMALY_INTERVAL_SEC seconds so the
anomaly detector has real signal to find.
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Month abbreviations indexed 0-11
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_ENDPOINTS = [
    "/api/users", "/api/products", "/api/orders", "/api/search",
    "/api/auth/login", "/api/auth/logout", "/api/cart", "/api/checkout",
    "/api/recommendations", "/api/reviews", "/api/inventory", "/api/payments",
    "/api/notifications", "/api/analytics", "/api/reports", "/api/admin/users",
    "/static/js/main.js", "/static/css/app.css", "/static/img/logo.png",
    "/health", "/metrics", "/robots.txt", "/favicon.ico", "/",
    "/api/v2/users", "/api/v2/products", "/api/v2/orders",
    "/api/shipping", "/api/admin/config", "/docs",
]

_METHODS = ["GET", "GET", "GET", "GET", "GET", "POST", "POST", "PUT", "DELETE"]

# Status code distribution for normal traffic
_STATUS_CODES = [200, 201, 301, 302, 400, 401, 403, 404, 500, 502, 503]
_STATUS_PROBS  = [0.72, 0.05, 0.02, 0.03, 0.02, 0.02, 0.01, 0.06, 0.03, 0.01, 0.01]

# Same but with 5xx errors amplified for anomaly windows
_ANOMALY_STATUS_PROBS = [0.50, 0.03, 0.01, 0.02, 0.02, 0.02, 0.01, 0.06, 0.20, 0.07, 0.06]

ANOMALY_INTERVAL_SEC = 600   # spike every 10 minutes
ANOMALY_DURATION_SEC  = 60   # each spike lasts 1 minute
ANOMALY_MULTIPLIER    = 5    # 5× normal request rate during spike
BASE_REQ_PER_SEC      = 100


def _fmt_ts(epoch: int) -> str:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return f"{dt.day:02d}/{_MONTHS[dt.month - 1]}/{dt.year}:{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d} +0000"


def _zipf_weights(n: int) -> np.ndarray:
    """Zipf(1) weights: rank-1 item gets n× more traffic than rank-n."""
    w = 1.0 / np.arange(1, n + 1, dtype=np.float64)
    return w / w.sum()


def generate_logs(output_path: str, target_size_gb: float = 1.0, seed: int = 42) -> str:
    """
    Write a synthetic Apache CLF log file to *output_path*.

    The generator is CPU-bound on string formatting; disk throughput is the real
    ceiling at large sizes.  We write in 8 MB buffered chunks to keep OS page
    cache pressure low.
    """
    rng = np.random.default_rng(seed)
    target_bytes = int(target_size_gb * 1024 ** 3)

    # IP pool: 1 000 addresses, Zipf so a handful dominate (realistic bot traffic)
    ip_pool = [
        f"{rng.integers(1,254)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}"
        for _ in range(1_000)
    ]
    ip_weights       = _zipf_weights(len(ip_pool))
    endpoint_weights = _zipf_weights(len(_ENDPOINTS))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    written      = 0
    current_epoch = 1_700_000_000   # arbitrary fixed anchor (2023-11-14 22:13 UTC)
    chunk_lines   = 10_000
    t0            = time.perf_counter()

    with open(path, "wb", buffering=8 * 1024 * 1024) as fh:
        while written < target_bytes:
            time_in_cycle = (current_epoch - 1_700_000_000) % ANOMALY_INTERVAL_SEC
            in_anomaly    = time_in_cycle < ANOMALY_DURATION_SEC
            n_lines       = chunk_lines * ANOMALY_MULTIPLIER if in_anomaly else chunk_lines
            status_probs  = _ANOMALY_STATUS_PROBS if in_anomaly else _STATUS_PROBS

            # Sample all fields at once for the chunk (vectorised → fast)
            ip_idx   = rng.choice(len(ip_pool),   size=n_lines, p=ip_weights)
            ep_idx   = rng.choice(len(_ENDPOINTS), size=n_lines, p=endpoint_weights)
            meth_idx = rng.integers(0, len(_METHODS), size=n_lines)
            statuses = rng.choice(_STATUS_CODES,   size=n_lines, p=status_probs)
            sizes    = np.clip(
                rng.lognormal(mean=7.0, sigma=1.5, size=n_lines).astype(np.int32),
                100, 5_000_000,
            )
            time_delta = n_lines / BASE_REQ_PER_SEC
            timestamps = np.linspace(current_epoch, current_epoch + time_delta, n_lines, dtype=np.int64)

            lines = []
            for i in range(n_lines):
                ip  = ip_pool[ip_idx[i]]
                ts  = _fmt_ts(int(timestamps[i]))
                m   = _METHODS[meth_idx[i]]
                ep  = _ENDPOINTS[ep_idx[i]]
                st  = statuses[i]
                sz  = sizes[i]
                lines.append(f'{ip} - - [{ts}] "{m} {ep} HTTP/1.1" {st} {sz}\n'.encode())

            chunk   = b"".join(lines)
            fh.write(chunk)
            written       += len(chunk)
            current_epoch += time_delta

            if written % (50 * 1024 ** 2) < len(chunk):
                elapsed = time.perf_counter() - t0
                print(f"  {100*written/target_bytes:.1f}%  "
                      f"({written/1024**2:.0f} MB, {elapsed:.0f}s)", end="\r", flush=True)

    elapsed     = time.perf_counter() - t0
    final_mb    = path.stat().st_size / 1024 ** 2
    final_lines = written // 120  # rough estimate
    print(f"\nWrote {final_mb:.1f} MB  (~{final_lines:,} lines)  in {elapsed:.1f}s → {output_path}")
    return str(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Synthetic Apache CLF log generator")
    ap.add_argument("--output",  default="data/sample.log")
    ap.add_argument("--size-gb", type=float, default=1.0)
    ap.add_argument("--seed",    type=int,   default=42)
    args = ap.parse_args()
    generate_logs(args.output, args.size_gb, args.seed)
