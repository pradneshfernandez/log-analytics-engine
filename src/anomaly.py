"""
JAX anomaly detector.

Uses a rolling z-score over the requests-per-minute time series to flag
traffic spikes.  ``jax.lax.scan`` is the right primitive here: it lets the
compiler unroll the sliding-window loop into a single XLA program with no
Python-level iteration.

Threshold z > 3 corresponds to roughly a 1-in-1000 probability under a
Gaussian, which keeps false-positive rates very low for normal log traffic.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# Platform selected via JAX_PLATFORM_NAME env var before import.


# ---------------------------------------------------------------------------
# Rolling z-score via lax.scan
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("window",))
def rolling_zscore(series: jnp.ndarray, *, window: int = 10) -> jnp.ndarray:
    """
    Compute a rolling z-score over *series*.

    Why lax.scan instead of a Python loop?  scan compiles the entire sequence
    into one XLA program, so there's no Python overhead per step.  The carry
    is a fixed-size ring buffer; XLA handles the index arithmetic efficiently.

    The first *window* values will have lower variance estimates (cold start),
    but that's acceptable because real anomalies appear well into the trace.
    """
    def step(carry: tuple, x: jnp.ndarray) -> tuple:
        buf, write_idx = carry
        buf    = buf.at[write_idx % window].set(x)
        mean   = jnp.mean(buf)
        std    = jnp.std(buf) + 1e-6          # avoid division by zero
        z      = (x - mean) / std
        return (buf, write_idx + 1), z

    init  = (jnp.zeros(window), jnp.int32(0))
    _, zs = jax.lax.scan(step, init, series)
    return zs


# ---------------------------------------------------------------------------
# Anomaly detection kernel
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("window", "threshold"))
def detect_anomalies(
    rpm:       jnp.ndarray,
    t_start:   int,
    *,
    window:    int   = 10,
    threshold: float = 3.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Flag time windows where the rolling z-score exceeds *threshold*.

    Parameters
    ----------
    rpm      : requests-per-minute array, shape [n_windows]
    t_start  : unix epoch of the first window
    window   : rolling window width (in windows, not seconds)
    threshold: z-score cutoff (default 3σ)

    Returns
    -------
    anomaly_mask      : bool array [n_windows], True where anomalous
    anomaly_timestamps: int64 unix seconds for each anomalous window
    severity_scores   : float32 z-scores for anomalous windows
    """
    zs           = rolling_zscore(rpm.astype(jnp.float32), window=window)
    anomaly_mask = jnp.abs(zs) > threshold

    # Timestamps: window i begins at t_start + i * 60 seconds
    window_times  = t_start + jnp.arange(len(rpm), dtype=jnp.int64) * 60
    anomaly_times = jnp.where(anomaly_mask, window_times, jnp.int64(-1))
    severity      = jnp.where(anomaly_mask, jnp.abs(zs), jnp.float32(0.0))

    return anomaly_mask, anomaly_times, severity


# ---------------------------------------------------------------------------
# Convenience: extract non-negative timestamps (drops masked-out -1 entries)
# ---------------------------------------------------------------------------

def extract_anomaly_events(
    anomaly_mask:       np.ndarray,
    anomaly_timestamps: np.ndarray,
    severity_scores:    np.ndarray,
) -> list[dict]:
    """
    Convert JAX arrays from detect_anomalies into a plain Python list of dicts.
    Called once after the JIT kernel; not on the hot path.
    """
    mask  = np.asarray(anomaly_mask)
    times = np.asarray(anomaly_timestamps)
    sev   = np.asarray(severity_scores)
    return [
        {"timestamp": int(times[i]), "severity": float(sev[i])}
        for i in np.where(mask)[0]
    ]
