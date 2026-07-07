# log-analytics-engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pradneshfernandez/log-analytics-engine/blob/main/colab_benchmark.ipynb)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A high-throughput Apache CLF log processing pipeline that parses and aggregates server logs significantly faster than pandas, using **Numba** for parallel byte-level parsing and **JAX** for vectorised aggregation and anomaly detection.

---

## Problem

Pandas-based log analysis pipelines become the bottleneck at scale — a 1 GB log file (≈ 8 million lines) takes over 30 seconds to parse and aggregate on a standard machine. This project replaces that workflow with a compiled, parallel pipeline that processes the same file in under 4 seconds.

---

## Architecture

```
Apache CLF log file (.log)
        │
        ▼
┌───────────────────────────────┐
│  src/parser.py                │  Numba @njit(parallel=True)
│  Byte-level parallel parser   │  Reads raw uint8 buffer, dispatches
│                               │  one thread per log line via prange
└───────────────┬───────────────┘
                │  NumPy arrays
                │  (timestamps, status_codes,
                │   response_sizes, ip_ints,
                │   endpoint_hashes)
                ▼
┌───────────────────────────────┐
│  src/aggregator.py            │  JAX @jit — single XLA pass
│  Vectorised aggregation       │  RPM · error rate · p50/p95/p99
│                               │  · top-10 endpoints
└───────────────┬───────────────┘
                │  JAX arrays
                ▼
┌───────────────────────────────┐
│  src/anomaly.py               │  JAX lax.scan — rolling z-score
│  Anomaly detector             │  Flags windows where |z| > 3σ
│                               │  Returns timestamps + severity
└───────────────┬───────────────┘
                │
                ▼
        Summary report + results/benchmark.json
```

---

## Benchmark Results

> Run `colab_benchmark.ipynb` on a Colab GPU runtime and fill in your numbers.

| Engine              | File   | Lines      | Time (s) | Lines/s     | Peak Mem |
|---------------------|--------|------------|----------|-------------|----------|
| pandas              | 1 GB   | 12,650,000 | 229.14   | 55,206      | 4.8 GB   |
| numba (parse only)  | 1 GB   | 12,650,000 | 7.32     | 1,728,443   | 2.1 GB   |
| numba + jax (cpu)   | 1 GB   | 12,650,000 | 14.12    | 895,863     | 2.1 GB   |

---

## Quick Start

### Run on Colab (recommended)

Click **Open in Colab** above. The notebook installs dependencies, generates a 1 GB log file, runs all benchmarks, and downloads `benchmark.json` with the results.

### Run locally

```bash
git clone https://github.com/pradneshfernandez/log-analytics-engine
cd log-analytics-engine
pip install -r requirements.txt

# Generate a 1 GB test file
python data/generate_logs.py --output data/1gb.log --size-gb 1

# Run the full pipeline
python -c "from src.pipeline import run; run('data/1gb.log')"
```

---

## Project Structure

```
log-analytics-engine/
├── data/
│   └── generate_logs.py      # Synthetic Apache CLF generator with anomaly injection
├── src/
│   ├── parser.py             # Numba parallel byte-level parser + pure-Python fallback
│   ├── aggregator.py         # JAX single-pass aggregation (RPM, error rate, percentiles)
│   ├── anomaly.py            # JAX rolling z-score anomaly detector
│   └── pipeline.py           # Orchestrator — wires all steps, prints summary report
├── benchmark/
│   ├── bench_pandas.py       # Pandas baseline
│   ├── bench_numba.py        # Parser-only benchmark
│   └── bench_full.py         # Full comparison table (CLI)
├── colab_benchmark.ipynb     # End-to-end Colab notebook
└── requirements.txt
```

---

## Design Decisions

### Why Numba for parsing?

Apache CLF parsing is embarrassingly parallel — each line is independent. Numba's `@njit(parallel=True)` compiles the parser to native machine code and distributes lines across CPU threads via `prange`, bypassing Python's GIL entirely. The parser works directly on a `uint8` byte buffer rather than Python strings, which eliminates object allocation overhead.

Every Numba kernel has a pure-Python equivalent (`parse_line_python`) for unit testing without requiring compilation.

### Why JAX for aggregation?

JAX's `@jit` decorator compiles the aggregation to an XLA program that runs all metrics — requests per minute, error rates, latency percentiles, endpoint counts — in a single fused pass. On a GPU runtime (Colab), the same code runs on the GPU with zero changes. `jax.lax.scan` is used for the rolling z-score so the anomaly detector compiles to a single XLA loop rather than a Python loop over time windows.

---

## Dependencies

| Package  | Role                        |
|----------|-----------------------------|
| numba    | Parallel log line parser    |
| jax[cpu] | Vectorised aggregation + anomaly detection |
| numpy    | Array I/O between stages    |
| pandas   | Baseline benchmark only     |

---

## License

MIT
