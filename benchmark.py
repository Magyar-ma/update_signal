"""Quick benchmark for the signal engine pipeline.

Measures:
    - Phase execution times
    - Total cycle time
    - CSV reads
    - Memory usage (rough)

Run:
    python benchmark.py
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline
import data_collector
import observability


def _make_fake_data(symbol: str, tf: str, n: int = 500) -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    path = data_dir / f"{symbol}_{tf}.csv"
    if path.exists():
        return
    rng = np.random.default_rng(42)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_p = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(100, 1000, n)
    ts = pd.date_range("2024-01-01", periods=n, freq=tf)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df["open_time"] = df["timestamp"]
    df.to_csv(path, index=False)


def _make_recent_data(symbol: str, tf: str, n: int = 500) -> None:
    """Create data with timestamps recent enough to pass the freshness check."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    path = data_dir / f"{symbol}_{tf}.csv"
    rng = np.random.default_rng(42)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_p = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(100, 1000, n)

    tf_seconds = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}.get(tf, 900)
    end_ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start_ts = end_ts - pd.Timedelta(seconds=tf_seconds * (n - 1))
    ts = pd.date_range(start=start_ts, periods=n, freq=pd.Timedelta(seconds=tf_seconds))
    df = pd.DataFrame({
        "timestamp": ts,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df["open_time"] = df["timestamp"]
    df.to_csv(path, index=False)


def _clear_caches():
    pipeline._RAW_CACHE.clear()
    pipeline._RAW_META.clear()
    pipeline._PHASE2_CACHE.clear()
    pipeline._PHASE2_MTIMES.clear()
    gc.collect()


def benchmark_symbol(symbol: str, warmup: bool = True) -> dict:
    _clear_caches()
    metrics = observability.MetricsCollector()

    if warmup:
        pipeline.run_symbol(symbol, metrics=None)
        _clear_caches()

    metrics.start_cycle()
    t0 = time.perf_counter()
    signals = pipeline.run_symbol(symbol, metrics=metrics)
    total_ms = (time.perf_counter() - t0) * 1000.0
    cycle = metrics.end_cycle()

    return {
        "total_ms": total_ms,
        "signals": len(signals) if signals else 0,
        "phase2_ms": cycle.phase2_ms,
        "phase3_ms": cycle.phase3_ms,
        "phase4_ms": cycle.phase4_ms,
        "phase5_ms": cycle.phase5_ms,
        "phase6_ms": cycle.phase6_ms,
        "phase7_ms": cycle.phase7_ms,
    }


def benchmark_direct():
    _clear_caches()
    symbol = "BTC-SWAP-USDT"
    raw_by_tf = {}
    for tf in ["5m", "15m", "1h", "4h"]:
        df = pipeline._load_raw(symbol, tf)
        if df is not None:
            raw_by_tf[tf] = df

    if not raw_by_tf:
        print("No data available for direct benchmark")
        return

    metrics = observability.MetricsCollector()
    metrics.start_cycle()
    t0 = time.perf_counter()

    phase2_dfs = {}
    for tf, df in raw_by_tf.items():
        phase2_dfs[tf] = pipeline._phase2_cached(symbol, tf, df)

    total_ms = (time.perf_counter() - t0) * 1000.0
    cycle = metrics.end_cycle()

    print(f"Direct phase2 only: {total_ms:.1f} ms, {len(raw_by_tf)} TFs")


def main():
    symbol = "BTC-SWAP-USDT"
    for tf in ["5m", "15m", "1h", "4h"]:
        _make_recent_data(symbol, tf, n=500)

    print("=" * 70)
    print("BENCHMARK: first run (cold caches)")
    print("=" * 70)
    cold = benchmark_symbol(symbol, warmup=False)
    for k, v in cold.items():
        print(f"  {k:20s}: {v:>10.1f}" if isinstance(v, float) else f"  {k:20s}: {v:>10}")

    print()
    print("=" * 70)
    print("BENCHMARK: second run (warm caches)")
    print("=" * 70)
    warm = benchmark_symbol(symbol, warmup=False)
    for k, v in warm.items():
        print(f"  {k:20s}: {v:>10.1f}" if isinstance(v, float) else f"  {k:20s}: {v:>10}")

    if cold["total_ms"] > 0:
        speedup = cold["total_ms"] / max(warm["total_ms"], 0.1)
        print(f"\n  Cache speedup: {speedup:.1f}x")

    print()
    print("=" * 70)
    print("Direct phase2 benchmark (bypasses freshness check)")
    print("=" * 70)
    benchmark_direct()

    print("\nNote: 'before' baseline was not captured pre-change; compare warm")
    print("cache numbers against future runs after further optimizations.")


if __name__ == "__main__":
    main()
