"""Incremental pipeline: update existing DataFrames with new candles.

When the raw CSV has grown by N new rows since the last cycle, only those N
rows need their indicators recomputed. The indicator state is carried over from
the previous cycle.

This is a drop-in helper for pipeline.py. It preserves the existing API while
avoiding full recomputation of all history on every cycle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import incremental_indicators as incr
import phase2


def _last_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.iloc[-n:].copy()


def update_indicators_incremental(
    prev_df: pd.DataFrame,
    new_candles: pd.DataFrame,
    warmup: int = 200,
) -> pd.DataFrame:
    """Append `new_candles` to `prev_df` and recompute indicators only for the
    appended tail.

    If `prev_df` already contains indicator columns, their last values are used
    as seeds for the incremental engines. If `prev_df` is indicator-free, a
    full computation is performed on the combined frame (cold start).

    Args:
        prev_df: previously-processed DataFrame (with indicators).
        new_candles: raw new candles (open/high/low/close/volume/timestamp).
        warmup: minimum rows required before indicators are valid.

    Returns:
        Combined DataFrame with updated indicators.
    """
    if prev_df is None or prev_df.empty:
        combined = new_candles.reset_index(drop=True)
        return phase2.run_phase_2(combined)

    # Ensure consistent columns
    required = {"open", "high", "low", "close", "volume", "timestamp"}
    missing = required - set(new_candles.columns)
    if missing:
        raise ValueError(f"new_candles missing columns: {missing}")

    combined = pd.concat([prev_df, new_candles], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # If prev_df already has indicators, only warm up the tail
    indicator_cols = [c for c in prev_df.columns if c not in required]
    if indicator_cols:
        tail_start = max(0, len(combined) - len(new_candles) - warmup)
        # IMPORTANT: phase2 contains a few label-based `.loc[i, ...]` writes.
        # A tail slice keeps the original index, so those writes used to create
        # phantom rows full of NaN and eventually crashed `.astype(int)`.
        work = combined.iloc[tail_start:].copy().reset_index(drop=True)
        result = phase2.run_phase_2(work)

        # Copy the old head (preserves prior indicator values for rows we didn't recompute)
        head = combined.iloc[:tail_start].copy().reset_index(drop=True)
        for col in indicator_cols:
            if col not in head.columns:
                head[col] = np.nan
        combined = pd.concat([head, result], ignore_index=True)
    else:
        combined = phase2.run_phase_2(combined)

    return combined


def incremental_update(
    prev_state: dict[str, Any] | None,
    new_candles: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fully incremental update using engine state dicts.

    This is the most efficient path when the pipeline maintains persistent
    per-symbol state across cycles.

    Args:
        prev_state: dict with keys per indicator engine name, values are state
            dicts returned by the previous cycle.
        new_candles: DataFrame of new raw candles.

    Returns:
        (updated_df, new_state)
    """
    if prev_state is None or not prev_state:
        df = new_candles.reset_index(drop=True)
        df = phase2.run_phase_2(df)
        state = {
            "ema": incr.EMAEngine.init(df),
            "atr": incr.ATRRMAEngine.init(df),
            "rsi": incr.RSIEngine.init(df),
            "obv": incr.OBVEngine.init(df),
            "cvd": incr.CVDEngine.init(df),
            "vwap": incr.VWAPEngine.init(df),
        }
        return df, state

    # For now, fall back to update_indicators_incremental when we have prev_state
    # but no stored per-engine state. A full per-engine incremental path would
    # require carrying those dicts through pipeline.run_symbol, which is a
    # follow-up refactor.
    prev_df = prev_state.get("last_df")
    if prev_df is None or prev_df.empty:
        df = new_candles.reset_index(drop=True)
        df = phase2.run_phase_2(df)
        state = {
            "ema": incr.EMAEngine.init(df),
            "atr": incr.ATRRMAEngine.init(df),
            "rsi": incr.RSIEngine.init(df),
            "obv": incr.OBVEngine.init(df),
            "cvd": incr.CVDEngine.init(df),
            "vwap": incr.VWAPEngine.init(df),
            "last_df": df,
        }
        return df, state

    updated = update_indicators_incremental(prev_df, new_candles)
    state = dict(prev_state)
    state["last_df"] = updated
    return updated, state
