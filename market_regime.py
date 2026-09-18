import pandas as pd
import numpy as np


def classify_market_regime(df):
    """Classify market regime and directional bias from the latest closed candle.

    Regime rules (no lookahead, non-repainting):
      - CHOPPY  : ADX < 20  OR  current Bollinger Band Width < rolling 20-candle mean BB width.
      - TRENDING: otherwise.

    Bias rules:
      - BULLISH  : close > VWAP  AND  close > EMA200
      - BEARISH  : close < VWAP  AND  close < EMA200
      - NEUTRAL  : mixed / missing data

    Args:
        df (pd.DataFrame): Must contain at least 20 rows with columns
            ``ADX``, ``BB_WIDTH``, ``VWAP``, ``EMA200``, ``close``.

    Returns:
        dict with keys:
            regime  : 'CHOPPY' | 'TRENDING'
            bias    : 'BULLISH' | 'BEARISH' | 'NEUTRAL'
            adx     : float (latest ADX)
            bb_width: float (latest BB_WIDTH)
            bb_width_avg_20: float (20-bar mean)
            reason  : str
    """
    out = {
        "regime": "CHOPPY",
        "bias": "NEUTRAL",
        "adx": np.nan,
        "bb_width": np.nan,
        "bb_width_avg_20": np.nan,
        "reason": "insufficient_data",
    }

    if df is None or len(df) < 20:
        return out

    adx = pd.to_numeric(df.get("ADX", pd.Series(dtype=float)), errors="coerce")
    bb_width = pd.to_numeric(df.get("BB_WIDTH", pd.Series(dtype=float)), errors="coerce")
    close = pd.to_numeric(df.get("close", pd.Series(dtype=float)), errors="coerce")
    vwap = pd.to_numeric(df.get("VWAP", pd.Series(dtype=float)), errors="coerce")
    ema200 = pd.to_numeric(df.get("EMA200", pd.Series(dtype=float)), errors="coerce")

    cur_adx = adx.iloc[-1]
    cur_bb = bb_width.iloc[-1]
    bb_avg = bb_width.tail(20).mean()

    out["adx"] = float(cur_adx) if pd.notna(cur_adx) else np.nan
    out["bb_width"] = float(cur_bb) if pd.notna(cur_bb) else np.nan
    out["bb_width_avg_20"] = float(bb_avg) if pd.notna(bb_avg) else np.nan

    reasons = []
    is_choppy = False

    if pd.notna(cur_adx) and cur_adx < 20:
        is_choppy = True
        reasons.append(f"ADX {cur_adx:.1f} < 20")

    if pd.notna(cur_bb) and pd.notna(bb_avg) and bb_avg > 0 and cur_bb < bb_avg:
        is_choppy = True
        reasons.append(f"BB_WIDTH {cur_bb:.2f} < 20-bar avg {bb_avg:.2f}")

    out["regime"] = "CHOPPY" if is_choppy else "TRENDING"
    out["reason"] = ", ".join(reasons) if reasons else "trending conditions"

    cur_close = close.iloc[-1]
    cur_vwap = vwap.iloc[-1]
    cur_ema200 = ema200.iloc[-1]

    if pd.notna(cur_close) and pd.notna(cur_vwap) and pd.notna(cur_ema200):
        if cur_close > cur_vwap and cur_close > cur_ema200:
            out["bias"] = "BULLISH"
        elif cur_close < cur_vwap and cur_close < cur_ema200:
            out["bias"] = "BEARISH"
        else:
            out["bias"] = "NEUTRAL"

    return out
