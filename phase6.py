import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

def calculate_volume_delta_and_cvd(df):
    df = df.copy()
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
    candle_range = (high - low).replace(0, 1e-8)
    buy_ratio = (close - low) / candle_range
    sell_ratio = (high - close) / candle_range
    buy_vol = vol * buy_ratio
    sell_vol = vol * sell_ratio
    df["VOLUME_DELTA"] = buy_vol - sell_vol
    df["CVD"] = df["VOLUME_DELTA"].cumsum()
    df["CVD_EMA_20"] = df["CVD"].ewm(span=20, adjust=False).mean()
    return df

def calculate_obv_metrics(df):
    df = df.copy()
    close, vol = df["close"], df["volume"]
    direction = np.sign(close.diff())
    df["OBV"] = (direction * vol).cumsum()
    df["OBV_EMA_20"] = df["OBV"].ewm(span=20, adjust=False).mean()
    df["OBV_STATE"] = np.where(df["OBV"] > df["OBV_EMA_20"], "BULLISH", "BEARISH")
    return df

def calculate_vwap_metrics(df):
    df = df.copy()
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
    typical = (high + low + close) / 3.0
    tp_vol = typical * vol
    if "timestamp" in df.columns:
        session = pd.to_datetime(df["timestamp"], errors="coerce").dt.floor("D")
    else:
        session = pd.Series(0, index=df.index)
    cum_tp = tp_vol.groupby(session).cumsum()
    cum_vol = vol.groupby(session).cumsum().replace(0, np.nan)
    df["VWAP"] = cum_tp / cum_vol
    df["VWAP_DISTANCE"] = ((close - df["VWAP"]) / df["VWAP"]) * 100.0
    return df

def calculate_relative_volume(df, window=20):
    df = df.copy()
    vol = df["volume"]
    vol_ma = vol.rolling(window, min_periods=window).mean().replace(0, 1e-8)
    df["RVOL"] = vol / vol_ma
    df["VOLUME_SPIKE"] = (df["RVOL"] >= 2.0).astype(int)
    return df

def calculate_volume_profile(df, lookback=120, bins=24):
    df = df.copy()
    n = len(df)
    if n < lookback:
        lookback = max(10, n)

    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)

    high_v = df["high"].values.astype(float)
    low_v = df["low"].values.astype(float)
    close_v = df["close"].values.astype(float)
    vol_v = df["volume"].values.astype(float)

    # Pre-allocate rolling histogram
    hist = np.zeros(bins, dtype=float)
    window_prices = np.zeros(lookback, dtype=float)
    window_vols = np.zeros(lookback, dtype=float)

    def _recompute(i):
        start = max(0, i - lookback + 1)
        hi = high_v[start : i + 1].max()
        lo = low_v[start : i + 1].min()
        if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
            return
        edges = np.linspace(lo, hi, bins + 1)
        prices = close_v[start : i + 1]
        vols = vol_v[start : i + 1]
        idxs = np.clip(np.digitize(prices, edges) - 1, 0, bins - 1)
        hist.fill(0.0)
        np.add.at(hist, idxs, vols)
        total = hist.sum()
        if total <= 0:
            return
        centers = (edges[:-1] + edges[1:]) / 2.0
        poc[i] = centers[hist.argmax()]
        order = np.argsort(hist)[::-1]
        cum = 0.0
        chosen = []
        for k in order:
            cum += hist[k]
            chosen.append(k)
            if cum >= total * 0.70:
                break
        vah[i] = centers[max(chosen)]
        val[i] = centers[min(chosen)]

    # Initialize first window
    _recompute(lookback - 1)

    # Slide forward: only recompute when price range changes significantly
    last_hi = high_v[:lookback].max()
    last_lo = low_v[:lookback].min()
    last_range = last_hi - last_lo

    for i in range(lookback, n):
        start = max(0, i - lookback + 1)
        cur_hi = high_v[start : i + 1].max()
        cur_lo = low_v[start : i + 1].min()
        cur_range = cur_hi - cur_lo

        # Recompute only when range expands/contracts by >20% or new high/low
        if (
            cur_range > last_range * 1.2
            or cur_range < last_range * 0.8
            or cur_hi > last_hi
            or cur_lo < last_lo
        ):
            _recompute(i)
            last_hi = cur_hi
            last_lo = cur_lo
            last_range = cur_range
        else:
            # Carry forward previous values (profile hasn't meaningfully changed)
            poc[i] = poc[i - 1]
            vah[i] = vah[i - 1]
            val[i] = val[i - 1]

    df["VP_POC"] = poc
    df["VP_VAH"] = vah
    df["VP_VAL"] = val
    df["VP_POC_DISTANCE_PCT"] = (df["close"] - df["VP_POC"]) / df["VP_POC"].replace(0, np.nan) * 100
    df["VP_ZONE"] = np.select(
        [df["close"] > df["VP_VAH"], df["close"] < df["VP_VAL"]],
        ["ABOVE_VALUE", "BELOW_VALUE"], default="IN_VALUE"
    )
    return df

def calculate_cvd_divergence(df, window=5, lookback=60):
    df = df.copy()
    n = len(df)
    df["CVD_DIV_BULL"] = np.zeros(n, dtype=np.int8)
    df["CVD_DIV_BEAR"] = np.zeros(n, dtype=np.int8)

    span = window * 2 + 1
    low_piv = (df["low"] == df["low"].rolling(span, center=False, min_periods=span).min()).fillna(False).values
    high_piv = (df["high"] == df["high"].rolling(span, center=False, min_periods=span).max()).fillna(False).values
    lows, highs, cvd = df["low"].values, df["high"].values, df["CVD"].values

    prev_low = prev_high = None
    for i in range(n):
        confirm = i + window
        if confirm >= n:
            break
        if low_piv[i]:
            if prev_low is not None and 0 < i - prev_low <= lookback:
                if lows[i] < lows[prev_low] and cvd[i] > cvd[prev_low]:
                    df.loc[confirm, "CVD_DIV_BULL"] = 1
            prev_low = i
        if high_piv[i]:
            if prev_high is not None and 0 < i - prev_high <= lookback:
                if highs[i] > highs[prev_high] and cvd[i] < cvd[prev_high]:
                    df.loc[confirm, "CVD_DIV_BEAR"] = 1
            prev_high = i

    df["CVD_STATE"] = np.where(df["CVD"] > df["CVD_EMA_20"], "BULLISH", "BEARISH")
    df["VOLUME_CONFIRMS"] = (
        (df["CVD_STATE"] == "BULLISH") & (df["OBV_STATE"] == "BULLISH") & (df["VOLUME_DELTA"] > 0)
    ).astype(int) - (
        (df["CVD_STATE"] == "BEARISH") & (df["OBV_STATE"] == "BEARISH") & (df["VOLUME_DELTA"] < 0)
    ).astype(int)
    return df

def process_phase6(symbols=None):
    if symbols is None:
        from config import SYMBOLS
        symbols = SYMBOLS
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        for tf in ["5m","15m","1h","4h"]:
            inp = DATA_DIR / f"{symbol}_{tf}_phase5.csv"
            out = DATA_DIR / f"{symbol}_{tf}_phase6.csv"
            
            if not inp.exists():
                continue
            if out.exists() and out.stat().st_mtime > inp.stat().st_mtime:
                continue
            
            df = pd.read_csv(inp)
            df = calculate_volume_delta_and_cvd(df)
            df = calculate_obv_metrics(df)
            df = calculate_vwap_metrics(df)
            df = calculate_relative_volume(df)
            df = calculate_volume_profile(df)
            df = calculate_cvd_divergence(df)
            df.to_csv(out, index=False)