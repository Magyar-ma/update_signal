import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# سرکوب هشدار بیضرر پراکندگی DataFrame (عملکرد را تحت تأثیر قرار نمیدهد)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

try:
    from config import SYMBOLS, TIMEFRAMES
except ImportError:
    SYMBOLS = ["BTC-SWAP-USDT"]
    TIMEFRAMES = ["5m", "15m", "1h", "4h"]

DATA_DIR = Path(__file__).resolve().parent / "data"
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

def rma(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def true_range(df):
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def calculate_atr(df, period=14):
    df["ATR"] = rma(true_range(df), period)
    return df

def calculate_ema(df):
    for p in [20,50,100,200]:
        df[f"EMA{p}"] = ema(df["close"], p)
    return df

def calculate_adx(df, period=14):
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr = rma(tr, period)
    plus_smoothed = rma(pd.Series(plus_dm, index=df.index), period)
    minus_smoothed = rma(pd.Series(minus_dm, index=df.index), period)
    plus_di = 100 * plus_smoothed / atr.replace(0, np.nan)
    minus_di = 100 * minus_smoothed / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    df["ADX"] = rma(dx, period)
    return df

def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50
    df["RSI"] = rsi
    return df

def calculate_macd(df):
    fast = ema(df["close"], 12)
    slow = ema(df["close"], 26)
    macd = fast - slow
    signal = ema(macd, 9)
    df["MACD"] = macd
    df["MACD_SIGNAL"] = signal
    df["MACD_HIST"] = macd - signal
    return df

def calculate_mfi(df, period=14):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical * df["volume"]
    direction = typical.diff()
    positive = money_flow.where(direction > 0, 0.0)
    negative = money_flow.where(direction < 0, 0.0)
    pos_sum = positive.rolling(period, min_periods=period).sum()
    neg_sum = negative.abs().rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - 100 / (1 + ratio)
    mfi[(neg_sum == 0) & (pos_sum > 0)] = 100
    mfi[(neg_sum == 0) & (pos_sum == 0)] = 50
    df["MFI"] = mfi
    return df

def calculate_supertrend(df, period=10, multiplier=3.0):
    atr = rma(true_range(df), period)
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = pd.Series(np.nan, index=df.index)
    final_lower = pd.Series(np.nan, index=df.index)
    direction = pd.Series(np.nan, index=df.index)
    supertrend = pd.Series(np.nan, index=df.index)

    valid = atr.notna()
    if valid.any():
        first = valid.idxmax()
        final_upper[first] = basic_upper[first]
        final_lower[first] = basic_lower[first]
        direction[first] = 1.0
        supertrend[first] = final_lower[first]

        prev = first
        for i in range(first+1, len(df)):
            if not valid[i]:
                continue
            if basic_upper[i] < final_upper[prev] or df["close"][prev] > final_upper[prev]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[prev]
            if basic_lower[i] > final_lower[prev] or df["close"][prev] < final_lower[prev]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[prev]

            if direction[prev] == -1:
                direction[i] = 1 if df["close"][i] > final_upper[i] else -1
            else:
                direction[i] = -1 if df["close"][i] < final_lower[i] else 1

            supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
            prev = i

    df["SUPERTREND"] = supertrend
    df["SUPERTREND_DIRECTION"] = direction
    return df

def calculate_qqe(df, rsi_period=14, smoothing=5, factor=4.236):
    if "RSI" in df.columns:
        rsi = df["RSI"]
    else:
        rsi = calculate_rsi(df, rsi_period)["RSI"]
    rsi_ma = ema(rsi, smoothing)
    rsi_atr = rma(rsi_ma.diff().abs(), (rsi_period * 2) - 1)
    dar = rsi_atr * factor

    long_band = pd.Series(np.nan, index=df.index)
    short_band = pd.Series(np.nan, index=df.index)
    trend = pd.Series(np.nan, index=df.index)

    valid = rsi_ma.notna() & dar.notna()
    if valid.any():
        first = valid.idxmax()
        long_band[first] = rsi_ma[first] - dar[first]
        short_band[first] = rsi_ma[first] + dar[first]
        trend[first] = 1.0
        prev = first
        for i in range(first+1, len(df)):
            if not valid[i]:
                continue
            new_long = rsi_ma[i] - dar[i]
            new_short = rsi_ma[i] + dar[i]
            if rsi_ma[prev] > long_band[prev]:
                long_band[i] = max(new_long, long_band[prev])
            else:
                long_band[i] = new_long
            if rsi_ma[prev] < short_band[prev]:
                short_band[i] = min(new_short, short_band[prev])
            else:
                short_band[i] = new_short
            if rsi_ma[i] > short_band[prev]:
                trend[i] = 1
            elif rsi_ma[i] < long_band[prev]:
                trend[i] = -1
            else:
                trend[i] = trend[prev]
            prev = i

    df["QQE_RSI"] = rsi
    df["QQE_RSI_MA"] = rsi_ma
    df["QQE_LONG_BAND"] = long_band
    df["QQE_SHORT_BAND"] = short_band
    df["QQE_TREND"] = trend
    return df

def calculate_wavetrend(df, channel_length=10, average_length=21, signal_length=4):
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    esa = ema(hlc3, channel_length)
    deviation = ema((hlc3 - esa).abs(), channel_length)
    denominator = 0.015 * deviation
    ci = np.where(denominator > 0, (hlc3 - esa) / denominator, 0.0)
    wt1 = ema(pd.Series(ci, index=df.index), average_length)
    wt2 = wt1.rolling(signal_length, min_periods=signal_length).mean()
    df["WT1"] = wt1
    df["WT2"] = wt2
    df["WT_HISTOGRAM"] = wt1 - wt2
    return df

def calculate_bollinger_keltner(df, period=20, bb_mult=2.0, kc_mult=1.5):
    close = df["close"]
    basis = close.rolling(period, min_periods=period).mean()
    dev = close.rolling(period, min_periods=period).std(ddof=0)
    df["BB_MID"] = basis
    df["BB_UPPER"] = basis + bb_mult * dev
    df["BB_LOWER"] = basis - bb_mult * dev
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / basis.replace(0, np.nan) * 100
    df["BB_PERCENT_B"] = (close - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"]).replace(0, np.nan)

    kc_atr = rma(true_range(df), period)
    kc_basis = ema(close, period)
    df["KC_UPPER"] = kc_basis + kc_mult * kc_atr
    df["KC_LOWER"] = kc_basis - kc_mult * kc_atr

    squeeze_on = (df["BB_UPPER"] < df["KC_UPPER"]) & (df["BB_LOWER"] > df["KC_LOWER"])
    df["SQUEEZE_ON"] = squeeze_on.fillna(False).astype(int)
    df["SQUEEZE_RELEASE"] = ((df["SQUEEZE_ON"].shift(1) == 1) & (df["SQUEEZE_ON"] == 0)).astype(int)
    width_pct = df["BB_WIDTH"].rolling(120, min_periods=30).rank(pct=True)
    df["BB_WIDTH_PERCENTILE"] = width_pct
    return df

def calculate_stoch_rsi(df, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi = df["RSI"] if "RSI" in df.columns else calculate_rsi(df, rsi_period)["RSI"]
    lowest = rsi.rolling(stoch_period, min_periods=stoch_period).min()
    highest = rsi.rolling(stoch_period, min_periods=stoch_period).max()
    stoch = 100 * (rsi - lowest) / (highest - lowest).replace(0, np.nan)
    df["STOCHRSI_K"] = stoch.rolling(k_smooth, min_periods=k_smooth).mean()
    df["STOCHRSI_D"] = df["STOCHRSI_K"].rolling(d_smooth, min_periods=d_smooth).mean()
    df["STOCHRSI_CROSS_UP"] = ((df["STOCHRSI_K"] > df["STOCHRSI_D"]) & (df["STOCHRSI_K"].shift(1) <= df["STOCHRSI_D"].shift(1)) & (df["STOCHRSI_K"] < 40)).astype(int)
    df["STOCHRSI_CROSS_DOWN"] = ((df["STOCHRSI_K"] < df["STOCHRSI_D"]) & (df["STOCHRSI_K"].shift(1) >= df["STOCHRSI_D"].shift(1)) & (df["STOCHRSI_K"] > 60)).astype(int)
    return df

def calculate_ichimoku(df, conv=9, base=26, span_b=52, displacement=26):
    high, low, close = df["high"], df["low"], df["close"]
    conversion = (high.rolling(conv, min_periods=conv).max() + low.rolling(conv, min_periods=conv).min()) / 2
    base_line = (high.rolling(base, min_periods=base).max() + low.rolling(base, min_periods=base).min()) / 2
    span_a = ((conversion + base_line) / 2).shift(displacement)
    span_b_line = ((high.rolling(span_b, min_periods=span_b).max() + low.rolling(span_b, min_periods=span_b).min()) / 2).shift(displacement)
    df["ICHI_CONVERSION"] = conversion
    df["ICHI_BASE"] = base_line
    df["ICHI_SPAN_A"] = span_a
    df["ICHI_SPAN_B"] = span_b_line
    cloud_top = pd.concat([span_a, span_b_line], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b_line], axis=1).min(axis=1)
    df["ICHI_CLOUD_TOP"] = cloud_top
    df["ICHI_CLOUD_BOTTOM"] = cloud_bottom
    df["ICHI_STATE"] = np.select(
        [(close > cloud_top) & (conversion > base_line),
         (close < cloud_bottom) & (conversion < base_line)],
        ["BULLISH", "BEARISH"], default="NEUTRAL"
    )
    return df

def calculate_cci(df, period=20):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical.rolling(period, min_periods=period).mean()
    mad = typical.rolling(period, min_periods=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["CCI"] = (typical - sma) / (0.015 * mad.replace(0, np.nan))
    return df

def calculate_williams_r(df, period=14):
    highest = df["high"].rolling(period, min_periods=period).max()
    lowest = df["low"].rolling(period, min_periods=period).min()
    df["WILLIAMS_R"] = -100 * (highest - df["close"]) / (highest - lowest).replace(0, np.nan)
    return df

def calculate_pivot_levels(df, period=50):
    high = df["high"].rolling(period, min_periods=period).max().shift(1)
    low = df["low"].rolling(period, min_periods=period).min().shift(1)
    close = df["close"].shift(1)
    pivot = (high + low + close) / 3
    df["PIVOT"] = pivot
    df["PIVOT_R1"] = 2 * pivot - low
    df["PIVOT_S1"] = 2 * pivot - high
    df["PIVOT_R2"] = pivot + (high - low)
    df["PIVOT_S2"] = pivot - (high - low)
    return df

def calculate_fibonacci_levels(df, lookback=100):
    swing_high = df["high"].rolling(lookback, min_periods=20).max()
    swing_low = df["low"].rolling(lookback, min_periods=20).min()
    rng = (swing_high - swing_low)
    df["FIB_HIGH"] = swing_high
    df["FIB_LOW"] = swing_low
    df["FIB_236"] = swing_high - rng * 0.236
    df["FIB_382"] = swing_high - rng * 0.382
    df["FIB_500"] = swing_high - rng * 0.500
    df["FIB_618"] = swing_high - rng * 0.618
    df["FIB_786"] = swing_high - rng * 0.786
    in_golden = (df["close"] >= df["FIB_786"]) & (df["close"] <= df["FIB_500"])
    df["IN_GOLDEN_POCKET"] = in_golden.fillna(False).astype(int)
    df["FIB_POSITION"] = ((df["close"] - swing_low) / rng.replace(0, np.nan)).clip(0, 1)
    return df

def calculate_atr_regime(df):
    atr = df["ATR"]
    atr_pct = atr / df["close"].replace(0, np.nan) * 100
    df["ATR_PERCENT"] = atr_pct
    rank = atr_pct.rolling(150, min_periods=40).rank(pct=True)
    df["VOLATILITY_REGIME"] = np.select(
        [rank >= 0.75, rank <= 0.25],
        ["HIGH", "LOW"], default="NORMAL"
    )
    adx = df.get("ADX", pd.Series(np.nan, index=df.index))
    df["MARKET_REGIME"] = np.select(
        [adx >= 25, adx < 18],
        ["TRENDING", "RANGING"], default="TRANSITIONAL"
    )
    return df

def _swing_pivots(series, window, mode):
    n = len(series)
    result = np.zeros(n, dtype=bool)
    for i in range(window, n):
        pivot = i - window
        left_start = max(0, pivot - window)
        right_end = min(n, i + 1)
        if mode == "high":
            if series.iloc[pivot] >= series.iloc[left_start:pivot].max() and series.iloc[pivot] >= series.iloc[pivot + 1:right_end].max():
                result[i] = True
        else:
            if series.iloc[pivot] <= series.iloc[left_start:pivot].min() and series.iloc[pivot] <= series.iloc[pivot + 1:right_end].min():
                result[i] = True
    return result

def calculate_divergences(df, window=5, lookback=60):
    n = len(df)
    for col in ["DIV_RSI_BULL", "DIV_RSI_BEAR", "DIV_MACD_BULL", "DIV_MACD_BEAR", "DIVERGENCE_SCORE"]:
        df[col] = np.zeros(n, dtype=np.int16)

    low_piv = _swing_pivots(df["low"], window, "low")
    high_piv = _swing_pivots(df["high"], window, "high")
    lows = df["low"].values
    highs = df["high"].values

    for name, ind_col in (("RSI", "RSI"), ("MACD", "MACD")):
        if ind_col not in df.columns:
            continue
        ind = df[ind_col].values
        prev_low_idx = None
        prev_high_idx = None
        for i in range(n):
            if low_piv[i]:
                actual_pivot = i - window
                if prev_low_idx is not None and 0 < actual_pivot - prev_low_idx <= lookback:
                    if lows[actual_pivot] < lows[prev_low_idx] and ind[actual_pivot] > ind[prev_low_idx]:
                        df.loc[i, f"DIV_{name}_BULL"] = 1
                prev_low_idx = actual_pivot
            if high_piv[i]:
                actual_pivot = i - window
                if prev_high_idx is not None and 0 < actual_pivot - prev_high_idx <= lookback:
                    if highs[actual_pivot] > highs[prev_high_idx] and ind[actual_pivot] < ind[prev_high_idx]:
                        df.loc[i, f"DIV_{name}_BEAR"] = 1
                prev_high_idx = actual_pivot

    bull = df["DIV_RSI_BULL"].rolling(5, min_periods=1).max() + df["DIV_MACD_BULL"].rolling(5, min_periods=1).max()
    bear = df["DIV_RSI_BEAR"].rolling(5, min_periods=1).max() + df["DIV_MACD_BEAR"].rolling(5, min_periods=1).max()
    # Rolling operations may produce NaN on malformed/partial tail frames.
    # Never let an indicator-quality issue crash the whole signal engine.
    df["DIVERGENCE_SCORE"] = (bull.fillna(0) - bear.fillna(0)).round().astype(np.int16)
    return df

def calculate_trend_state(df):
    bullish = (df["EMA20"] > df["EMA50"]) & (df["EMA50"] > df["EMA100"]) & (df["EMA100"] > df["EMA200"]) & (df["SUPERTREND_DIRECTION"] == 1)
    bearish = (df["EMA20"] < df["EMA50"]) & (df["EMA50"] < df["EMA100"]) & (df["EMA100"] < df["EMA200"]) & (df["SUPERTREND_DIRECTION"] == -1)
    df["TREND_STATE"] = np.select([bullish, bearish], ["BULLISH", "BEARISH"], default="NEUTRAL")
    return df

def calculate_momentum_state(df):
    bullish = (df["RSI"] > 50) & (df["MACD"] > df["MACD_SIGNAL"]) & (df["MFI"] > 50) & (df["QQE_TREND"] == 1) & (df["WT1"] > df["WT2"])
    bearish = (df["RSI"] < 50) & (df["MACD"] < df["MACD_SIGNAL"]) & (df["MFI"] < 50) & (df["QQE_TREND"] == -1) & (df["WT1"] < df["WT2"])
    df["MOMENTUM_STATE"] = np.select([bullish, bearish], ["BULLISH", "BEARISH"], default="NEUTRAL")
    return df

def calculate_confluence_state(df):
    votes = pd.Series(0, index=df.index, dtype=int)
    votes += np.where(df["TREND_STATE"] == "BULLISH", 1, np.where(df["TREND_STATE"] == "BEARISH", -1, 0))
    votes += np.where(df["MOMENTUM_STATE"] == "BULLISH", 1, np.where(df["MOMENTUM_STATE"] == "BEARISH", -1, 0))
    votes += np.where(df["ICHI_STATE"] == "BULLISH", 1, np.where(df["ICHI_STATE"] == "BEARISH", -1, 0))
    votes += np.where(df["close"] > df["EMA200"], 1, -1)
    votes += np.where(df["ADX"] >= 25, np.where(df["PLUS_DI"] > df["MINUS_DI"], 1, -1), 0)
    votes += np.sign(df["DIVERGENCE_SCORE"].fillna(0)).astype(int)
    df["INDICATOR_VOTES"] = votes
    df["CONFLUENCE_STATE"] = np.select(
        [votes >= 4, votes >= 2, votes <= -4, votes <= -2],
        ["STRONG_BULLISH", "BULLISH", "STRONG_BEARISH", "BEARISH"], default="NEUTRAL"
    )
    return df

def calculate_linear_regression_channel(df, window=100):
    df = df.copy()
    close = df["close"].values
    n = len(df)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    for i in range(window, n):
        y = close[i - window:i]
        ym = y.mean()
        slope = ((x - xm) * (y - ym)).sum() / denom if denom > 0 else 0.0
        intercept = ym - slope * xm
        std = y.std(ddof=0)
        fitted = intercept + slope * (window - 1)
        mid[i] = fitted
        upper[i] = fitted + 2 * std
        lower[i] = fitted - 2 * std
    df["LRC_MID"] = mid
    df["LRC_UPPER"] = upper
    df["LRC_LOWER"] = lower
    df["LRC_SLOPE"] = (df["LRC_MID"].diff(5) / df["LRC_MID"].replace(0, np.nan)) * 100
    df["LRC_STATE"] = np.where(
        df["close"] > df["LRC_MID"], "BULLISH",
        np.where(df["close"] < df["LRC_MID"], "BEARISH", "NEUTRAL")
    )
    df["LRC_TOUCH_BAND"] = (
        (df["low"] <= df["LRC_LOWER"]) | (df["high"] >= df["LRC_UPPER"])
    ).astype(int)
    return df


def calculate_elder_ray(df, period=13):
    ema_close = ema(df["close"], period)
    bull = df["high"] - ema_close
    bear = df["low"] - ema_close
    df["ELDER_BULL"] = bull
    df["ELDER_BEAR"] = bear
    df["ELDER_STATE"] = np.where(bull > 0, "BULLISH", np.where(bear < 0, "BEARISH", "NEUTRAL"))
    return df


def calculate_keltner_channel(df, period=20, mult=2.0):
    atr = rma(true_range(df), period)
    basis = ema(df["close"], period)
    df["KC_CH_MID"] = basis
    df["KC_CH_UPPER"] = basis + mult * atr
    df["KC_CH_LOWER"] = basis - mult * atr
    df["KC_CH_BREAK_UP"] = (df["close"] > df["KC_CH_UPPER"]).astype(int)
    df["KC_CH_BREAK_DOWN"] = (df["close"] < df["KC_CH_LOWER"]).astype(int)
    return df


def calculate_vwap_bands(df, mult=1.5):
    if "VWAP" not in df.columns:
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].replace(0, np.nan)
        df["VWAP"] = (typical * vol).cumsum() / vol.cumsum()
    window = min(200, len(df))
    if window > 5:
        dev = df["close"].rolling(window, min_periods=window).std(ddof=0)
    else:
        dev = pd.Series(0.0, index=df.index)
    df["VWAP_BAND_UP"] = df["VWAP"] + mult * dev
    df["VWAP_BAND_DOWN"] = df["VWAP"] - mult * dev
    df["VWAP_STATE"] = np.where(df["close"] > df["VWAP_BAND_UP"], "ABOVE",
                          np.where(df["close"] < df["VWAP_BAND_DOWN"], "BELOW", "INSIDE"))
    return df


def tag_session_killzone(df):
    if "timestamp" not in df.columns:
        df["SESSION"] = "UNKNOWN"
        df["KILLZONE"] = "NONE"
        df["IS_NY_SESSION"] = 0
        return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    hour = ts.dt.hour
    # نشستهای اصلی فارکس/کریپتو (UTC)
    london = (hour >= 7) & (hour < 16)
    newyork = (hour >= 13) & (hour < 21)
    asia = (hour >= 0) & (hour < 7)
    session = np.where(london, "LONDON", np.where(newyork, "NEWYORK", np.where(asia, "ASIA", "OFF")))
    df["SESSION"] = session
    df["IS_NY_SESSION"] = newyork.astype(int)
    # کیلزونهای معاملاتی: بازگشایی لندن/نیویورک
    kill = np.where((hour >= 8) & (hour < 10), "LONDON_OPEN",
             np.where((hour >= 13) & (hour < 15), "NY_OPEN",
             np.where((hour >= 2) & (hour < 4), "ASIA_OPEN", "NONE")))
    df["KILLZONE"] = kill
    return df


def run_phase_2(df):
    df = calculate_ema(df)
    df = calculate_atr(df)
    df = calculate_adx(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_mfi(df)
    df = calculate_supertrend(df)
    df = calculate_qqe(df)
    df = calculate_wavetrend(df)
    df = calculate_bollinger_keltner(df)
    df = calculate_stoch_rsi(df)
    df = calculate_ichimoku(df)
    df = calculate_cci(df)
    df = calculate_williams_r(df)
    df = calculate_pivot_levels(df)
    df = calculate_fibonacci_levels(df)
    df = calculate_atr_regime(df)
    df = calculate_divergences(df)
    df = calculate_linear_regression_channel(df)
    df = calculate_elder_ray(df)
    df = calculate_keltner_channel(df)
    df = calculate_vwap_bands(df)
    df = tag_session_killzone(df)
    df = calculate_trend_state(df)
    df = calculate_momentum_state(df)
    df = calculate_confluence_state(df)
    return df

def process_file(symbol, timeframe, input_file, output_file):
    # اگر فایل ورودی وجود نداشته باشد، کاری نمی‌کنیم
    if not input_file.exists():
        return

    # 🚀 اگر فایل خروجی جدیدتر از فایل ورودی باشد، یعنی قبلاً پردازش شده → SKIP
    if output_file.exists() and output_file.stat().st_mtime > input_file.stat().st_mtime:
        return

    # در غیر این صورت، پردازش را انجام بده
    df = pd.read_csv(input_file)
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if len(df) < 100:
        return
    df = run_phase_2(df)
    df.to_csv(output_file, index=False)

def process_phase2(symbols=None):
    if symbols is None:
        from config import SYMBOLS
        symbols = SYMBOLS
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        for tf in TIMEFRAMES:
            inp = DATA_DIR / f"{symbol}_{tf}.csv"
            out = DATA_DIR / f"{symbol}_{tf}_phase2.csv"
            process_file(symbol, tf, inp, out)