# ============================================================
# phase3.py - اصلاح کامل باگ‌های Look-Ahead و MTF Merge
# ============================================================
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"

SWING_LEFT = 3
SWING_RIGHT = 3
EQUAL_TOLERANCE = 0.0008
EQUAL_MIN_DISTANCE = 3
EQUAL_MAX_DISTANCE = 120
EQUAL_REACTION_BARS = 2
LIQUIDITY_MAX_AGE = 150
MIN_EVENT_DISTANCE = 3
DISPLACEMENT_ATR_MULTIPLIER = 1.0
DISPLACEMENT_BODY_RATIO = 0.60
PD_LOOKBACK = 100
SWEEP_MIN_WICK_RATIO = 0.25

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan

def relative_distance(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.inf
    return abs(float(a) - float(b)) / (abs(float(b)) + 1e-12)

def ensure_timestamp(df):
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        for col in ["open_time", "time", "datetime", "date"]:
            if col in df.columns:
                df["timestamp"] = pd.to_datetime(df[col], errors="coerce")
                break
        else:
            df["timestamp"] = pd.to_datetime(np.arange(len(df)), unit='s', origin='2026-01-01')
    return df

def sort_chronologically(df):
    return df.sort_values("timestamp").reset_index(drop=True)

def load_dataframe(path):
    df = pd.read_csv(path)
    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("Insufficient candles")
    df = ensure_timestamp(df)
    return sort_chronologically(df)

def true_range(df):
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def calculate_atr(df, period=14):
    tr = true_range(df)
    return tr.rolling(period, min_periods=period).mean()

# ============================================================
# توابع اصلاح شده بدون Look-Ahead Bias
# ============================================================
def detect_swings_numpy(highs, lows, swing_left=3, swing_right=3):
    n = len(highs)
    swing_high = np.zeros(n, dtype=np.int8)
    swing_low = np.zeros(n, dtype=np.int8)
    swing_high_price = np.full(n, np.nan, dtype=np.float64)
    swing_low_price = np.full(n, np.nan, dtype=np.float64)

    # اصلاح: سواینگ فقط پس از تایید شدن کامل روی کندل تایید ثبت می‌شود
    for i in range(swing_left + swing_right, n):
        pivot = i - swing_right
        if highs[pivot] >= highs[pivot-swing_left:pivot].max() and highs[pivot] >= highs[pivot+1:i+1].max():
            swing_high[i] = 1
            swing_high_price[i] = highs[pivot]
            
        if lows[pivot] <= lows[pivot-swing_left:pivot].min() and lows[pivot] <= lows[pivot+1:i+1].min():
            swing_low[i] = 1
            swing_low_price[i] = lows[pivot]

    return swing_high, swing_low, swing_high_price, swing_low_price

def detect_swings(df):
    highs = df["high"].values
    lows = df["low"].values
    sh, sl, shp, slp = detect_swings_numpy(highs, lows, SWING_LEFT, SWING_RIGHT)
    df["SWING_HIGH"] = sh
    df["SWING_LOW"] = sl
    df["SWING_HIGH_PRICE"] = shp
    df["SWING_LOW_PRICE"] = slp
    return df

def classify_structure_points(df):
    n = len(df)
    structure_point = ["" for _ in range(n)]
    last_high = np.nan
    last_low = np.nan
    for i in range(n):
        labels = []
        if df["SWING_HIGH"].iloc[i] == 1:
            price = df["SWING_HIGH_PRICE"].iloc[i]
            if pd.notna(price):
                if pd.notna(last_high):
                    labels.append("HH" if price > last_high else "LH")
                last_high = price
        if df["SWING_LOW"].iloc[i] == 1:
            price = df["SWING_LOW_PRICE"].iloc[i]
            if pd.notna(price):
                if pd.notna(last_low):
                    labels.append("HL" if price > last_low else "LL")
                last_low = price
        structure_point[i] = "|".join(labels)
    df["STRUCTURE_POINT"] = structure_point
    return df

def detect_equal_levels(df):
    n = len(df)
    equal_high = np.zeros(n, dtype=np.int8)
    equal_low = np.zeros(n, dtype=np.int8)
    equal_high_level = np.full(n, np.nan)
    equal_low_level = np.full(n, np.nan)
    confirmed_highs = []
    confirmed_lows = []

    for i in range(n):
        if df["SWING_HIGH"].iloc[i] == 1:
            price = safe_float(df["SWING_HIGH_PRICE"].iloc[i])
            if pd.notna(price):
                matched = None
                for prev_idx, prev_price in reversed(confirmed_highs):
                    dist = i - prev_idx
                    if dist < EQUAL_MIN_DISTANCE:
                        continue
                    if dist > EQUAL_MAX_DISTANCE:
                        break
                    if relative_distance(price, prev_price) <= EQUAL_TOLERANCE:
                        matched = (prev_idx, prev_price)
                        break
                if matched is not None:
                    prev_idx, prev_price = matched
                    level = (price + prev_price) / 2.0
                    # فقط سواینگ تاییدکننده (i) به‌عنوان EQUAL_HIGH علامت می‌خورد، نه
                    # سواینگ قدیمی‌تر (prev_idx). قبلاً هر دو علامت می‌خوردند که یعنی
                    # کندل قدیمی‌تر، در همان لحظه‌ی خودش، اطلاعاتی از آینده (تطبیق i)
                    # داشت — این دقیقاً همان Look-Ahead Bias‌ای است که این فایل ادعا
                    # می‌کند اصلاح شده؛ نتیجه‌اش بکتست‌هایی بود که کیفیت سطوح لیکوییدیتی
                    # را بهتر از چیزی که واقعاً در آن لحظه قابل‌دانستن بود گزارش می‌دادند.
                    equal_high[i] = 1
                    equal_high_level[i] = level
                confirmed_highs.append((i, price))

        if df["SWING_LOW"].iloc[i] == 1:
            price = safe_float(df["SWING_LOW_PRICE"].iloc[i])
            if pd.notna(price):
                matched = None
                for prev_idx, prev_price in reversed(confirmed_lows):
                    dist = i - prev_idx
                    if dist < EQUAL_MIN_DISTANCE:
                        continue
                    if dist > EQUAL_MAX_DISTANCE:
                        break
                    if relative_distance(price, prev_price) <= EQUAL_TOLERANCE:
                        matched = (prev_idx, prev_price)
                        break
                if matched is not None:
                    prev_idx, prev_price = matched
                    level = (price + prev_price) / 2.0
                    equal_low[i] = 1
                    equal_low_level[i] = level
                confirmed_lows.append((i, price))

    df["EQUAL_HIGH"] = equal_high
    df["EQUAL_LOW"] = equal_low
    df["EQUAL_HIGH_LEVEL"] = equal_high_level
    df["EQUAL_LOW_LEVEL"] = equal_low_level
    return df

def build_liquidity_pools(df):
    n = len(df)
    buy_side = np.full(n, np.nan)
    sell_side = np.full(n, np.nan)
    buy_type = ["" for _ in range(n)]
    sell_type = ["" for _ in range(n)]
    highs, lows = [], []

    for i in range(n):
        if df["SWING_HIGH"].iloc[i] == 1:
            price = safe_float(df["SWING_HIGH_PRICE"].iloc[i])
            if pd.notna(price):
                ltype = "EQUAL_HIGH" if df["EQUAL_HIGH"].iloc[i] == 1 else "SWING_HIGH"
                highs.append((i, price, ltype))
        if df["SWING_LOW"].iloc[i] == 1:
            price = safe_float(df["SWING_LOW_PRICE"].iloc[i])
            if pd.notna(price):
                ltype = "EQUAL_LOW" if df["EQUAL_LOW"].iloc[i] == 1 else "SWING_LOW"
                lows.append((i, price, ltype))

        close = df["close"].iloc[i]
        highs = [h for h in highs if i - h[0] <= LIQUIDITY_MAX_AGE]
        lows = [l for l in lows if i - l[0] <= LIQUIDITY_MAX_AGE]

        above = [h for h in highs if h[1] > close]
        below = [l for l in lows if l[1] < close]
        if above:
            sel = min(above, key=lambda x: x[1] - close)
            buy_side[i] = sel[1]
            buy_type[i] = sel[2]
        if below:
            sel = max(below, key=lambda x: x[1])
            sell_side[i] = sel[1]
            sell_type[i] = sel[2]

    df["BUY_SIDE_LIQUIDITY"] = buy_side
    df["BUY_SIDE_LIQUIDITY_TYPE"] = buy_type
    df["SELL_SIDE_LIQUIDITY"] = sell_side
    df["SELL_SIDE_LIQUIDITY_TYPE"] = sell_type
    return df

def detect_liquidity_sweeps(df):
    n = len(df)
    sweep = np.zeros(n, dtype=np.int8)
    direction = ["" for _ in range(n)]
    swept_level = np.full(n, np.nan)
    swept_type = ["" for _ in range(n)]
    sweep_scope = ["" for _ in range(n)]
    high_levels, low_levels = [], []
    consumed_highs, consumed_lows = set(), set()

    for i in range(n):
        if df["SWING_HIGH"].iloc[i] == 1:
            price = safe_float(df["SWING_HIGH_PRICE"].iloc[i])
            if pd.notna(price):
                ltype = "EQUAL_HIGH" if df["EQUAL_HIGH"].iloc[i] == 1 else "SWING_HIGH"
                high_levels.append((i, price, ltype))
        if df["SWING_LOW"].iloc[i] == 1:
            price = safe_float(df["SWING_LOW_PRICE"].iloc[i])
            if pd.notna(price):
                ltype = "EQUAL_LOW" if df["EQUAL_LOW"].iloc[i] == 1 else "SWING_LOW"
                low_levels.append((i, price, ltype))

        high, low = df["high"].iloc[i], df["low"].iloc[i]
        close, open_p = df["close"].iloc[i], df["open"].iloc[i]
        candle_range = high - low
        if candle_range <= 0:
            candle_range = 1e-8

        candidates = []
        for l_idx, l_price, ltype in reversed(high_levels):
            if l_idx >= i: continue
            if i - l_idx > LIQUIDITY_MAX_AGE: break
            key = (l_idx, round(l_price, 12))
            if key in consumed_highs: continue
            upper_wick = high - max(open_p, close)
            if upper_wick / candle_range >= SWEEP_MIN_WICK_RATIO and high > l_price and close < l_price:
                candidates.append((l_idx, l_price, ltype))
        if candidates:
            selected = candidates[0]
            sweep[i] = 1
            direction[i] = "BEARISH"
            swept_level[i] = selected[1]
            swept_type[i] = selected[2]
            sweep_scope[i] = "EXTERNAL" if selected[2] == "EQUAL_HIGH" else "INTERNAL"
            consumed_highs.add((selected[0], round(selected[1], 12)))

        candidates = []
        for l_idx, l_price, ltype in reversed(low_levels):
            if l_idx >= i: continue
            if i - l_idx > LIQUIDITY_MAX_AGE: break
            key = (l_idx, round(l_price, 12))
            if key in consumed_lows: continue
            lower_wick = min(open_p, close) - low
            if lower_wick / candle_range >= SWEEP_MIN_WICK_RATIO and low < l_price and close > l_price:
                candidates.append((l_idx, l_price, ltype))
        if candidates:
            selected = candidates[0]
            sweep[i] = 1
            direction[i] = "BULLISH"
            swept_level[i] = selected[1]
            swept_type[i] = selected[2]
            sweep_scope[i] = "EXTERNAL" if selected[2] == "EQUAL_LOW" else "INTERNAL"
            consumed_lows.add((selected[0], round(selected[1], 12)))

    df["LIQUIDITY_SWEEP"] = sweep
    df["LIQUIDITY_SWEEP_DIRECTION"] = direction
    df["LIQUIDITY_SWEPT_LEVEL"] = swept_level
    df["LIQUIDITY_SWEPT_TYPE"] = swept_type
    df["LIQUIDITY_SWEEP_SCOPE"] = sweep_scope
    return df

def detect_displacement(df):
    n = len(df)
    displacement = np.zeros(n, dtype=np.int8)
    direction = ["" for _ in range(n)]
    atr = calculate_atr(df)
    if "ATR" in df.columns:
        atr = df["ATR"].combine_first(atr)
    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    body_ratio = body / candle_range.replace(0, np.nan)
    for i in range(n):
        if pd.notna(atr.iloc[i]) and atr.iloc[i] > 0:
            if body_ratio.iloc[i] >= DISPLACEMENT_BODY_RATIO and body.iloc[i] >= atr.iloc[i] * DISPLACEMENT_ATR_MULTIPLIER:
                displacement[i] = 1
                direction[i] = "BULLISH" if df["close"].iloc[i] > df["open"].iloc[i] else "BEARISH"
    df["DISPLACEMENT"] = displacement
    df["DISPLACEMENT_DIRECTION"] = direction
    return df

def detect_structure_events(df):
    n = len(df)
    bos = np.zeros(n, dtype=np.int8)
    bos_dir = ["" for _ in range(n)]
    choch = np.zeros(n, dtype=np.int8)
    choch_dir = ["" for _ in range(n)]
    mss = np.zeros(n, dtype=np.int8)
    mss_dir = ["" for _ in range(n)]
    swing_highs, swing_lows = [], []
    bias = "NEUTRAL"
    last_bos = -999999
    last_choch = -999999
    last_mss = -999999
    recent_sweep_idx = -999999
    recent_sweep_dir = ""

    for i in range(n):
        if df["SWING_HIGH"].iloc[i] == 1:
            p = safe_float(df["SWING_HIGH_PRICE"].iloc[i])
            if pd.notna(p):
                swing_highs.append((i, p))
        if df["SWING_LOW"].iloc[i] == 1:
            p = safe_float(df["SWING_LOW_PRICE"].iloc[i])
            if pd.notna(p):
                swing_lows.append((i, p))

        if df["LIQUIDITY_SWEEP"].iloc[i] == 1:
            recent_sweep_idx = i
            recent_sweep_dir = df["LIQUIDITY_SWEEP_DIRECTION"].iloc[i]

        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i-1] if i > 0 else np.nan

        active_high = next((item for item in reversed(swing_highs) if item[0] < i), None)
        active_low = next((item for item in reversed(swing_lows) if item[0] < i), None)

        bullish_break = active_high is not None and pd.notna(prev_close) and prev_close <= active_high[1] and close > active_high[1]
        bearish_break = active_low is not None and pd.notna(prev_close) and prev_close >= active_low[1] and close < active_low[1]

        if bullish_break:
            if bias == "BEARISH":
                if i - last_choch >= MIN_EVENT_DISTANCE:
                    choch[i] = 1
                    choch_dir[i] = "BULLISH"
                    last_choch = i
                if (0 <= i - recent_sweep_idx <= 12 and recent_sweep_dir == "BEARISH") and (df["DISPLACEMENT"].iloc[i] == 1 and df["DISPLACEMENT_DIRECTION"].iloc[i] == "BULLISH") and i - last_mss >= MIN_EVENT_DISTANCE:
                    mss[i] = 1
                    mss_dir[i] = "BULLISH"
                    last_mss = i
            else:
                if i - last_bos >= MIN_EVENT_DISTANCE:
                    bos[i] = 1
                    bos_dir[i] = "BULLISH"
                    last_bos = i
            bias = "BULLISH"
        if bearish_break:
            if bias == "BULLISH":
                if i - last_choch >= MIN_EVENT_DISTANCE:
                    choch[i] = 1
                    choch_dir[i] = "BEARISH"
                    last_choch = i
                if (0 <= i - recent_sweep_idx <= 12 and recent_sweep_dir == "BULLISH") and (df["DISPLACEMENT"].iloc[i] == 1 and df["DISPLACEMENT_DIRECTION"].iloc[i] == "BEARISH") and i - last_mss >= MIN_EVENT_DISTANCE:
                    mss[i] = 1
                    mss_dir[i] = "BEARISH"
                    last_mss = i
            else:
                if i - last_bos >= MIN_EVENT_DISTANCE:
                    bos[i] = 1
                    bos_dir[i] = "BEARISH"
                    last_bos = i
            bias = "BEARISH"

    df["BOS"] = bos
    df["BOS_DIRECTION"] = bos_dir
    df["CHOCH"] = choch
    df["CHOCH_DIRECTION"] = choch_dir
    df["MSS"] = mss
    df["MSS_DIRECTION"] = mss_dir
    return df

def classify_liquidity(df):
    internal = ["" for _ in range(len(df))]
    external = ["" for _ in range(len(df))]
    for i in range(len(df)):
        bt = df["BUY_SIDE_LIQUIDITY_TYPE"].iloc[i]
        st = df["SELL_SIDE_LIQUIDITY_TYPE"].iloc[i]
        if bt == "EQUAL_HIGH":
            external[i] = "BUY_SIDE"
        elif bt == "SWING_HIGH":
            internal[i] = "BUY_SIDE"
        if st == "EQUAL_LOW":
            external[i] = "SELL_SIDE"
        elif st == "SWING_LOW":
            internal[i] = "SELL_SIDE"
    df["INTERNAL_LIQUIDITY"] = internal
    df["EXTERNAL_LIQUIDITY"] = external
    return df

def calculate_premium_discount(df):
    high = df["high"].rolling(PD_LOOKBACK, min_periods=1).max()
    low = df["low"].rolling(PD_LOOKBACK, min_periods=1).min()
    eq = (high + low) / 2
    zone = np.where(df["close"] > eq, "PREMIUM", np.where(df["close"] < eq, "DISCOUNT", "EQUILIBRIUM"))
    df["PD_HIGH"] = high
    df["PD_LOW"] = low
    df["EQUILIBRIUM"] = eq
    df["PREMIUM_DISCOUNT"] = zone
    return df

def bars_since_event(series):
    result = np.full(len(series), np.nan)
    last = None
    for i, val in enumerate(series):
        if val == 1:
            last = i
            result[i] = 0
        elif last is not None:
            result[i] = i - last
    return result

def add_bars_since(df):
    df["BARS_SINCE_BOS"] = bars_since_event(df["BOS"].values)
    df["BARS_SINCE_CHOCH"] = bars_since_event(df["CHOCH"].values)
    df["BARS_SINCE_MSS"] = bars_since_event(df["MSS"].values)
    df["BARS_SINCE_SWEEP"] = bars_since_event(df["LIQUIDITY_SWEEP"].values)
    return df

def rebuild_market_structure(df):
    state = []
    current = "NEUTRAL"
    for i in range(len(df)):
        if df["CHOCH"].iloc[i] == 1:
            current = df["CHOCH_DIRECTION"].iloc[i]
        elif df["BOS"].iloc[i] == 1:
            current = df["BOS_DIRECTION"].iloc[i]
        elif df["MSS"].iloc[i] == 1:
            current = df["MSS_DIRECTION"].iloc[i]
        state.append(current)
    df["MARKET_STRUCTURE"] = state
    return df

def calculate_structure_score_and_signal(df):
    scores = []
    for i in range(len(df)):
        score = 0
        struct = df["MARKET_STRUCTURE"].iloc[i]
        if struct == "BULLISH":
            score += 2
        elif struct == "BEARISH":
            score -= 2
        if df["BOS"].iloc[i] == 1:
            score += 3 if df["BOS_DIRECTION"].iloc[i] == "BULLISH" else -3
        if df["CHOCH"].iloc[i] == 1:
            score += 2 if df["CHOCH_DIRECTION"].iloc[i] == "BULLISH" else -2
        if df["MSS"].iloc[i] == 1:
            score += 5 if df["MSS_DIRECTION"].iloc[i] == "BULLISH" else -5
        if df["LIQUIDITY_SWEEP"].iloc[i] == 1:
            score += 2 if df["LIQUIDITY_SWEEP_DIRECTION"].iloc[i] == "BULLISH" else -2
        scores.append(score)
    df["STRUCTURE_SCORE"] = scores
    df["STRUCTURE_SIGNAL"] = np.select(
        [df["STRUCTURE_SCORE"] >= 7, df["STRUCTURE_SCORE"] >= 3, df["STRUCTURE_SCORE"] <= -7, df["STRUCTURE_SCORE"] <= -3],
        ["STRONG_BULLISH", "BULLISH", "STRONG_BEARISH", "BEARISH"],
        default="NEUTRAL"
    )
    return df

# ============================================================
# اصلاح ایمن ادغام MTF Context
# ============================================================
def add_mtf_context_for_symbol(symbol, dataframes):
    tf_minutes = {"1h": 60, "4h": 240}

    def closed_htf_state(target_df, source_df, source_tf):
        if source_df is None or source_df.empty or "timestamp" not in target_df.columns:
            return np.full(len(target_df), "UNKNOWN", dtype=object)
        left = target_df[["timestamp"]].copy()
        left["_row_order"] = np.arange(len(left))
        left["_lookup"] = pd.to_datetime(left["timestamp"]) - pd.Timedelta(
            minutes=tf_minutes[source_tf]
        )
        left = left.dropna(subset=["_lookup"])
        if left.empty:
            return np.full(len(target_df), "UNKNOWN", dtype=object)
        right = source_df[["timestamp", "MARKET_STRUCTURE"]].copy()
        right["timestamp"] = pd.to_datetime(right["timestamp"])
        right = right.sort_values("timestamp")
        merged = pd.merge_asof(
            left.sort_values("_lookup"),
            right,
            left_on="_lookup",
            right_on="timestamp",
            direction="backward",
        )
        out = pd.Series("UNKNOWN", index=target_df.index, dtype=object)
        out.iloc[left["_row_order"]] = (
            merged.sort_values("_row_order")["MARKET_STRUCTURE"]
            .fillna("UNKNOWN")
            .to_numpy()
        )
        return out.to_numpy()

    # تخصیص دستهای ستونهای MTF برای جلوگیری از پراکندگی DataFrame
    cols = {}
    for tf, df in dataframes.items():
        h4 = np.full(len(df), "UNKNOWN", dtype=object)
        h1 = np.full(len(df), "UNKNOWN", dtype=object)
        if tf == "4h" and "MARKET_STRUCTURE" in df.columns:
            h4 = df["MARKET_STRUCTURE"].to_numpy()
        elif "4h" in dataframes and "MARKET_STRUCTURE" in dataframes["4h"].columns:
            h4 = closed_htf_state(df, dataframes["4h"], "4h")

        if tf == "1h" and "MARKET_STRUCTURE" in df.columns:
            h1 = df["MARKET_STRUCTURE"].to_numpy()
        elif "1h" in dataframes and "MARKET_STRUCTURE" in dataframes["1h"].columns:
            h1 = closed_htf_state(df, dataframes["1h"], "1h")

        conflict = np.full(len(df), "N/A", dtype=object)
        if tf != "4h":
            conflict = np.where(
                (pd.Series(h4).isin(["BULLISH", "BEARISH"])) &
                (pd.Series(h1).isin(["BULLISH", "BEARISH"])) &
                (pd.Series(h4) != pd.Series(h1)),
                "CONFLICT", "NONE"
            )
        cols[tf] = {"HTF_4H_BIAS": h4, "HTF_1H_STRUCTURE": h1, "MTF_CONFLICT": conflict}

    for tf, df in dataframes.items():
        df["HTF_4H_BIAS"] = cols[tf]["HTF_4H_BIAS"]
        df["HTF_1H_STRUCTURE"] = cols[tf]["HTF_1H_STRUCTURE"]
        df["MTF_CONFLICT"] = cols[tf]["MTF_CONFLICT"]
    return dataframes

def process_symbol(symbol):
    dataframes = {}
    for tf in ["5m", "15m", "1h", "4h"]:
        inp = DATA_DIR / f"{symbol}_{tf}_phase2.csv"
        if not inp.exists():
            continue

        try:
            df = load_dataframe(inp)
            df = detect_swings(df)
            df = classify_structure_points(df)
            df = detect_equal_levels(df)
            df = build_liquidity_pools(df)
            df = detect_liquidity_sweeps(df)
            df = classify_liquidity(df)
            df = detect_displacement(df)
            df = detect_structure_events(df)
            df = rebuild_market_structure(df)
            df = calculate_premium_discount(df)
            df = add_bars_since(df)
            df = calculate_structure_score_and_signal(df)
            dataframes[tf] = df
        except Exception as e:
            logger.exception(f"Error processing {symbol} {tf}: {e}")

    if not dataframes:
        return

    dataframes = add_mtf_context_for_symbol(symbol, dataframes)
    for tf, df in dataframes.items():
        out = DATA_DIR / f"{symbol}_{tf}_phase3.csv"
        df.to_csv(out, index=False)

def process_phase3(symbols=None):
    if symbols is None:
        from config import SYMBOLS
        symbols = SYMBOLS
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        process_symbol(symbol)