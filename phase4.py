import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
FVG_MIN_GAP = 0.0
FVG_MAX_AGE = 500
ATR_PERIOD = 14
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan

def ensure_timestamp(df):
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df.get("open_time", np.arange(len(df))), errors="coerce")
    return df

def sort_chronologically(df):
    return df.sort_values("timestamp").reset_index(drop=True)

def load_phase3(path):
    df = pd.read_csv(path)
    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    df = ensure_timestamp(df)
    return sort_chronologically(df)

def _phase4_atr(df):
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    if "ATR" in df.columns:
        atr = df["ATR"].combine_first(atr)
    return atr

def detect_fvg_imbalance_and_mitigation(df):
    df = df.copy()
    n = len(df)
    cols_int = ["FVG", "IMBALANCE", "FVG_CE_TOUCHED", "FVG_ACTIVE_COUNT",
                "FVG_ACTIVE_BULLISH_COUNT", "FVG_ACTIVE_BEARISH_COUNT"]
    cols_f = ["FVG_TOP", "FVG_BOTTOM", "FVG_SIZE", "FVG_CE_LEVEL", "FVG_ATR_RATIO",
              "FVG_MITIGATION_PRICE", "FVG_FORMATION_INDEX", "FVG_MITIGATION_INDEX",
              "FVG_AGE", "ACTIVE_FVG_TOP", "ACTIVE_FVG_BOTTOM", "ACTIVE_FVG_CE_LEVEL",
              "ACTIVE_FVG_AGE", "ACTIVE_FVG_ATR_RATIO"]
    cols_s = ["FVG_DIRECTION", "FVG_ID", "FVG_MITIGATION_STATUS", "FVG_MITIGATION_DIRECTION",
              "ACTIVE_FVG_ID", "ACTIVE_FVG_DIRECTION", "ACTIVE_FVG_STATUS", "IMBALANCE_DIRECTION"]
    for c in cols_int:
        df[c] = np.zeros(n, dtype=np.int8)
    for c in cols_f:
        df[c] = np.nan
    for c in cols_s:
        df[c] = ""

    if n < 3:
        return df

    high = df["high"].values
    low = df["low"].values
    atr_series = _phase4_atr(df).values
    disp = df["DISPLACEMENT"].values if "DISPLACEMENT" in df.columns else np.zeros(n)
    disp_dir = df["DISPLACEMENT_DIRECTION"].values if "DISPLACEMENT_DIRECTION" in df.columns else np.array([""] * n)

    # تشخیص FVG بهصورت وکتوری (gap بین کندل i و i-2)
    bull_gap = low[2:] > high[:-2]
    bear_gap = high[2:] < low[:-2]
    fvg_idx = np.where(bull_gap | bear_gap)[0] + 2

    col_FVG = df.columns.get_loc("FVG")
    col_ID = df.columns.get_loc("FVG_ID")
    col_DIR = df.columns.get_loc("FVG_DIRECTION")
    col_TOP = df.columns.get_loc("FVG_TOP")
    col_BOT = df.columns.get_loc("FVG_BOTTOM")
    col_SIZE = df.columns.get_loc("FVG_SIZE")
    col_CE = df.columns.get_loc("FVG_CE_LEVEL")
    col_ATRR = df.columns.get_loc("FVG_ATR_RATIO")
    col_FI = df.columns.get_loc("FVG_FORMATION_INDEX")
    col_IMB = df.columns.get_loc("IMBALANCE")
    col_IMBD = df.columns.get_loc("IMBALANCE_DIRECTION")

    next_id = 1
    for i in fvg_idx:
        if bull_gap[i - 2]:
            gap = low[i] - high[i - 2]
            if gap <= FVG_MIN_GAP:
                continue
            bot, top = high[i - 2], low[i]
            direction = "BULLISH"
            is_imb = 1 if (disp[i - 1] == 1 and disp_dir[i - 1] == "BULLISH") else 0
        else:
            gap = low[i - 2] - high[i]
            if gap <= FVG_MIN_GAP:
                continue
            bot, top = high[i], low[i - 2]
            direction = "BEARISH"
            is_imb = 1 if (disp[i - 1] == 1 and disp_dir[i - 1] == "BEARISH") else 0
        curr_atr = atr_series[i]
        ce = (bot + top) / 2.0
        atr_r = gap / curr_atr if np.isfinite(curr_atr) and curr_atr > 0 else np.nan
        z_id = f"FVG_{next_id:06d}"; next_id += 1
        df.iat[i, col_FVG] = 1
        df.iat[i, col_ID] = z_id
        df.iat[i, col_DIR] = direction
        df.iat[i, col_TOP] = top
        df.iat[i, col_BOT] = bot
        df.iat[i, col_SIZE] = gap
        df.iat[i, col_CE] = ce
        df.iat[i, col_ATRR] = atr_r
        df.iat[i, col_FI] = i
        df.iat[i, col_IMB] = is_imb
        df.iat[i, col_IMBD] = direction if is_imb else ""

    # پیگیری نواحی فعال (تعداد نواحی کم است)
    fvg_formed = np.where(df["FVG"].values == 1)[0]
    formed_dir = df["FVG_DIRECTION"].values[fvg_formed]
    formed_bot = df["FVG_BOTTOM"].values[fvg_formed].astype(float)
    formed_top = df["FVG_TOP"].values[fvg_formed].astype(float)
    formed_ce = df["FVG_CE_LEVEL"].values[fvg_formed].astype(float)
    formed_atr = df["FVG_ATR_RATIO"].values[fvg_formed].astype(float)

    bull_cnt_arr = np.zeros(n, dtype=np.int16)
    bear_cnt_arr = np.zeros(n, dtype=np.int16)
    at_top = np.full(n, np.nan)
    at_bot = np.full(n, np.nan)
    at_ce = np.full(n, np.nan)
    at_age = np.full(n, np.nan)
    at_atr = np.full(n, np.nan)

    active = []
    formed_pos = {int(idx): pos for pos, idx in enumerate(fvg_formed)}
    for i in range(2, n):
        if i in formed_pos:
            active.append(formed_pos[i])
        li, hi = low[i], high[i]
        remaining = []
        for idx in active:
            age = i - fvg_formed[idx]
            if age > FVG_MAX_AGE:
                continue
            if formed_dir[idx] == "BULLISH" and li <= formed_bot[idx]:
                continue
            if formed_dir[idx] == "BEARISH" and hi >= formed_top[idx]:
                continue
            remaining.append(idx)
        active = remaining

        bc = sum(1 for idx in active if formed_dir[idx] == "BULLISH")
        kc = sum(1 for idx in active if formed_dir[idx] == "BEARISH")
        bull_cnt_arr[i] = bc
        bear_cnt_arr[i] = kc
        if active:
            last = active[-1]
            at_top[i] = formed_top[last]
            at_bot[i] = formed_bot[last]
            at_ce[i] = formed_ce[last]
            at_age[i] = i - fvg_formed[last]
            at_atr[i] = formed_atr[last]

    df["FVG_ACTIVE_BULLISH_COUNT"] = bull_cnt_arr
    df["FVG_ACTIVE_BEARISH_COUNT"] = bear_cnt_arr
    df["FVG_ACTIVE_COUNT"] = bull_cnt_arr + bear_cnt_arr
    df["ACTIVE_FVG_TOP"] = at_top
    df["ACTIVE_FVG_BOTTOM"] = at_bot
    df["ACTIVE_FVG_CE_LEVEL"] = at_ce
    df["ACTIVE_FVG_AGE"] = at_age
    df["ACTIVE_FVG_ATR_RATIO"] = at_atr
    return df


def detect_order_blocks(df):
    df = df.copy()
    n = len(df)
    for col in ["ORDER_BLOCK", "OB_ACTIVE_COUNT", "OB_MITIGATED", "OB_IN_ZONE"]:
        df[col] = np.zeros(n, dtype=np.int8)
    for col in ["OB_TOP", "OB_BOTTOM", "OB_MID", "OB_AGE", "OB_STRENGTH",
                "ACTIVE_OB_TOP", "ACTIVE_OB_BOTTOM", "ACTIVE_OB_MID",
                "ACTIVE_OB_AGE", "ACTIVE_OB_STRENGTH", "OB_DISTANCE_PCT"]:
        df[col] = np.nan
    for col in ["OB_DIRECTION", "ACTIVE_OB_DIRECTION", "OB_IN_ZONE_DIRECTION"]:
        df[col] = ""

    if n < 5:
        return df

    atr_series = _phase4_atr(df).values
    has_disp = "DISPLACEMENT" in df.columns
    open_v = df["open"].values
    high_v = df["high"].values
    low_v = df["low"].values
    close_v = df["close"].values
    disp_v = df["DISPLACEMENT"].values if has_disp else np.zeros(n)
    disp_dir_v = df["DISPLACEMENT_DIRECTION"].values if "DISPLACEMENT_DIRECTION" in df.columns else np.array([""] * n)

    c_OB = df.columns.get_loc("ORDER_BLOCK")
    c_OBD = df.columns.get_loc("OB_DIRECTION")
    c_OBT = df.columns.get_loc("OB_TOP")
    c_OBB = df.columns.get_loc("OB_BOTTOM")
    c_OBM = df.columns.get_loc("OB_MID")
    c_OBS = df.columns.get_loc("OB_STRENGTH")
    c_OAC = df.columns.get_loc("OB_ACTIVE_COUNT")
    c_OBMI = df.columns.get_loc("OB_MITIGATED")
    c_OBIZ = df.columns.get_loc("OB_IN_ZONE")
    c_OBIZD = df.columns.get_loc("OB_IN_ZONE_DIRECTION")
    c_AOBT = df.columns.get_loc("ACTIVE_OB_TOP")
    c_AOBB = df.columns.get_loc("ACTIVE_OB_BOTTOM")
    c_AOBM = df.columns.get_loc("ACTIVE_OB_MID")
    c_AOBA = df.columns.get_loc("ACTIVE_OB_AGE")
    c_AOBS = df.columns.get_loc("ACTIVE_OB_STRENGTH")
    c_OBDP = df.columns.get_loc("OB_DISTANCE_PCT")

    # تشخیص OB روی کندلهای حرکت انفجاری
    ob_formed = []
    for i in range(1, n):
        curr_atr = atr_series[i]
        if not (disp_v[i] == 1 and np.isfinite(curr_atr) and curr_atr > 0):
            continue
        direction = disp_dir_v[i]
        base = i - 1
        if direction == "BULLISH" and close_v[base] < open_v[base]:
            top, bottom = max(open_v[base], high_v[base]), low_v[base]
            strength = abs(close_v[i] - open_v[i]) / curr_atr
            df.iat[i, c_OB] = 1
            df.iat[i, c_OBD] = "BULLISH"
            df.iat[i, c_OBT] = top
            df.iat[i, c_OBB] = bottom
            df.iat[i, c_OBM] = (top + bottom) / 2.0
            df.iat[i, c_OBS] = strength
            ob_formed.append((i, "BULLISH", top, bottom, strength))
        elif direction == "BEARISH" and close_v[base] > open_v[base]:
            top, bottom = high_v[base], min(open_v[base], low_v[base])
            strength = abs(close_v[i] - open_v[i]) / curr_atr
            df.iat[i, c_OB] = 1
            df.iat[i, c_OBD] = "BEARISH"
            df.iat[i, c_OBT] = top
            df.iat[i, c_OBB] = bottom
            df.iat[i, c_OBM] = (top + bottom) / 2.0
            df.iat[i, c_OBS] = strength
            ob_formed.append((i, "BEARISH", top, bottom, strength))

    idx_arr = np.array([x[0] for x in ob_formed])
    dir_arr = np.array([x[1] for x in ob_formed])
    top_arr = np.array([x[2] for x in ob_formed], dtype=float)
    bot_arr = np.array([x[3] for x in ob_formed], dtype=float)
    str_arr = np.array([x[4] for x in ob_formed], dtype=float)

    ob_active_cnt = np.zeros(n, dtype=np.int16)
    ob_mit = np.zeros(n, dtype=np.int8)
    ob_in_zone = np.zeros(n, dtype=np.int8)
    ob_in_zone_dir = np.array([""] * n)
    a_top = np.full(n, np.nan)
    a_bot = np.full(n, np.nan)
    a_mid = np.full(n, np.nan)
    a_age = np.full(n, np.nan)
    a_str = np.full(n, np.nan)
    ob_dist = np.full(n, np.nan)

    active = []
    for i in range(1, n):
        li, hi, ci = low_v[i], high_v[i], close_v[i]
        remaining = []
        for k in active:
            age = i - idx_arr[k]
            if age > FVG_MAX_AGE:
                continue
            broken = (dir_arr[k] == "BULLISH" and ci < bot_arr[k]) or                      (dir_arr[k] == "BEARISH" and ci > top_arr[k])
            if broken:
                ob_mit[i] = 1
                continue
            if li <= top_arr[k] and hi >= bot_arr[k]:
                ob_in_zone[i] = 1
                ob_in_zone_dir[i] = dir_arr[k]
            remaining.append(k)
        active = remaining

        ob_active_cnt[i] = len(active)
        if active:
            last = active[-1]
            mid = (top_arr[last] + bot_arr[last]) / 2.0
            a_top[i] = top_arr[last]
            a_bot[i] = bot_arr[last]
            a_mid[i] = mid
            a_age[i] = i - idx_arr[last]
            a_str[i] = str_arr[last]
            if mid > 0:
                ob_dist[i] = (ci - mid) / mid * 100.0

    df["OB_ACTIVE_COUNT"] = ob_active_cnt
    df["OB_MITIGATED"] = ob_mit
    df["OB_IN_ZONE"] = ob_in_zone
    df["OB_IN_ZONE_DIRECTION"] = ob_in_zone_dir
    df["ACTIVE_OB_TOP"] = a_top
    df["ACTIVE_OB_BOTTOM"] = a_bot
    df["ACTIVE_OB_MID"] = a_mid
    df["ACTIVE_OB_AGE"] = a_age
    df["ACTIVE_OB_STRENGTH"] = a_str
    df["OB_DISTANCE_PCT"] = ob_dist
    return df


def detect_breaker_blocks(df):
    """Breaker = Order Block که قبلاً شکسته شده بود و حالا به عنوان مقاومت/حمایت برمیگردد.
    Mitigation Block = OB که فقط تا سطح 50% نفوذ شده (تأیید نشده کامل)."""
    df = df.copy()
    n = len(df)
    cols_int = ["BREAKER_BLOCK", "MB_BLOCK"]
    cols_f = ["BREAKER_TOP", "BREAKER_BOTTOM", "BREAKER_MID", "MB_TOP", "MB_BOTTOM", "MB_MID"]
    cols_s = ["BREAKER_DIRECTION", "MB_DIRECTION", "MB_CONFIRMED"]
    for c in cols_int:
        df[c] = np.zeros(n, dtype=np.int8)
    for c in cols_f:
        df[c] = np.nan
    for c in cols_s:
        df[c] = ""

    if n < 5 or "DISPLACEMENT" not in df.columns:
        return df

    open_v = df["open"].values
    high_v = df["high"].values
    low_v = df["low"].values
    close_v = df["close"].values
    disp_v = df["DISPLACEMENT"].values
    disp_dir_v = df["DISPLACEMENT_DIRECTION"].values
    atr_series = _phase4_atr(df)

    breakers = []
    for i in range(1, n):
        curr_atr = atr_series.iloc[i]
        if not (disp_v[i] == 1 and pd.notna(curr_atr) and curr_atr > 0):
            continue
        direction = disp_dir_v[i]
        base = i - 1
        # بریکر صعودی: قبلاً یک OB نزولی شکسته شده بود، حالا قیمت بالای آن برگشته است
        if direction == "BULLISH":
            # OB نزولی قبلی (high کندل پایه بالای close کندل فعلی)
            top, bottom = high_v[base], min(open_v[base], low_v[base])
            if low_v[i] <= bottom and close_v[i] > bottom:
                mid = (top + bottom) / 2.0
                df.loc[i, "BREAKER_BLOCK"] = 1
                df.loc[i, "BREAKER_DIRECTION"] = "BULLISH"
                df.loc[i, "BREAKER_TOP"] = top
                df.loc[i, "BREAKER_BOTTOM"] = bottom
                df.loc[i, "BREAKER_MID"] = mid
                breakers.append({"idx": i, "direction": "BULLISH", "top": top, "bottom": bottom})
                # Mitigation Block: نفوذ فقط تا 50% کندل مخالف
                if low_v[i] >= mid:
                    df.loc[i, "MB_BLOCK"] = 1
                    df.loc[i, "MB_DIRECTION"] = "BULLISH"
                    df.loc[i, "MB_TOP"] = top
                    df.loc[i, "MB_BOTTOM"] = bottom
                    df.loc[i, "MB_MID"] = mid
                    df.loc[i, "MB_CONFIRMED"] = "NO"
                elif low_v[i] <= bottom:
                    df.loc[i, "MB_BLOCK"] = 1
                    df.loc[i, "MB_DIRECTION"] = "BULLISH"
                    df.loc[i, "MB_TOP"] = top
                    df.loc[i, "MB_BOTTOM"] = bottom
                    df.loc[i, "MB_MID"] = mid
                    df.loc[i, "MB_CONFIRMED"] = "YES"
        elif direction == "BEARISH":
            top, bottom = max(open_v[base], high_v[base]), low_v[base]
            if high_v[i] >= top and close_v[i] < top:
                mid = (top + bottom) / 2.0
                df.loc[i, "BREAKER_BLOCK"] = 1
                df.loc[i, "BREAKER_DIRECTION"] = "BEARISH"
                df.loc[i, "BREAKER_TOP"] = top
                df.loc[i, "BREAKER_BOTTOM"] = bottom
                df.loc[i, "BREAKER_MID"] = mid
                breakers.append({"idx": i, "direction": "BEARISH", "top": top, "bottom": bottom})
                if high_v[i] <= mid:
                    df.loc[i, "MB_BLOCK"] = 1
                    df.loc[i, "MB_DIRECTION"] = "BEARISH"
                    df.loc[i, "MB_TOP"] = top
                    df.loc[i, "MB_BOTTOM"] = bottom
                    df.loc[i, "MB_MID"] = mid
                    df.loc[i, "MB_CONFIRMED"] = "NO"
                elif high_v[i] >= top:
                    df.loc[i, "MB_BLOCK"] = 1
                    df.loc[i, "MB_DIRECTION"] = "BEARISH"
                    df.loc[i, "MB_TOP"] = top
                    df.loc[i, "MB_BOTTOM"] = bottom
                    df.loc[i, "MB_MID"] = mid
                    df.loc[i, "MB_CONFIRMED"] = "YES"

    # نگه داشتن آخرین بریکر فعال در هر ردیف
    active_breakers = []
    for i in range(1, n):
        for bb in active_breakers:
            if bb["idx"] == i:
                df.loc[i, "ACTIVE_BREAKER_DIRECTION"] = bb["direction"]
                df.loc[i, "ACTIVE_BREAKER_TOP"] = bb["top"]
                df.loc[i, "ACTIVE_BREAKER_BOTTOM"] = bb["bottom"]
                break
        active_breakers = [bb for bb in active_breakers if i - bb["idx"] <= FVG_MAX_AGE]
        if df.loc[i, "BREAKER_BLOCK"] == 1:
            active_breakers.append({"idx": i, "direction": df.at[i, "BREAKER_DIRECTION"],
                                    "top": df.at[i, "BREAKER_TOP"], "bottom": df.at[i, "BREAKER_BOTTOM"]})
    return df

def process_file(input_file):
    out_file = input_file.parent / input_file.name.replace("_phase3.csv", "_phase4.csv")

    # اگر فایل خروجی جدیدتر از ورودی است → SKIP
    if out_file.exists() and out_file.stat().st_mtime > input_file.stat().st_mtime:
        return

    # در غیر این صورت پردازش کن
    df = load_phase3(input_file)
    df = detect_fvg_imbalance_and_mitigation(df)
    df = detect_order_blocks(df)
    df = detect_breaker_blocks(df)
    df.to_csv(out_file, index=False)

def process_phase4(symbols=None):
    if symbols is None:
        from config import SYMBOLS
        symbols = SYMBOLS
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        for tf in ["5m","15m","1h","4h"]:
            inp = DATA_DIR / f"{symbol}_{tf}_phase3.csv"
            if inp.exists():
                process_file(inp)