import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

def detect_candlestick_patterns(df):
    df = df.copy()
    open_p, high_p, low_p, close_p = df["open"], df["high"], df["low"], df["close"]
    body = (close_p - open_p).abs()
    candle_range = (high_p - low_p).replace(0, 1e-8)
    df["PATTERN_DOJI"] = (body <= (candle_range * 0.1)).astype(int)
    prev_close = close_p.shift(1)
    prev_open = open_p.shift(1)
    bull_eng = (prev_close < prev_open) & (close_p > open_p) & (close_p >= prev_open) & (open_p <= prev_close)
    bear_eng = (prev_close > prev_open) & (close_p < open_p) & (close_p <= prev_open) & (open_p >= prev_close)
    df["PATTERN_BULL_ENGULFING"] = bull_eng.astype(int)
    df["PATTERN_BEAR_ENGULFING"] = bear_eng.astype(int)
    upper_wick = high_p - np.maximum(open_p, close_p)
    lower_wick = np.minimum(open_p, close_p) - low_p
    df["PATTERN_BULL_PINBAR"] = ((lower_wick >= 2*body) & (upper_wick <= 0.5*body)).astype(int)
    df["PATTERN_BEAR_PINBAR"] = ((upper_wick >= 2*body) & (lower_wick <= 0.5*body)).astype(int)
    return df

def detect_chart_patterns(df, window=20):
    df = df.copy()
    highs, lows = df["high"], df["low"]

    def slope(series):
        if len(series) < window:
            return 0.0
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]

    high_slopes = highs.rolling(window, min_periods=window).apply(slope, raw=True).fillna(0)
    low_slopes = lows.rolling(window, min_periods=window).apply(slope, raw=True).fillna(0)
    slope_diff = (high_slopes - low_slopes).abs()

    df["PATTERN_CHANNEL_UP"] = ((high_slopes > 1e-4) & (low_slopes > 1e-4) & (slope_diff < 5e-4)).astype(int)
    df["PATTERN_CHANNEL_DOWN"] = ((high_slopes < -1e-4) & (low_slopes < -1e-4) & (slope_diff < 5e-4)).astype(int)
    df["PATTERN_TRIANGLE_SYM"] = ((high_slopes < -1e-4) & (low_slopes > 1e-4)).astype(int)
    df["PATTERN_TRIANGLE_ASC"] = ((high_slopes.abs() < 1e-4) & (low_slopes > 1e-4)).astype(int)
    df["PATTERN_TRIANGLE_DESC"] = ((high_slopes < -1e-4) & (low_slopes.abs() < 1e-4)).astype(int)

    displacement_dir = df.get("DISPLACEMENT_DIRECTION", pd.Series("", index=df.index)).astype(str)
    recent_bull_disp = displacement_dir.eq("BULLISH").rolling(10, min_periods=1).max().astype(bool)
    recent_bear_disp = displacement_dir.eq("BEARISH").rolling(10, min_periods=1).max().astype(bool)
    df["PATTERN_BULL_FLAG"] = (recent_bull_disp & (high_slopes < 0) & (low_slopes < 0)).astype(int)
    df["PATTERN_BEAR_FLAG"] = (recent_bear_disp & (high_slopes > 0) & (low_slopes > 0)).astype(int)
    df["PATTERN_RISING_WEDGE"] = ((high_slopes > 0) & (low_slopes > 0) & (high_slopes < low_slopes)).astype(int)
    df["PATTERN_FALLING_WEDGE"] = ((high_slopes < 0) & (low_slopes < 0) & (high_slopes < low_slopes)).astype(int)
    return df

def detect_harmonic_patterns(df):
    df = df.copy()
    df["HARMONIC_PATTERN"] = "NONE"
    df["HARMONIC_TYPE"] = "NONE"
    df["HARMONIC_STRENGTH"] = 0.0

    n = len(df)
    if n < 40:
        return df

    high_p = df["high"].values
    low_p = df["low"].values

    def _pivots(arr, mode="high", window=4):
        s = pd.Series(arr)
        if mode == "high":
            cond = arr == s.rolling(window * 2 + 1, center=False, min_periods=window * 2 + 1).max().values
        else:
            cond = arr == s.rolling(window * 2 + 1, center=False, min_periods=window * 2 + 1).min().values
        return np.where(cond)[0]

    piv_hi = _pivots(high_p, "high", 4)
    piv_lo = _pivots(low_p, "low", 4)

    def _build_zigzag():
        pts = []
        for idx in piv_hi:
            pts.append((idx, float(high_p[idx]), "HIGH"))
        for idx in piv_lo:
            pts.append((idx, float(low_p[idx]), "LOW"))
        pts.sort(key=lambda x: x[0])
        zz = []
        for pt in pts:
            if zz and zz[-1][2] == pt[2]:
                if pt[2] == "HIGH" and pt[1] > zz[-1][1]:
                    zz[-1] = pt
                elif pt[2] == "LOW" and pt[1] < zz[-1][1]:
                    zz[-1] = pt
                continue
            zz.append(pt)
        return zz

    zz = _build_zigzag()
    if len(zz) < 5:
        return df

    def _in_tol(val, lo, hi, tol=0.05):
        return lo * (1 - tol) <= val <= hi * (1 + tol)

    def _pattern_score(pattern, ratios, ideal):
        score = 0.0
        for r, (lo, hi) in zip(ratios, ideal):
            if lo <= r <= hi:
                score += 1.0
            elif lo * 0.92 <= r <= hi * 1.08:
                score += 0.6
            elif lo * 0.85 <= r <= hi * 1.15:
                score += 0.3
        return score / len(ideal)

    pattern_map = {}

    for i in range(4, len(zz)):
        _, pX, _ = zz[i - 4]
        _, pA, _ = zz[i - 3]
        _, pB, _ = zz[i - 2]
        _, pC, _ = zz[i - 1]
        idx_D, pD, type_D = zz[i]

        XA = abs(pA - pX)
        AB = abs(pB - pA)
        BC = abs(pC - pB)
        CD = abs(pD - pC)
        AD = abs(pD - pX)
        XD = abs(pD - pX)
        XC = abs(pC - pX)
        BD = abs(pD - pB)

        if XA == 0 or AB == 0 or BC == 0:
            continue

        rAB = AB / XA
        rBC = BC / AB if AB != 0 else 0
        rCD = CD / BC if BC != 0 else 0
        rXD = XD / XA if XA != 0 else 0
        rXC = XC / XA if XA != 0 else 0
        rBD = BD / BC if BC != 0 else 0
        rAD = AD / XA if XA != 0 else 0

        p_type = "BULLISH" if type_D == "LOW" else "BEARISH"
        best = ("NONE", 0.0)

        # Gartley
        ratios = (rAB, rBC, rCD, rXD)
        ideal = ((0.618, 0.618), (0.382, 0.886), (1.272, 1.618), (0.786, 0.786))
        if _in_tol(rAB, 0.55, 0.65) and _in_tol(rBC, 0.382, 0.886) and _in_tol(rXD, 0.70, 0.82):
            s = _pattern_score("GARTLEY", ratios, ideal)
            if s > best[1]:
                best = ("GARTLEY", s)

        # Bat
        if _in_tol(rAB, 0.382, 0.50) and _in_tol(rBC, 0.382, 0.886) and _in_tol(rXD, 0.618, 0.70) and _in_tol(rAD, 1.618, 2.618):
            s = _pattern_score("BAT", ratios, ((0.382, 0.50), (0.382, 0.886), (1.272, 1.618), (0.618, 0.70)))
            if s > best[1]:
                best = ("BAT", s)

        # Butterfly
        if _in_tol(rAB, 0.618, 0.786) and _in_tol(rBC, 0.382, 0.886) and _in_tol(rCD, 1.618, 2.618) and _in_tol(rXD, 1.272, 1.414):
            s = _pattern_score("BUTTERFLY", ratios, ((0.618, 0.786), (0.382, 0.886), (1.618, 2.618), (1.272, 1.414)))
            if s > best[1]:
                best = ("BUTTERFLY", s)

        # Crab
        if _in_tol(rAB, 0.382, 0.618) and _in_tol(rBC, 0.382, 0.886) and _in_tol(rCD, 2.618, 3.618) and _in_tol(rXD, 1.618, 1.786):
            s = _pattern_score("CRAB", ratios, ((0.382, 0.618), (0.382, 0.886), (2.618, 3.618), (1.618, 1.786)))
            if s > best[1]:
                best = ("CRAB", s)

        # Shark
        if _in_tol(rAB, 0.44, 0.618) and _in_tol(rBC, 1.618, 2.24) and _in_tol(rXD, 0.886, 1.13) and _in_tol(rXC, 1.618, 2.24):
            s = _pattern_score("SHARK", ratios, ((0.44, 0.618), (1.618, 2.24), (0.0, 0.0), (0.886, 1.13)))
            if s > best[1]:
                best = ("SHARK", s)

        # Cypher
        if _in_tol(rAB, 0.382, 0.618) and _in_tol(rBC, 1.13, 1.414) and _in_tol(rCD, 1.618, 2.0) and _in_tol(rXD, 0.786, 0.886):
            s = _pattern_score("CYPHER", ratios, ((0.382, 0.618), (1.13, 1.414), (1.618, 2.0), (0.786, 0.886)))
            if s > best[1]:
                best = ("CYPHER", s)

        # 5-0
        if _in_tol(rAB, 0.50, 0.618) and _in_tol(rBC, 0.618, 0.786) and _in_tol(rCD, 1.0, 1.618) and _in_tol(rXD, 0.50, 0.886):
            s = _pattern_score("5-0", ratios, ((0.50, 0.618), (0.618, 0.786), (1.0, 1.618), (0.50, 0.886)))
            if s > best[1]:
                best = ("5-0", s)

        pattern, strength = best
        if pattern != "NONE" and strength >= 0.5:
            pattern_map[idx_D] = (pattern, p_type, strength)

    patterns = []
    types = []
    strengths = []
    for idx in range(n):
        if idx in pattern_map:
            p, t, s = pattern_map[idx]
            patterns.append(p)
            types.append(t)
            strengths.append(round(float(s), 2))
        else:
            patterns.append("NONE")
            types.append("NONE")
            strengths.append(0.0)

    df["HARMONIC_PATTERN"] = patterns
    df["HARMONIC_TYPE"] = types
    df["HARMONIC_STRENGTH"] = strengths
    return df

def detect_classic_patterns(df, tol=0.012, swing_window=5, lookback=60):
    """تشخیص الگوهای کلاسیک: Double/Triple Top-Bottom، Head & Shoulders، Rectangle.
    بر اساس نقاط سقف/کف سوینگ (نه look-ahead)."""
    df = df.copy()
    df["CLASSIC_PATTERN"] = "NONE"
    df["CLASSIC_TYPE"] = "NONE"

    high_p = df["high"].values
    low_p = df["low"].values
    n = len(df)
    if n < 3 * swing_window + 1:
        return df

    piv_hi = (df["high"] == df["high"].rolling(swing_window * 2 + 1, center=False, min_periods=swing_window * 2 + 1).max()).fillna(False)
    piv_lo = (df["low"] == df["low"].rolling(swing_window * 2 + 1, center=False, min_periods=swing_window * 2 + 1).min()).fillna(False)

    highs = [(i, high_p[i]) for i in range(n) if piv_hi[i]]
    lows = [(i, low_p[i]) for i in range(n) if piv_lo[i]]

    def near(a, b):
        return abs(a - b) / (abs(a) + 1e-12) <= tol

    for i in range(n):
        confirm = i + swing_window
        if confirm >= n:
            break
        # --- Double / Triple Top (Bearish) ---
        h = [(idx, p) for idx, p in highs if idx < i and i - idx <= lookback]
        if len(h) >= 2:
            # دو سقف نزدیک به هم با یک کف میانی
            for a in range(len(h) - 1):
                ia, pa = h[a]
                ib, pb = h[a + 1]
                if near(pa, pb):
                    mids = [p for idx, p in lows if ia < idx < ib]
                    if mids and min(mids) < min(pa, pb) * (1 - tol):
                        if low_p[confirm] < min(mids):
                            df.loc[confirm, "CLASSIC_PATTERN"] = "DOUBLE_TOP"
                            df.loc[confirm, "CLASSIC_TYPE"] = "BEARISH"
                            break
        # --- Double / Triple Bottom (Bullish) ---
        l = [(idx, p) for idx, p in lows if idx < i and i - idx <= lookback]
        if len(l) >= 2:
            for a in range(len(l) - 1):
                ia, pa = l[a]
                ib, pb = l[a + 1]
                if near(pa, pb):
                    mids = [p for idx, p in highs if ia < idx < ib]
                    if mids and max(mids) > max(pa, pb) * (1 + tol):
                        if high_p[confirm] > max(mids):
                            df.loc[confirm, "CLASSIC_PATTERN"] = "DOUBLE_BOTTOM"
                            df.loc[confirm, "CLASSIC_TYPE"] = "BULLISH"
                            break
        # --- Head & Shoulders (Bearish) ---
        if len(h) >= 3:
            for a in range(len(h) - 2):
                ia, pa = h[a]
                ib, pb = h[a + 1]
                ic, pc = h[a + 2]
                if pb > pa and pb > pc and near(pa, pc):
                    necks = [p for idx, p in lows if ia < idx < ic]
                    if necks:
                        neck = max(necks)
                        if low_p[confirm] < neck:
                            df.loc[confirm, "CLASSIC_PATTERN"] = "HEAD_SHOULDERS"
                            df.loc[confirm, "CLASSIC_TYPE"] = "BEARISH"
                            break
        # --- Inverse Head & Shoulders (Bullish) ---
        if len(l) >= 3:
            for a in range(len(l) - 2):
                ia, pa = l[a]
                ib, pb = l[a + 1]
                ic, pc = l[a + 2]
                if pb < pa and pb < pc and near(pa, pc):
                    necks = [p for idx, p in highs if ia < idx < ic]
                    if necks:
                        neck = min(necks)
                        if high_p[confirm] > neck:
                            df.loc[confirm, "CLASSIC_PATTERN"] = "INV_HEAD_SHOULDERS"
                            df.loc[confirm, "CLASSIC_TYPE"] = "BULLISH"
                            break
        # --- Rectangle (Continuation) ---
        if len(h) >= 1 and len(l) >= 1:
            hr = [p for idx, p in highs if idx < i and i - idx <= lookback and piv_hi[idx]]
            lr = [p for idx, p in lows if idx < i and i - idx <= lookback and piv_lo[idx]]
            if len(hr) >= 2 and len(lr) >= 2:
                up_res = max(hr)
                down_sup = min(lr)
                if up_res > down_sup and (up_res - down_sup) / down_sup < 0.08:
                    if high_p[confirm] > up_res:
                        df.loc[confirm, "CLASSIC_PATTERN"] = "RECTANGLE_BREAK_UP"
                        df.loc[confirm, "CLASSIC_TYPE"] = "BULLISH"
                    elif low_p[confirm] < down_sup:
                        df.loc[confirm, "CLASSIC_PATTERN"] = "RECTANGLE_BREAK_DOWN"
                        df.loc[confirm, "CLASSIC_TYPE"] = "BEARISH"
    return df


def detect_elliott_wave(df, swing_window=5, lookback=120):
    """تشخیص ساختار موج الیوت (امواج 1-2-3-4-5 و اصلاح ABC/Flat/Zigzag/Triangle) روی نقاط سوینگ."""
    df = df.copy()
    df["ELLIOTT_WAVE"] = "NONE"
    df["ELLIOTT_TYPE"] = "NONE"
    df["ELLIOTT_WAVE_COUNT"] = 0

    n = len(df)
    if n < 3 * swing_window + 1:
        return df

    high_p = df["high"].values
    low_p = df["low"].values

    def _pivots(arr, mode="high", window=4):
        s = pd.Series(arr)
        if mode == "high":
            cond = arr == s.rolling(window * 2 + 1, center=True, min_periods=window * 2 + 1).max().values
        else:
            cond = arr == s.rolling(window * 2 + 1, center=True, min_periods=window * 2 + 1).min().values
        return np.where(cond)[0]

    piv_hi = _pivots(high_p, "high", 4)
    piv_lo = _pivots(low_p, "low", 4)

    def _build_zigzag():
        pts = []
        for idx in piv_hi:
            pts.append((idx, float(high_p[idx]), "HIGH"))
        for idx in piv_lo:
            pts.append((idx, float(low_p[idx]), "LOW"))
        pts.sort(key=lambda x: x[0])
        zz = []
        for pt in pts:
            if zz and zz[-1][2] == pt[2]:
                if pt[2] == "HIGH" and pt[1] > zz[-1][1]:
                    zz[-1] = pt
                elif pt[2] == "LOW" and pt[1] < zz[-1][1]:
                    zz[-1] = pt
                continue
            zz.append(pt)
        return zz

    zz = _build_zigzag()
    if len(zz) < 5:
        return df

    def _wave_count_bull(w):
        return w[0][2] == "LOW" and w[1][2] == "HIGH" and w[2][2] == "LOW" and w[3][2] == "HIGH" and w[4][2] == "LOW"

    def _wave_count_bear(w):
        return w[0][2] == "HIGH" and w[1][2] == "LOW" and w[2][2] == "HIGH" and w[3][2] == "LOW" and w[4][2] == "HIGH"

    def _impulse_rules_bull(w):
        _, p1, _ = w[1]
        _, p2, _ = w[2]
        _, p3, _ = w[3]
        _, p4, _ = w[4]
        wave1 = abs(w[1][1] - w[0][1])
        wave2 = abs(w[2][1] - w[1][1])
        wave3 = abs(w[3][1] - w[2][1])
        wave4 = abs(w[4][1] - w[3][1])
        if wave1 == 0:
            return False
        if wave2 > wave1:
            return False
        if wave3 < wave1:
            return False
        if wave4 < wave3 and wave4 > wave2:
            return False
        if wave4 >= wave1:
            return False
        if p2 < w[0][1] or p2 > p1:
            return False
        if p4 < w[2][1] or p4 > p3:
            return False
        return True

    def _impulse_rules_bear(w):
        _, p1, _ = w[1]
        _, p2, _ = w[2]
        _, p3, _ = w[3]
        _, p4, _ = w[4]
        wave1 = abs(w[1][1] - w[0][1])
        wave2 = abs(w[2][1] - w[1][1])
        wave3 = abs(w[3][1] - w[2][1])
        wave4 = abs(w[4][1] - w[3][1])
        if wave1 == 0:
            return False
        if wave2 > wave1:
            return False
        if wave3 < wave1:
            return False
        if wave4 < wave3 and wave4 > wave2:
            return False
        if wave4 >= wave1:
            return False
        if p2 > w[0][1] or p2 < p1:
            return False
        if p4 > w[2][1] or p4 < p3:
            return False
        return True

    def _abc_rules_bull(w):
        _, p1, _ = w[1]
        _, p2, _ = w[2]
        waveA = abs(w[1][1] - w[0][1])
        waveB = abs(w[2][1] - w[1][1])
        if waveA == 0:
            return False
        if waveB > waveA * 0.886:
            return False
        if p2 > p1:
            return False
        return True

    def _abc_rules_bear(w):
        _, p1, _ = w[1]
        _, p2, _ = w[2]
        waveA = abs(w[1][1] - w[0][1])
        waveB = abs(w[2][1] - w[1][1])
        if waveA == 0:
            return False
        if waveB > waveA * 0.886:
            return False
        if p2 < p1:
            return False
        return True

    def _flat_bull(w):
        _, p1, _ = w[1]
        _, p2, _ = w[2]
        waveA = abs(w[1][1] - w[0][1])
        waveB = abs(w[2][1] - w[1][1])
        return 0.618 * waveA <= waveB <= waveA and p2 >= p1

    def _flat_bear(w):
        _, p1, _ = w[1]
        _, p2, _ = w[2]
        waveA = abs(w[1][1] - w[0][1])
        waveB = abs(w[2][1] - w[1][1])
        return 0.618 * waveA <= waveB <= waveA and p2 <= p1

    def _zigzag_bull(w):
        _, p1, _ = w[1]
        _, p2, _ = w[2]
        waveB = abs(w[2][1] - w[1][1])
        waveA = abs(w[1][1] - w[0][1])
        return waveB < waveA * 0.618 and p2 > p1

    def _zigzag_bear(w):
        _, p1, _ = w[1]
        _, p2, _ = w[2]
        waveB = abs(w[2][1] - w[1][1])
        waveA = abs(w[1][1] - w[0][1])
        return waveB < waveA * 0.618 and p2 < p1

    def _triangle_bull(w):
        _, p1, _ = w[1]
        _, p2, _ = w[2]
        p0 = w[0][1]
        p3 = w[3][1]
        p4 = w[4][1]
        return p0 < p1 and p2 < p1 and p2 < p4 and p4 < p1 and p3 > p2

    def _triangle_bear(w):
        _, p1, _ = w[1]
        _, p2, _ = w[2]
        p0 = w[0][1]
        p3 = w[3][1]
        p4 = w[4][1]
        return p0 > p1 and p2 > p1 and p2 > p4 and p4 > p1 and p3 < p2

    for i in range(4, len(zz)):
        w5 = zz[i - 4 : i + 1]
        confirm = zz[i][0] + swing_window
        if confirm >= n:
            continue

        if _wave_count_bull(w5):
            if _impulse_rules_bull(w5):
                df.loc[confirm, "ELLIOTT_WAVE"] = "IMPULSE_5_UP"
                df.loc[confirm, "ELLIOTT_TYPE"] = "BULLISH"
                df.loc[confirm, "ELLIOTT_WAVE_COUNT"] = 5
                continue
            if _abc_rules_bull(w5):
                df.loc[confirm, "ELLIOTT_WAVE"] = "ABC_CORRECTION"
                df.loc[confirm, "ELLIOTT_TYPE"] = "BULLISH"
                df.loc[confirm, "ELLIOTT_WAVE_COUNT"] = 3
                continue
            if _flat_bull(w5):
                df.loc[confirm, "ELLIOTT_WAVE"] = "FLAT_CORRECTION"
                df.loc[confirm, "ELLIOTT_TYPE"] = "BULLISH"
                df.loc[confirm, "ELLIOTT_WAVE_COUNT"] = 3
                continue
            if _zigzag_bull(w5):
                df.loc[confirm, "ELLIOTT_WAVE"] = "ZIGZAG_CORRECTION"
                df.loc[confirm, "ELLIOTT_TYPE"] = "BULLISH"
                df.loc[confirm, "ELLIOTT_WAVE_COUNT"] = 3
                continue
            if _triangle_bull(w5):
                df.loc[confirm, "ELLIOTT_WAVE"] = "TRIANGLE_CORRECTION"
                df.loc[confirm, "ELLIOTT_TYPE"] = "BULLISH"
                df.loc[confirm, "ELLIOTT_WAVE_COUNT"] = 5
                continue

        if _wave_count_bear(w5):
            if _impulse_rules_bear(w5):
                df.loc[confirm, "ELLIOTT_WAVE"] = "IMPULSE_5_DOWN"
                df.loc[confirm, "ELLIOTT_TYPE"] = "BEARISH"
                df.loc[confirm, "ELLIOTT_WAVE_COUNT"] = 5
                continue
            if _abc_rules_bear(w5):
                df.loc[confirm, "ELLIOTT_WAVE"] = "ABC_CORRECTION"
                df.loc[confirm, "ELLIOTT_TYPE"] = "BEARISH"
                df.loc[confirm, "ELLIOTT_WAVE_COUNT"] = 3
                continue
            if _flat_bear(w5):
                df.loc[confirm, "ELLIOTT_WAVE"] = "FLAT_CORRECTION"
                df.loc[confirm, "ELLIOTT_TYPE"] = "BEARISH"
                df.loc[confirm, "ELLIOTT_WAVE_COUNT"] = 3
                continue
            if _zigzag_bear(w5):
                df.loc[confirm, "ELLIOTT_WAVE"] = "ZIGZAG_CORRECTION"
                df.loc[confirm, "ELLIOTT_TYPE"] = "BEARISH"
                df.loc[confirm, "ELLIOTT_WAVE_COUNT"] = 3
                continue
            if _triangle_bear(w5):
                df.loc[confirm, "ELLIOTT_WAVE"] = "TRIANGLE_CORRECTION"
                df.loc[confirm, "ELLIOTT_TYPE"] = "BEARISH"
                df.loc[confirm, "ELLIOTT_WAVE_COUNT"] = 5
                continue

    return df


def process_phase5(symbols=None, deep=True):
    if symbols is None:
        from config import SYMBOLS
        symbols = SYMBOLS
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        for tf in ["5m","15m","1h","4h"]:
            inp = DATA_DIR / f"{symbol}_{tf}_phase4.csv"
            out = DATA_DIR / f"{symbol}_{tf}_phase5.csv"
            
            if not inp.exists():
                continue
            if out.exists() and out.stat().st_mtime > inp.stat().st_mtime:
                continue
            
            df = pd.read_csv(inp)
            df = detect_candlestick_patterns(df)
            df = detect_chart_patterns(df)
            if deep:
                df = detect_harmonic_patterns(df)
                df = detect_classic_patterns(df)
                df = detect_elliott_wave(df)
            df.to_csv(out, index=False)