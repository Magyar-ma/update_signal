import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

try:
    from config import CAPITAL
except ImportError:
    CAPITAL = 1000.0

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MIN_SCORE_THRESHOLD = 25  # 0-100 weighted scale (was 35). NOTE: recalibrate after
                          # backtesting — the score scale changed with the 50/30/20 fusion.
MAX_RISK_PERCENT = 3.5
ACCOUNT_RISK_PERCENT = 1.0
MIN_RR_TP1 = 2.0
MIN_CONFLUENCE_COUNT = 2
MIN_SCORE_MARGIN = 15  # حداقل فاصله نمره صعودی/نزولی — قبلاً در config.py مستند شده بود ولی هرگز اعمال نمی‌شد (was 15)

# --- Dynamic-score fusion weights, kept in sync with confidence_scoring.py ---
#   SMC (structure + zones)                 : 50%
#   CVD / Volume Delta                      : 30%
#   Phase-2 indicators + classical patterns : 20%
SMC_WEIGHT = 0.50
CVD_WEIGHT = 0.30
P2_WEIGHT = 0.20

# Nominal per-pillar maxima used to normalise each pillar onto a 0-100 scale before
# weighting. Derived from the max contribution each _score_component_* can emit;
# STRUCTURE_SCORE is bounded at 14 by phase3.calculate_structure_score_and_signal.
_SMC_MAX = 32.0   # structure(24) + zones(8)
_CVD_MAX = 10.0   # OBV + CVD + VWAP + Volume Spike + Volume Delta
_P2_MAX = 52.0    # trend(12) + momentum(9) + patterns(22) + confluence(9)

# Signals are only drawn from the freshest candles, so score metrics are computed on
# this trailing window instead of the full history. Must be >= the Pass-1 lookback.
DYNAMIC_SCORE_WINDOW = 30

# Signal-quality: max distance (in ATR) price may sit from the active Order Block.
MAX_OB_DISTANCE_ATR = 2.5


def _passes_score_gate(bull_score, bear_score):
    """Only accept a direction when it clears BOTH the minimum absolute score AND a
    minimum margin over the opposing side. A 46-vs-45 read is a coin flip dressed up
    as a 46/100 confident signal — MIN_SCORE_MARGIN was defined and documented in
    config.py from the start but was never actually checked anywhere. Returns the
    accepted direction ('long'/'short') or None if the read is too weak or too
    contested to trust."""
    if bull_score < MIN_SCORE_THRESHOLD and bear_score < MIN_SCORE_THRESHOLD:
        return None
    if abs(bull_score - bear_score) < MIN_SCORE_MARGIN:
        return None
    return "long" if bull_score > bear_score else "short"


def _fmt(price):
    p = abs(float(price))
    if p >= 1000:
        return f"{price:,.2f}"
    if p >= 10:
        return f"{price:.3f}"
    if p >= 0.1:
        return f"{price:.5f}"
    return f"{price:.8f}"


def _pct(target, entry):
    return (target - entry) / entry * 100.0


def _dir_label(direction):
    return "🟢 LONG" if direction == "long" else "🔴 SHORT"


def _dedupe_levels(levels, min_pct_diff=0.001):
    """Remove levels within min_pct_diff of each other."""
    if not levels:
        return []
    levels = sorted(set(levels))
    result = [levels[0]]
    for l in levels[1:]:
        if abs(l - result[-1]) / result[-1] > min_pct_diff:
            result.append(l)
    return result


def _find_structural_levels(df, idx, direction, lookback=50):
    """Find all swing highs/lows before idx, categorized by side of entry."""
    start = max(0, idx - lookback)
    swing_highs, swing_lows = [], []
    for i in range(start, idx):
        r = df.loc[i]
        if r.get('SWING_HIGH') == 1 and pd.notna(r.get('SWING_HIGH_PRICE')):
            p = float(r['SWING_HIGH_PRICE'])
            if np.isfinite(p) and p > 0:
                swing_highs.append(p)
        if r.get('SWING_LOW') == 1 and pd.notna(r.get('SWING_LOW_PRICE')):
            p = float(r['SWING_LOW_PRICE'])
            if np.isfinite(p) and p > 0:
                swing_lows.append(p)

    if direction == "long":
        supports = sorted([p for p in swing_lows if p < df.at[idx, "close"]], reverse=True)
        resistances = sorted([p for p in swing_highs if p > df.at[idx, "close"]])
        return supports, resistances
    else:
        resistances = sorted([p for p in swing_highs if p > df.at[idx, "close"]])
        supports = sorted([p for p in swing_lows if p < df.at[idx, "close"]], reverse=True)
        return supports, resistances


_TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}


def resolve_signal_state(candles, plan, direction, end_ts):
    """State machine for TP1/TP2/TP3 + SL with partial position sizing.

    States: OPEN -> TP1_HIT -> TP2_HIT -> TP3_HIT (full close)
            Any state -> SL_HIT (close remaining position)

    Conservative same-candle policy: if both SL and TP are touched in the same
    candle, SL is assumed to have filled first.

    Args:
        candles: DataFrame of future candles (must include low/high/timestamp).
        plan: Trade plan dict with entry, sl, tp1, tp2, tp3, risk.
        direction: 'long' or 'short'.
        end_ts: Maximum timestamp to evaluate.

    Returns:
        dict with keys:
            status, outcome, exit_price, exit_time,
            hit_tp1_time, hit_tp2_time, hit_tp3_time, hit_sl_time,
            pnl_r, remaining_position
    """
    state = "OPEN"
    remaining = 1.0
    partial_pnl_r = 0.0
    exit_price = None
    exit_reason = None
    exit_time = None
    hit_tp1_time = None
    hit_tp2_time = None
    hit_tp3_time = None
    hit_sl_time = None

    entry = plan["entry"]
    sl = plan["sl"]
    tp1 = plan["tp1"]
    tp2 = plan["tp2"]
    tp3 = plan["tp3"]
    risk = plan["risk"]

    if risk <= 0:
        return {
            "status": "PENDING", "outcome": "PENDING", "exit_price": None, "exit_time": None,
            "hit_tp1_time": None, "hit_tp2_time": None, "hit_tp3_time": None, "hit_sl_time": None,
            "pnl_r": 0.0, "remaining_position": 1.0,
        }

    for _, check_row in candles.iterrows():
        check_ts = pd.to_datetime(check_row["timestamp"])
        if check_ts > end_ts:
            break
        low = float(check_row["low"])
        high = float(check_row["high"])

        if state == "OPEN":
            if direction == "long":
                if low <= sl:
                    partial_pnl_r += remaining * (sl - entry) / risk
                    state = "SL_HIT"
                    exit_price = sl
                    exit_reason = "SL_HIT"
                    exit_time = check_ts
                    hit_sl_time = exit_time
                    remaining = 0.0
                    break
                elif high >= tp1:
                    partial_pnl_r += 0.3 * (tp1 - entry) / risk
                    remaining = 0.7
                    state = "TP1_HIT"
                    hit_tp1_time = check_ts
            else:
                if high >= sl:
                    partial_pnl_r += remaining * (entry - sl) / risk
                    state = "SL_HIT"
                    exit_price = sl
                    exit_reason = "SL_HIT"
                    exit_time = check_ts
                    hit_sl_time = exit_time
                    remaining = 0.0
                    break
                elif low <= tp1:
                    partial_pnl_r += 0.3 * (entry - tp1) / risk
                    remaining = 0.7
                    state = "TP1_HIT"
                    hit_tp1_time = check_ts

        elif state == "TP1_HIT":
            if direction == "long":
                if low <= sl:
                    partial_pnl_r += remaining * (sl - entry) / risk
                    state = "SL_HIT"
                    exit_price = sl
                    exit_reason = "SL_HIT"
                    exit_time = check_ts
                    hit_sl_time = exit_time
                    remaining = 0.0
                    break
                elif high >= tp2:
                    partial_pnl_r += 0.3 * (tp2 - entry) / risk
                    remaining = 0.4
                    state = "TP2_HIT"
                    hit_tp2_time = check_ts
            else:
                if high >= sl:
                    partial_pnl_r += remaining * (entry - sl) / risk
                    state = "SL_HIT"
                    exit_price = sl
                    exit_reason = "SL_HIT"
                    exit_time = check_ts
                    hit_sl_time = exit_time
                    remaining = 0.0
                    break
                elif low <= tp2:
                    partial_pnl_r += 0.3 * (entry - tp2) / risk
                    remaining = 0.4
                    state = "TP2_HIT"
                    hit_tp2_time = check_ts

        elif state == "TP2_HIT":
            if direction == "long":
                if low <= sl:
                    partial_pnl_r += remaining * (sl - entry) / risk
                    state = "SL_HIT"
                    exit_price = sl
                    exit_reason = "SL_HIT"
                    exit_time = check_ts
                    hit_sl_time = exit_time
                    remaining = 0.0
                    break
                elif high >= tp3:
                    partial_pnl_r += 0.4 * (tp3 - entry) / risk
                    remaining = 0.0
                    state = "TP3_HIT"
                    exit_price = tp3
                    exit_reason = "TP3_HIT"
                    exit_time = check_ts
                    hit_tp3_time = exit_time
                    break
            else:
                if high >= sl:
                    partial_pnl_r += remaining * (entry - sl) / risk
                    state = "SL_HIT"
                    exit_price = sl
                    exit_reason = "SL_HIT"
                    exit_time = check_ts
                    hit_sl_time = exit_time
                    remaining = 0.0
                    break
                elif low <= tp3:
                    partial_pnl_r += 0.4 * (entry - tp3) / risk
                    remaining = 0.0
                    state = "TP3_HIT"
                    exit_price = tp3
                    exit_reason = "TP3_HIT"
                    exit_time = check_ts
                    hit_tp3_time = exit_time
                    break

    if state == "OPEN":
        status = "PENDING"
        outcome = "PENDING"
    elif state == "SL_HIT":
        status = "SL_HIT"
        outcome = "LOSS"
    elif state in ("TP1_HIT", "TP2_HIT", "TP3_HIT"):
        status = state
        outcome = "WIN"
    else:
        status = "PENDING"
        outcome = "PENDING"

    return {
        "status": status,
        "outcome": outcome,
        "exit_price": exit_price,
        "exit_time": exit_time,
        "hit_tp1_time": hit_tp1_time,
        "hit_tp2_time": hit_tp2_time,
        "hit_tp3_time": hit_tp3_time,
        "hit_sl_time": hit_sl_time,
        "pnl_r": round(partial_pnl_r, 3),
        "remaining_position": remaining,
    }


def _closed_state_at(df, timestamp, timeframe):
    """Return the last fully closed HTF structure available at timestamp (no lookahead:
    only candles that had actually closed by `timestamp` are considered)."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    close_after = pd.to_datetime(timestamp) - pd.Timedelta(
        minutes=_TF_MINUTES[timeframe]
    )
    candidates = df[pd.to_datetime(df["timestamp"], errors="coerce") <= close_after]
    if candidates.empty:
        return None
    state = candidates.iloc[-1].get("MARKET_STRUCTURE")
    return state if state in ("BULLISH", "BEARISH") else None


def _top_down_direction(dfs, timestamp, direction):
    """Require aligned, closed 4h/1h market structure before allowing an execution
    trigger — this is the live no-lookahead top-down gate used when generating signals
    (dfs holds the actual per-timeframe dataframes, not just precomputed HTF columns)."""
    expected = "BULLISH" if direction == "long" else "BEARISH"
    h4 = _closed_state_at(dfs.get("4h"), timestamp, "4h")
    h1 = _closed_state_at(dfs.get("1h"), timestamp, "1h")
    return h4 == expected and h1 == expected


def _recent_sweep_level(df, idx, want_direction, lookback=10):
    """Price level of the most recent liquidity sweep in want_direction (BULLISH/BEARISH),
    within `lookback` candles before idx. This is the level that must NOT be revisited if a
    sweep-reversal thesis is correct, so it's a natural, tight invalidation point."""
    if not {"LIQUIDITY_SWEEP", "LIQUIDITY_SWEEP_DIRECTION", "LIQUIDITY_SWEPT_LEVEL"} <= set(df.columns):
        return None
    start = max(0, idx - lookback)
    for i in range(idx - 1, start - 1, -1):
        if df.at[i, "LIQUIDITY_SWEEP"] == 1 and df.at[i, "LIQUIDITY_SWEEP_DIRECTION"] == want_direction:
            level = df.at[i, "LIQUIDITY_SWEPT_LEVEL"]
            if pd.notna(level):
                return float(level)
    return None


def _collect_sl_candidates(df, idx, direction, supports, resistances):
    """Real, data-backed levels whose breach genuinely invalidates the trade thesis:
    nearest opposing swing, the liquidity pool the trade is drawing from, any active
    order block / FVG acting as support or resistance, and a recent liquidity sweep
    level. A professional stop sits just beyond the closest of these — tight, but at
    a place that actually means the idea was wrong if touched."""
    row = df.iloc[idx]
    close = float(row["close"])
    candidates = []

    def _get(col):
        return row.get(col) if col in df.columns else None

    if direction == "long":
        if supports:
            candidates.append((supports[0], f"Swing Low @ {_fmt(supports[0])}"))
        sell_liq = _get("SELL_SIDE_LIQUIDITY")
        if pd.notna(sell_liq) and float(sell_liq) < close:
            candidates.append((float(sell_liq), f"Liquidity Pool @ {_fmt(sell_liq)}"))
        ob_bot = _get("ACTIVE_OB_BOTTOM")
        if pd.notna(ob_bot) and float(ob_bot) < close:
            candidates.append((float(ob_bot), f"Order Block @ {_fmt(ob_bot)}"))
        fvg_bot = _get("ACTIVE_FVG_BOTTOM")
        if pd.notna(fvg_bot) and float(fvg_bot) < close:
            candidates.append((float(fvg_bot), f"FVG @ {_fmt(fvg_bot)}"))
        sweep = _recent_sweep_level(df, idx, "BULLISH")
        if sweep is not None and sweep < close:
            candidates.append((sweep, f"Swept Low @ {_fmt(sweep)}"))
    else:
        if resistances:
            candidates.append((resistances[0], f"Swing High @ {_fmt(resistances[0])}"))
        buy_liq = _get("BUY_SIDE_LIQUIDITY")
        if pd.notna(buy_liq) and float(buy_liq) > close:
            candidates.append((float(buy_liq), f"Liquidity Pool @ {_fmt(buy_liq)}"))
        ob_top = _get("ACTIVE_OB_TOP")
        if pd.notna(ob_top) and float(ob_top) > close:
            candidates.append((float(ob_top), f"Order Block @ {_fmt(ob_top)}"))
        fvg_top = _get("ACTIVE_FVG_TOP")
        if pd.notna(fvg_top) and float(fvg_top) > close:
            candidates.append((float(fvg_top), f"FVG @ {_fmt(fvg_top)}"))
        sweep = _recent_sweep_level(df, idx, "BEARISH")
        if sweep is not None and sweep > close:
            candidates.append((sweep, f"Swept High @ {_fmt(sweep)}"))

    return candidates


def _collect_tp_candidates(df, idx, direction, supports, resistances):
    """Real, opposing-side levels price is realistically drawn toward: further swing
    points, the nearest resting liquidity pool, and any active OB/FVG acting as
    resistance (for longs) or support (for shorts). These are actual places other
    traders' stops/orders cluster — far more meaningful than an arbitrary R-multiple."""
    row = df.iloc[idx]
    close = float(row["close"])
    levels = []

    def _get(col):
        return row.get(col) if col in df.columns else None

    if direction == "long":
        levels.extend(resistances)
        buy_liq = _get("BUY_SIDE_LIQUIDITY")
        if pd.notna(buy_liq) and float(buy_liq) > close:
            levels.append(float(buy_liq))
        ob_top = _get("ACTIVE_OB_TOP")
        if pd.notna(ob_top) and float(ob_top) > close:
            levels.append(float(ob_top))
        fvg_top = _get("ACTIVE_FVG_TOP")
        if pd.notna(fvg_top) and float(fvg_top) > close:
            levels.append(float(fvg_top))
    else:
        levels.extend(supports)
        sell_liq = _get("SELL_SIDE_LIQUIDITY")
        if pd.notna(sell_liq) and float(sell_liq) < close:
            levels.append(float(sell_liq))
        ob_bot = _get("ACTIVE_OB_BOTTOM")
        if pd.notna(ob_bot) and float(ob_bot) < close:
            levels.append(float(ob_bot))
        fvg_bot = _get("ACTIVE_FVG_BOTTOM")
        if pd.notna(fvg_bot) and float(fvg_bot) < close:
            levels.append(float(fvg_bot))

    return levels


def _project_rr(entry, risk, r_mult, direction):
    return entry + risk * r_mult if direction == "long" else entry - risk * r_mult


def _regime_r_multiples(market_regime, adx):
    """Target distance (in R) scales with how much room the market is likely to give:
    a strong, confirmed trend (ADX >= 25 and MARKET_REGIME == TRENDING) can realistically
    run further before reversing; a choppy/ranging market gets taken profit on sooner
    rather than holding out for a target chop rarely reaches."""
    if market_regime == "TRENDING" and np.isfinite(adx) and adx >= 25:
        return (1.5, 3.0, 5.0)
    if market_regime == "RANGING":
        return (1.5, 2.0, 2.5)
    return (1.5, 2.2, 3.0)


def build_trade_plan(df, idx, direction):
    plan = _build_base_plan(df, idx, direction)
    if not plan:
        return None
    timestamp = df.at[idx, "timestamp"] if "timestamp" in df.columns else None
    if hasattr(timestamp, "isoformat"):
        open_time = timestamp.isoformat()
    else:
        open_time = str(timestamp) if timestamp is not None else str(idx)
    plan["open_time"] = open_time
    plan["direction"] = direction
    plan["close_time"] = None
    plan["tp1_hit_time"] = None
    plan["tp2_hit_time"] = None
    plan["tp3_hit_time"] = None
    plan["sl_hit_time"] = None
    plan["exit_price"] = None
    plan["exit_reason"] = None
    return plan


def _build_base_plan(df, idx, direction):
    close = float(df.at[idx, "close"])
    high = float(df.at[idx, "high"])
    low = float(df.at[idx, "low"])
    atr = df["ATR"].iloc[idx] if ("ATR" in df.columns and idx < len(df["ATR"])) else np.nan
    if not np.isfinite(atr) or atr <= 0:
        atr = close * 0.01

    prev_range = 0.0
    if idx > 0:
        prev_high = float(df.at[idx - 1, "high"])
        prev_low = float(df.at[idx - 1, "low"])
        prev_range = max(0.0, prev_high - prev_low)

    market_regime = df.at[idx, "MARKET_REGIME"] if "MARKET_REGIME" in df.columns else "NORMAL"
    volatility_regime = df.at[idx, "VOLATILITY_REGIME"] if "VOLATILITY_REGIME" in df.columns else "NORMAL"
    adx = float(df.at[idx, "ADX"]) if "ADX" in df.columns and pd.notna(df.at[idx, "ADX"]) else np.nan

    # Optional but important: don't fight the higher timeframes. If 4h/1h structure is
    # already computed (phase3's MTF context) and it's flatly opposed to this trade's
    # direction, any SL/TP we compute is theater — the setup itself is invalid.
    expected_bias = "BULLISH" if direction == "long" else "BEARISH"
    h4_bias = df.at[idx, "HTF_4H_BIAS"] if "HTF_4H_BIAS" in df.columns else "UNKNOWN"
    h1_struct = df.at[idx, "HTF_1H_STRUCTURE"] if "HTF_1H_STRUCTURE" in df.columns else "UNKNOWN"
    opposite_bias = "BEARISH" if direction == "long" else "BULLISH"
    if h4_bias == opposite_bias and h1_struct == opposite_bias:
        return None

    if market_regime == "TRENDING":
        sl_atr_buffer = 0.8
    elif market_regime == "RANGING":
        sl_atr_buffer = 1.5
    else:
        sl_atr_buffer = 1.2

    if volatility_regime == "HIGH":
        sl_atr_buffer = max(sl_atr_buffer, 1.5)
    elif volatility_regime == "LOW":
        sl_atr_buffer = min(sl_atr_buffer, 1.0)

    # Entry is ALWAYS the actual close price
    entry = close

    supports, resistances = _find_structural_levels(df, idx, direction, lookback=50)

    # --- Stop loss: nearest real invalidation level (swing / liquidity pool / OB / FVG /
    # sweep), not just "the first swing point found". Tighter-but-still-valid beats
    # arbitrarily distant. ---
    sl_candidates = _collect_sl_candidates(df, idx, direction, supports, resistances)
    if direction == "long":
        if sl_candidates:
            nearest_price, zone_source = max(sl_candidates, key=lambda c: c[0])
            sl = nearest_price - sl_atr_buffer * atr
        else:
            sl = entry - sl_atr_buffer * atr
            zone_source = "ATR Zone (no structure found)"
        invalidation = f"بستهشدن کندل زیر {_fmt(sl)}"
    else:
        if sl_candidates:
            nearest_price, zone_source = min(sl_candidates, key=lambda c: c[0])
            sl = nearest_price + sl_atr_buffer * atr
        else:
            sl = entry + sl_atr_buffer * atr
            zone_source = "ATR Zone (no structure found)"
        invalidation = f"بستهشدن کندل بالای {_fmt(sl)}"

    risk = abs(entry - sl)

    # Dynamic SL minimum: ensure stop gives the trade room to breathe. If the previous
    # candle range is larger than the current stop distance, widen the stop to at least
    # half the previous range so noise doesn't immediately knock it out.
    min_sl_distance = max(0.0, prev_range * 0.5)
    if 0 < risk < min_sl_distance:
        if direction == "long":
            sl = entry - min_sl_distance
        else:
            sl = entry + min_sl_distance
        risk = abs(entry - sl)

    if risk <= 0:
        return None

    # Position sizing: calculate exact units so that if SL is hit, the loss equals
    # ACCOUNT_RISK_PERCENT of CAPITAL.
    position_size = (ACCOUNT_RISK_PERCENT / 100.0 * CAPITAL) / risk

    # Reject stops that are structurally too wide instead of silently shrinking them —
    # a clamped stop no longer sits at the level the zone_source claims, which is
    # exactly what made SL/TP untrustworthy before. If the real structure demands more
    # room than our risk tolerance allows, this just isn't a tradeable setup right now.
    risk_pct = abs(_pct(sl, entry))
    if risk_pct > MAX_RISK_PERCENT or risk_pct <= 0.05:
        return None

    # --- Take profit: real opposing-side levels (further swings, resting liquidity,
    # opposing OB/FVG), ordered nearest-first. Only fall back to R-multiples for
    # whichever of TP1/TP2/TP3 has no real level behind it, and scale those multiples
    # to the current trend/volatility regime instead of a fixed 1.5/2.5/3.5 always. ---
    raw_tp_levels = _collect_tp_candidates(df, idx, direction, supports, resistances)
    tp_candidates = _dedupe_levels(raw_tp_levels, min_pct_diff=0.0015)
    tp_sources = []
    if direction == "short":
        tp_candidates = tp_candidates[::-1]  # nearest-to-entry (highest) first

    r_multiples = _regime_r_multiples(market_regime, adx)
    n_cand = len(tp_candidates)

    # Missing slots are filled by projecting forward from whatever the previous slot
    # ended up being (a real level, or an earlier projection) — never from a flat
    # R-multiple off entry alone. That guarantees the fill is always further out than
    # a real level that happens to already exceed it, and that the regime's R-spacing
    # (tight for RANGING, wide for a confirmed TRENDING move) actually shows up even
    # when only one or two real levels were found.
    if n_cand >= 3:
        tp1, tp2, tp3 = tp_candidates[0], tp_candidates[1], tp_candidates[2]
        tp_sources = [f"Level @ {_fmt(tp1)}", f"Level @ {_fmt(tp2)}", f"Level @ {_fmt(tp3)}"]
    elif n_cand == 2:
        tp1, tp2 = tp_candidates
        gap = r_multiples[2] - r_multiples[1]
        tp3 = _project_rr(tp2, risk, gap, direction)
        tp_sources = [f"Level @ {_fmt(tp1)}", f"Level @ {_fmt(tp2)}", f"+{gap:.1f}R past TP2"]
    elif n_cand == 1:
        tp1 = tp_candidates[0]
        gap1 = r_multiples[1] - r_multiples[0]
        gap2 = r_multiples[2] - r_multiples[1]
        tp2 = _project_rr(tp1, risk, gap1, direction)
        tp3 = _project_rr(tp2, risk, gap2, direction)
        tp_sources = [f"Level @ {_fmt(tp1)}", f"+{gap1:.1f}R past TP1", f"+{gap2:.1f}R past TP2"]
    else:
        tp1 = _project_rr(entry, risk, r_multiples[0], direction)
        tp2 = _project_rr(tp1, risk, r_multiples[1] - r_multiples[0], direction)
        tp3 = _project_rr(tp2, risk, r_multiples[2] - r_multiples[1], direction)
        tp_sources = [f"{r_multiples[0]:.1f}R Projection", f"{r_multiples[1]:.1f}R Projection", f"{r_multiples[2]:.1f}R Projection"]

    # Safety-net ordering only — never stretch tp1 itself to hit a minimum R:R.
    # If the nearest real target doesn't clear MIN_RR_TP1, the setup is rejected below
    # instead of inventing a further "target" with nothing behind it.
    if direction == "long":
        if tp2 <= tp1:
            tp2 = tp1 + atr
        if tp3 <= tp2:
            tp3 = tp2 + atr
    else:
        if tp2 >= tp1:
            tp2 = tp1 - atr
        if tp3 >= tp2:
            tp3 = tp2 - atr

    rr1 = abs(tp1 - entry) / risk
    if rr1 < MIN_RR_TP1:
        return None

    leverage = int(max(1, min(20, ACCOUNT_RISK_PERCENT / risk_pct * 10)))

    return {
        "entry": entry, "zone_low": low, "zone_high": high, "zone_source": zone_source,
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "risk": risk, "risk_pct": risk_pct,
        "rr1": rr1, "rr2": abs(tp2 - entry) / risk, "rr3": abs(tp3 - entry) / risk,
        "atr": atr, "trail_atr_mult": 2.0, "leverage": leverage, "invalidation": invalidation,
        "tp_sources": tp_sources, "position_size": position_size,
    }


def backtest_signals(symbol, timeframe="15m", lookback_hours=48, max_future_hours=12, step_minutes=10):
    """بکتست سیگنالهای گذشته با گام 10 دقیقهای و بررسی 12 ساعته بعد از TP3/SL.

    Args:
        symbol (str): نماد مورد نظر.
        timeframe (str): تایمفریم مورد نظر.
        lookback_hours (int): تعداد ساعات گذشته برای بررسی.
        max_future_hours (int): حداکثر ساعات آینده برای بررسی بعد از آخرین checkpoint.
        step_minutes (int): گام زمانی بررسی به دقیقه.

    Returns:
        list: نتایج بکتست برای هر سیگنال.
    """
    import pandas as pd
    from pathlib import Path

    DATA_DIR = Path(__file__).resolve().parent / "data"
    input_file = DATA_DIR / f"{symbol}_{timeframe}_phase7.csv"
    if not input_file.exists():
        return []

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logging.error(f"Error reading {input_file}: {e}")
        return []

    if len(df) < 2:
        return []

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return []

    tf_minutes = int(timeframe.replace("m", "").replace("h", "60"))
    lookback_candles = int(lookback_hours * 60 / tf_minutes)
    df = df.tail(lookback_candles).copy()

    results = []
    total_checkpoints = int((lookback_hours + max_future_hours) * 60 / step_minutes)

    for idx, row in df.iterrows():
        score_bull = float(row.get("DYNAMIC_SCORE_BULL", 0))
        score_bear = float(row.get("DYNAMIC_SCORE_BEAR", 0))
        direction = _passes_score_gate(score_bull, score_bear)
        if direction is None:
            continue
        try:
            plan = build_trade_plan(df, idx, direction)
            if not plan:
                continue
        except Exception as e:
            logging.warning(f"[BACKTEST] Error building trade plan for idx={idx}: {e}")
            continue

        signal_ts = pd.to_datetime(row["timestamp"])
        # Keep the opening timestamp local to this backtest iteration.
        # `open_time` used to be referenced in the result dict without being
        # defined in this scope, causing a NameError after the trade resolved.
        open_time = plan.get("open_time", signal_ts.isoformat())
        end_ts = signal_ts + pd.Timedelta(hours=lookback_hours + max_future_hours)
        future_mask = df["timestamp"] > signal_ts
        future_df = df.loc[future_mask].copy()
        if future_df.empty:
            continue

        resolution = resolve_signal_state(future_df, plan, direction, end_ts)
        status = resolution["status"]
        outcome = resolution["outcome"]
        exit_price = resolution["exit_price"]
        exit_time = resolution["exit_time"]
        hit_tp1_time = resolution["hit_tp1_time"]
        hit_tp2_time = resolution["hit_tp2_time"]
        hit_tp3_time = resolution["hit_tp3_time"]
        hit_sl_time = resolution["hit_sl_time"]
        pnl_r = resolution["pnl_r"]

        if status == "PENDING":
            last_valid = future_df[future_df["timestamp"] <= end_ts]
            if not last_valid.empty:
                last_check = last_valid.iloc[-1]
                exit_price = float(last_check["close"])
                exit_time = pd.to_datetime(last_check["timestamp"])
            else:
                exit_price = float(future_df.iloc[-1]["close"])
                exit_time = pd.to_datetime(future_df.iloc[-1]["timestamp"])

        # Determine final outcome
        if status in ("TP3_HIT", "TP2_HIT", "TP1_HIT"):
            outcome = "WIN"
        elif status == "SL_HIT":
            outcome = "LOSS"
        else:
            outcome = "PENDING"

        results.append({
            "timestamp": str(row["timestamp"]),
            "direction": direction,
            "score": max(score_bull, score_bear),
            "status": status,
            "outcome": outcome,
            "hit_tp1_time": str(hit_tp1_time) if hit_tp1_time else None,
            "hit_tp2_time": str(hit_tp2_time) if hit_tp2_time else None,
            "hit_tp3_time": str(hit_tp3_time) if hit_tp3_time else None,
            "hit_tp": str(hit_tp3_time) if hit_tp3_time else None,
            "hit_sl": str(hit_sl_time) if hit_sl_time else None,
            "entry": plan["entry"],
            "sl": plan["sl"],
            "tp1": plan["tp1"],
            "tp2": plan["tp2"],
            "tp3": plan["tp3"],
            "exit_price": exit_price,
            "exit_time": str(exit_time) if exit_time else None,
            "open_time": open_time,
            "close_time": str(exit_time) if exit_time else None,
            "zone_source": plan["zone_source"],
            "risk_pct": plan["risk_pct"],
            "rr1": plan["rr1"],
            "rr2": plan["rr2"],
            "rr3": plan["rr3"],
            "pnl_r": pnl_r,
            "duration_hours": round((pd.to_datetime(exit_time) - signal_ts).total_seconds() / 3600, 2) if exit_time else None,
        })

    return results


def _score_component_trend(df):
    n = len(df)
    bull = pd.Series(0.0, index=df.index)
    bear = pd.Series(0.0, index=df.index)
    if "TREND_STATE" in df.columns:
        bull += np.where(df["TREND_STATE"] == "BULLISH", 6.0, 0.0)
        bear += np.where(df["TREND_STATE"] == "BEARISH", 6.0, 0.0)
    if "MARKET_REGIME" in df.columns:
        bull += np.where(df["MARKET_REGIME"] == "TRENDING", 2.0, 0.0)
        bear += np.where(df["MARKET_REGIME"] == "TRENDING", 2.0, 0.0)
    if all(c in df.columns for c in ["EMA20", "EMA50", "EMA100", "EMA200"]):
        # EMA stack is already captured in TREND_STATE (which also requires SuperTrend
        # alignment), so we skip it here to avoid double-counting the same condition.
        pass
    if "EMA200" in df.columns and "close" in df.columns:
        bull += np.where(df["close"] > df["EMA200"], 2.0, 0.0)
        bear += np.where(df["close"] < df["EMA200"], 2.0, 0.0)
    if "ICHI_STATE" in df.columns:
        bull += np.where(df["ICHI_STATE"] == "BULLISH", 2.0, 0.0)
        bear += np.where(df["ICHI_STATE"] == "BEARISH", 2.0, 0.0)
    return bull, bear


def _score_component_momentum(df):
    n = len(df)
    bull = pd.Series(0.0, index=df.index)
    bear = pd.Series(0.0, index=df.index)
    if "MOMENTUM_STATE" in df.columns:
        bull += np.where(df["MOMENTUM_STATE"] == "BULLISH", 4.0, 0.0)
        bear += np.where(df["MOMENTUM_STATE"] == "BEARISH", 4.0, 0.0)
    if "RSI" in df.columns:
        rsi = pd.to_numeric(df["RSI"], errors="coerce").fillna(50)
        bull += np.where(rsi > 55, 2.0, 0.0)
        bear += np.where(rsi < 45, 2.0, 0.0)
    if "MACD_HIST" in df.columns:
        macd = pd.to_numeric(df["MACD_HIST"], errors="coerce").fillna(0)
        bull += np.where(macd > 0, 2.0, 0.0)
        bear += np.where(macd < 0, 2.0, 0.0)
    if "MFI" in df.columns:
        mfi = pd.to_numeric(df["MFI"], errors="coerce").fillna(50)
        bull += np.where(mfi > 55, 1.0, 0.0)
        bear += np.where(mfi < 45, 1.0, 0.0)
    return bull, bear


def _score_component_structure(df):
    n = len(df)
    bull = pd.Series(0.0, index=df.index)
    bear = pd.Series(0.0, index=df.index)
    if "STRUCTURE_SCORE" in df.columns:
        ss = pd.to_numeric(df["STRUCTURE_SCORE"], errors="coerce").fillna(0)
        bull += np.where(ss > 0, ss, 0.0)
        bear += np.where(ss < 0, -ss, 0.0)
    if "STRUCTURE_SIGNAL" in df.columns:
        bull += np.where(df["STRUCTURE_SIGNAL"] == "BULLISH", 4.0, 0.0)
        bull += np.where(df["STRUCTURE_SIGNAL"] == "STRONG_BULLISH", 8.0, 0.0)
        bear += np.where(df["STRUCTURE_SIGNAL"] == "BEARISH", 4.0, 0.0)
        bear += np.where(df["STRUCTURE_SIGNAL"] == "STRONG_BEARISH", 8.0, 0.0)
    if "MARKET_STRUCTURE" in df.columns:
        bull += np.where(df["MARKET_STRUCTURE"] == "BULLISH", 2.0, 0.0)
        bear += np.where(df["MARKET_STRUCTURE"] == "BEARISH", 2.0, 0.0)
    return bull, bear


def _score_component_zones(df):
    n = len(df)
    bull = pd.Series(0.0, index=df.index)
    bear = pd.Series(0.0, index=df.index)
    if "ACTIVE_OB_DIRECTION" in df.columns:
        bull += np.where(df["ACTIVE_OB_DIRECTION"] == "BULLISH", 4.0, 0.0)
        bear += np.where(df["ACTIVE_OB_DIRECTION"] == "BEARISH", 4.0, 0.0)
    if "ACTIVE_FVG_DIRECTION" in df.columns:
        bull += np.where(df["ACTIVE_FVG_DIRECTION"] == "BULLISH", 2.0, 0.0)
        bear += np.where(df["ACTIVE_FVG_DIRECTION"] == "BEARISH", 2.0, 0.0)
    if "ACTIVE_BREAKER_DIRECTION" in df.columns:
        bull += np.where(df["ACTIVE_BREAKER_DIRECTION"] == "BULLISH", 2.0, 0.0)
        bear += np.where(df["ACTIVE_BREAKER_DIRECTION"] == "BEARISH", 2.0, 0.0)
    return bull, bear


def _score_component_volume(df):
    n = len(df)
    bull = pd.Series(0.0, index=df.index)
    bear = pd.Series(0.0, index=df.index)
    if "OBV_STATE" in df.columns:
        bull += np.where(df["OBV_STATE"] == "BULLISH", 2.0, 0.0)
        bear += np.where(df["OBV_STATE"] == "BEARISH", 2.0, 0.0)
    if "CVD_STATE" in df.columns:
        bull += np.where(df["CVD_STATE"] == "BULLISH", 2.0, 0.0)
        bear += np.where(df["CVD_STATE"] == "BEARISH", 2.0, 0.0)
    if "VWAP_STATE" in df.columns:
        bull += np.where(df["VWAP_STATE"] == "ABOVE", 2.0, 0.0)
        bear += np.where(df["VWAP_STATE"] == "BELOW", 2.0, 0.0)
    if "VOLUME_SPIKE" in df.columns and "VOLUME_DELTA" in df.columns:
        vs = pd.to_numeric(df["VOLUME_SPIKE"], errors="coerce").fillna(0)
        vd = pd.to_numeric(df["VOLUME_DELTA"], errors="coerce").fillna(0)
        bull += np.where((vs > 0) & (vd > 0), 2.0, 0.0)
        bear += np.where((vs > 0) & (vd < 0), 2.0, 0.0)
    elif "VOLUME_SPIKE" in df.columns:
        vs = pd.to_numeric(df["VOLUME_SPIKE"], errors="coerce").fillna(0)
        bull += np.where(vs > 0, 1.0, 0.0)
        bear += np.where(vs > 0, 1.0, 0.0)
    if "VOLUME_DELTA" in df.columns:
        vd = pd.to_numeric(df["VOLUME_DELTA"], errors="coerce").fillna(0)
        bull += np.where(vd > 0, 2.0, 0.0)
        bear += np.where(vd < 0, 2.0, 0.0)
    return bull, bear


def _score_component_patterns(df):
    n = len(df)
    bull = pd.Series(0.0, index=df.index)
    bear = pd.Series(0.0, index=df.index)
    bull_cols = [
        "PATTERN_BULL_ENGULFING", "PATTERN_BULL_PINBAR",
        "PATTERN_BULL_FLAG", "PATTERN_CHANNEL_UP",
        "PATTERN_TRIANGLE_ASC",
    ]
    bear_cols = [
        "PATTERN_BEAR_ENGULFING", "PATTERN_BEAR_PINBAR",
        "PATTERN_BEAR_FLAG", "PATTERN_CHANNEL_DOWN",
        "PATTERN_TRIANGLE_DESC", "PATTERN_FALLING_WEDGE",
    ]
    for col in bull_cols:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").fillna(0)
            bull += np.where(v > 0, 2.0, 0.0)
    for col in bear_cols:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").fillna(0)
            bear += np.where(v > 0, 2.0, 0.0)
    if "HARMONIC_PATTERN" in df.columns and "HARMONIC_TYPE" in df.columns:
        htype = df["HARMONIC_TYPE"].astype(str)
        hstr = pd.to_numeric(df.get("HARMONIC_STRENGTH", 0.0), errors="coerce").fillna(0.0)
        bull += np.where((htype == "BULLISH") & (df["HARMONIC_PATTERN"] != "NONE"), 4.0 * hstr.clip(lower=0.0, upper=1.0), 0.0)
        bear += np.where((htype == "BEARISH") & (df["HARMONIC_PATTERN"] != "NONE"), 4.0 * hstr.clip(lower=0.0, upper=1.0), 0.0)
    if "ELLIOTT_WAVE" in df.columns and "ELLIOTT_TYPE" in df.columns:
        etype = df["ELLIOTT_TYPE"].astype(str)
        ecount = pd.to_numeric(df.get("ELLIOTT_WAVE_COUNT", 0), errors="coerce").fillna(0)
        bull += np.where((etype == "BULLISH") & (df["ELLIOTT_WAVE"] != "NONE"), 3.0 + ecount.clip(lower=0.0, upper=5.0), 0.0)
        bear += np.where((etype == "BEARISH") & (df["ELLIOTT_WAVE"] != "NONE"), 3.0 + ecount.clip(lower=0.0, upper=5.0), 0.0)
    return bull, bear


def _score_component_confluence(df):
    """Cross-indicator consensus vote from phase2's calculate_confluence_state — combines
    TREND_STATE, MOMENTUM_STATE, Ichimoku, EMA200 position, ADX/DI, and divergence into a
    single -6..+6 vote. This was computed on every candle from the start but never actually
    fed into the score, so a signal where every one of these independently agreed got no
    more credit than one where they were split down the middle."""
    bull = pd.Series(0.0, index=df.index)
    bear = pd.Series(0.0, index=df.index)
    if "INDICATOR_VOTES" in df.columns:
        votes = pd.to_numeric(df["INDICATOR_VOTES"], errors="coerce").fillna(0)
        bull += np.where(votes > 0, votes * 1.5, 0.0)
        bear += np.where(votes < 0, -votes * 1.5, 0.0)
    return bull, bear


def _calculate_dynamic_scores(df, window=DYNAMIC_SCORE_WINDOW):
    """Weighted 0-100 directional scores using the 50/30/20 SMC/CVD/Phase-2 split
    shared with confidence_scoring.py.

    Performance: metrics are computed only on the last ``window`` candles (localized
    indexing on a single tail slice). Signals are only ever drawn from the most recent
    handful of bars, so re-scoring the full multi-hundred-row history every cycle was
    pure waste. Rows older than the window are returned as 0.0.

    Returns:
        tuple: ``(dynamic_bull, dynamic_bear, components)`` where ``dynamic_*`` are the
        full-length 0-100 weighted scores and ``components`` maps each raw
        per-side component series (full-length, original index) for CSV export.
    """
    empty_pairs = ("trend", "momentum", "structure", "zones", "volume", "patterns", "confluence")
    full_bull = pd.Series(0.0, index=df.index)
    full_bear = pd.Series(0.0, index=df.index)
    components = {k: (pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)) for k in empty_pairs}

    n = len(df)
    if n == 0:
        return full_bull, full_bear, components

    # Localized indexing: operate on the trailing window only.
    win = df.tail(min(int(window), n))
    win_index = win.index

    t_bull, t_bear = _score_component_trend(win)
    m_bull, m_bear = _score_component_momentum(win)
    s_bull, s_bear = _score_component_structure(win)
    z_bull, z_bear = _score_component_zones(win)
    v_bull, v_bear = _score_component_volume(win)
    p_bull, p_bear = _score_component_patterns(win)
    c_bull, c_bear = _score_component_confluence(win)

    # --- Group raw components into the three weighted pillars ---
    smc_bull = s_bull + z_bull            # SMC: market structure + OB/FVG/breaker zones
    smc_bear = s_bear + z_bear
    cvd_bull = v_bull                     # CVD / Volume Delta pillar
    cvd_bear = v_bear
    p2_bull = t_bull + m_bull + p_bull + c_bull   # Phase-2 indicators + classical patterns
    p2_bear = t_bear + m_bear + p_bear + c_bear

    def _norm(series, max_val):
        return (series / max_val * 100.0).clip(lower=0.0, upper=100.0)

    win_bull = (SMC_WEIGHT * _norm(smc_bull, _SMC_MAX)
                + CVD_WEIGHT * _norm(cvd_bull, _CVD_MAX)
                + P2_WEIGHT * _norm(p2_bull, _P2_MAX)).clip(lower=0.0, upper=100.0)
    win_bear = (SMC_WEIGHT * _norm(smc_bear, _SMC_MAX)
                + CVD_WEIGHT * _norm(cvd_bear, _CVD_MAX)
                + P2_WEIGHT * _norm(p2_bear, _P2_MAX)).clip(lower=0.0, upper=100.0)

    full_bull.loc[win_index] = win_bull
    full_bear.loc[win_index] = win_bear

    def _place(bull, bear):
        fb = pd.Series(0.0, index=df.index); fb.loc[win_index] = bull
        fr = pd.Series(0.0, index=df.index); fr.loc[win_index] = bear
        return fb, fr

    components["trend"] = _place(t_bull, t_bear)
    components["momentum"] = _place(m_bull, m_bear)
    components["structure"] = _place(s_bull, s_bear)
    components["zones"] = _place(z_bull, z_bear)
    components["volume"] = _place(v_bull, v_bear)
    components["patterns"] = _place(p_bull, p_bear)
    components["confluence"] = _place(c_bull, c_bear)

    return full_bull, full_bear, components


def _passes_smc_filters(df, idx, direction):
    """Signal-quality SMC gates applied at the candidate candle:

      1. OB proximity — price must sit within ``MAX_OB_DISTANCE_ATR * ATR`` of the
         active Order Block zone (distance 0 when price is inside the zone). Chasing a
         setup far from its OB is where late, low-quality entries come from.
      2. Opposing FVG — reject when an active Fair Value Gap points against the trade
         (a BEARISH active FVG under a long, or a BULLISH one under a short): an
         unfilled opposite imbalance is an obvious magnet against the position.

    A gate is skipped only when its underlying data is genuinely absent (NaN); bad
    data is never silently treated as a pass. Returns True when the signal survives."""
    row = df.iloc[idx]
    close = float(row["close"])

    atr = row.get("ATR")
    try:
        atr = float(atr)
    except (TypeError, ValueError):
        atr = np.nan
    if not np.isfinite(atr) or atr <= 0:
        atr = close * 0.01

    # 1) Proximity to the active Order Block
    ob_top = row.get("ACTIVE_OB_TOP")
    ob_bot = row.get("ACTIVE_OB_BOTTOM")
    if pd.notna(ob_top) and pd.notna(ob_bot):
        lo, hi = sorted((float(ob_top), float(ob_bot)))
        if close < lo:
            dist = lo - close
        elif close > hi:
            dist = close - hi
        else:
            dist = 0.0
        if dist > MAX_OB_DISTANCE_ATR * atr:
            return False

    # 2) Active opposing FVG
    fvg_dir = row.get("ACTIVE_FVG_DIRECTION")
    if isinstance(fvg_dir, str) and fvg_dir:
        opposing = "BEARISH" if direction == "long" else "BULLISH"
        if fvg_dir.upper() == opposing:
            return False

    return True


def _build_reasons(row, direction):
    reasons = []
    bullish_internal = "BULLISH" if direction == "long" else "BEARISH"
    bearish_internal = "BEARISH" if direction == "long" else "BULLISH"
    if "STRUCTURE_SIGNAL" in row and row["STRUCTURE_SIGNAL"] in (
        "BULLISH", "STRONG_BULLISH", "BEARISH", "STRONG_BEARISH"
    ):
        reasons.append("Market Structure")
    if "TREND_STATE" in row and row["TREND_STATE"] == bullish_internal:
        reasons.append("Trend State")
    if "MOMENTUM_STATE" in row and row["MOMENTUM_STATE"] == bullish_internal:
        reasons.append("Momentum")
    if "ACTIVE_OB_DIRECTION" in row and row["ACTIVE_OB_DIRECTION"] == bullish_internal:
        reasons.append("Order Block")
    if "ACTIVE_FVG_DIRECTION" in row and row["ACTIVE_FVG_DIRECTION"] == bullish_internal:
        reasons.append("Active FVG")
    if "SUPERTREND_DIRECTION" in row:
        st = row["SUPERTREND_DIRECTION"]
        if direction == "long" and st == 1:
            reasons.append("SuperTrend")
        elif direction == "short" and st == -1:
            reasons.append("SuperTrend")
    if "ADX" in row and "PLUS_DI" in row and "MINUS_DI" in row:
        adx = float(row.get("ADX", 0) or 0)
        if adx >= 25:
            plus_di = float(row.get("PLUS_DI", 0) or 0)
            minus_di = float(row.get("MINUS_DI", 0) or 0)
            if direction == "long" and plus_di > minus_di:
                reasons.append("ADX/DI")
            elif direction == "short" and minus_di > plus_di:
                reasons.append("ADX/DI")
    if "VWAP_STATE" in row and row["VWAP_STATE"] == ("ABOVE" if direction == "long" else "BELOW"):
        reasons.append("VWAP")
    if "OBV_STATE" in row and row["OBV_STATE"] == bullish_internal:
        reasons.append("OBV")
    if "CVD_STATE" in row and row["CVD_STATE"] == bullish_internal:
        reasons.append("CVD")
    if "HARMONIC_PATTERN" in row and row.get("HARMONIC_PATTERN") not in (None, "NONE"):
        reasons.append(f"Harmonic {row.get('HARMONIC_PATTERN')}")
    if "ELLIOTT_WAVE" in row and row.get("ELLIOTT_WAVE") not in (None, "NONE"):
        reasons.append(f"Elliott {row.get('ELLIOTT_WAVE')}")
    if "VOLUME_SPIKE" in row and float(row.get("VOLUME_SPIKE", 0) or 0) > 0:
        reasons.append("Volume Spike")
    if "INDICATOR_VOTES" in row:
        votes = float(row.get("INDICATOR_VOTES", 0) or 0)
        if (direction == "long" and votes >= 4) or (direction == "short" and votes <= -4):
            reasons.append("Multi-Indicator Confluence")
    if not reasons:
        reasons.append("Composite Score")
    return reasons


def _format_card(symbol, direction, score, timestamp, plan, reasons, exec_tf, agreeing=None):
    label = _dir_label(direction)
    agreeing_str = ", ".join(agreeing) if agreeing else exec_tf
    open_time = plan.get("open_time", timestamp)
    tp1_time = plan.get("tp1_hit_time", "—")
    tp2_time = plan.get("tp2_hit_time", "—")
    tp3_time = plan.get("tp3_hit_time", "—")
    sl_time = plan.get("sl_hit_time", "—")
    close_time = plan.get("close_time", "—")
    exit_price = plan.get("exit_price")
    tp_sources = plan.get("tp_sources") or ["-", "-", "-"]
    lines = [
        f"{'='*55}",
        f"📊 {symbol} | {label}",
        f"{'='*55}",
        f"🎯 Score : {score:.1f}/100 🌟",
        f"⏱ TF     : {agreeing_str}",
        f"🕒 Signal : {timestamp}",
        f"🕐 Open   : {open_time}",
        f"",
        f"💰 Entry  : {_fmt(plan['entry'])} 📍",
        f"📈 R:R    = {plan['rr1']:.1f} | ⚠️ Risk: {plan['risk_pct']:.2f}% | Lev: {plan.get('leverage', '-')}",
        f"📍 Zone   : {plan['zone_source']}",
        f"🔗 Invalid: {plan['invalidation']}",
        f"",
        f"✅ TP1  {_fmt(plan['tp1']):>15} 🎯 | {tp1_time}  ({tp_sources[0]})",
        f"✅ TP2  {_fmt(plan['tp2']):>15} 🎯 | {tp2_time}  ({tp_sources[1]})",
        f"✅ TP3  {_fmt(plan['tp3']):>15} 🎯 | {tp3_time}  ({tp_sources[2]})",
        f"🛑 SL    {_fmt(plan['sl']):>15} ❌ | {sl_time}",
        f"🏁 Close {_fmt(exit_price) if exit_price else '—':>15}   | {close_time}",
        f"",
        f"📌 Reasons:",
    ]
    for r in reasons:
        lines.append(f"   • {r}")
    lines.append(f"{'='*55}")
    return "\n".join(lines)


def process_phase7(symbols=None, return_objects=True, dataframes_by_symbol=None, use_closed_candles_only=None):
    if symbols is None:
        from config import SYMBOLS
        symbols = SYMBOLS

    if use_closed_candles_only is None:
        try:
            from config import USE_CLOSED_CANDLES_ONLY
            use_closed_candles_only = USE_CLOSED_CANDLES_ONLY
        except ImportError:
            use_closed_candles_only = False

    if dataframes_by_symbol is None:
        dataframes_by_symbol = {}

    DATA_DIR = Path(__file__).resolve().parent / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_signals = []

    for symbol in symbols:
        dfs = dataframes_by_symbol.get(symbol, {})
        if not dfs:
            continue

        scored_dfs = {}
        for tf, df in dfs.items():
            if df is None or df.empty:
                continue
            df = df.copy()
            bull, bear, comp = _calculate_dynamic_scores(df)
            df["SCORE_TREND_BULL"], df["SCORE_TREND_BEAR"] = comp["trend"]
            df["SCORE_MOMENTUM_BULL"], df["SCORE_MOMENTUM_BEAR"] = comp["momentum"]
            df["SCORE_STRUCTURE_BULL"], df["SCORE_STRUCTURE_BEAR"] = comp["structure"]
            df["SCORE_ZONES_BULL"], df["SCORE_ZONES_BEAR"] = comp["zones"]
            df["SCORE_VOLUME_BULL"], df["SCORE_VOLUME_BEAR"] = comp["volume"]
            df["SCORE_PATTERNS_BULL"], df["SCORE_PATTERNS_BEAR"] = comp["patterns"]
            df["SCORE_CONFLUENCE_BULL"], df["SCORE_CONFLUENCE_BEAR"] = comp["confluence"]
            df["DYNAMIC_SCORE_BULL"] = bull
            df["DYNAMIC_SCORE_BEAR"] = bear
            scored_dfs[tf] = df

        for tf, df in scored_dfs.items():
            out = DATA_DIR / f"{symbol}_{tf}_phase7.csv"
            df.to_csv(out, index=False)

        if not return_objects:
            continue

        # Pass 1: find each timeframe's own best candidate independently (no sending
        # yet). Previously each timeframe was sent as its own separate signal — a
        # symbol could get a 5m LONG and a 15m SHORT in the same cycle, or two
        # near-identical entries that only survived notifier's dedup by accident of
        # which one sorted first. That's "one signal per currency" done wrong: single
        # slices of data going out the door independently instead of being weighed
        # against each other first.
        tf_candidates = {}
        for tf, df in scored_dfs.items():
            if tf not in ("5m", "15m"):
                continue
            n = len(df)
            lookback = min(12, n)  # Pass-1 lookback broadened to 12
            best_sig = None
            best_score = 0

            if use_closed_candles_only:
                last_valid_idx = n - 2
                if last_valid_idx < 0:
                    continue
                best_idx = last_valid_idx
                idx_range = range(max(0, n - lookback - 1), last_valid_idx + 1)
            else:
                best_idx = n - 1
                idx_range = range(n - lookback, n)

            for idx in idx_range:
                bull_score = float(df.at[idx, "DYNAMIC_SCORE_BULL"]) if "DYNAMIC_SCORE_BULL" in df.columns else 0
                bear_score = float(df.at[idx, "DYNAMIC_SCORE_BEAR"]) if "DYNAMIC_SCORE_BEAR" in df.columns else 0
                direction = _passes_score_gate(bull_score, bear_score)
                if direction is None:
                    continue
                # Trend-regime gate: skip RANGING only; TRENDING and TRANSITIONAL
                # are accepted. Applied immediately after the score gate.
                if "MARKET_REGIME" in df.columns and df.at[idx, "MARKET_REGIME"] == "RANGING":
                    continue
                # Signal-quality SMC gates: OB proximity + reject opposing active FVG.
                if not _passes_smc_filters(df, idx, direction):
                    continue
                score = max(bull_score, bear_score)
                try:
                    signal_time = df.at[idx, "timestamp"]
                    if not _top_down_direction(dfs, signal_time, direction):
                        continue
                    plan = build_trade_plan(df, idx, direction)
                    if not plan:
                        continue
                    age = best_idx - idx
                    recency_weight = 1.0 - (age / (lookback + 1)) * 0.5
                    adjusted_score = score * recency_weight
                    if adjusted_score > best_score:
                        best_sig = (idx, direction, score, plan)
                        best_score = adjusted_score
                        best_idx = idx
                except Exception as e:
                    logging.debug(f"[PHASE7] Candidate skip idx={idx} dir={direction}: {e}")
                    continue
            if best_sig is not None:
                tf_candidates[tf] = {"df": df, **dict(zip(("idx", "direction", "score", "plan"), best_sig))}

        # Pass 2: real confluence — integrate the two timeframes' reads instead of
        # sending either in isolation. MIN_CONFLUENCE_COUNT was defined and documented
        # from the start ("حداقل تعداد تایمفریم همراستا") but never actually enforced;
        # any single timeframe's noise could go out as a full-confidence signal. Now,
        # require at least MIN_CONFLUENCE_COUNT execution timeframes to independently
        # agree on the same direction before anything is sent for this symbol.
        by_direction = {}
        for tf, cand in tf_candidates.items():
            by_direction.setdefault(cand["direction"], []).append(tf)

        for direction, agreeing_tfs in by_direction.items():
            if len(agreeing_tfs) < MIN_CONFLUENCE_COUNT:
                continue  # only one timeframe saw it — not enough corroboration to trust

            # Use the timeframe with the strongest (recency-weighted) read as the basis
            # for the actual trade plan/entry, but credit every agreeing timeframe.
            primary_tf = max(agreeing_tfs, key=lambda t: tf_candidates[t]["score"])
            primary = tf_candidates[primary_tf]
            idx, plan, df = primary["idx"], primary["plan"], primary["df"]

            # Small, capped confluence bonus — being right on two timeframes is more
            # trustworthy than being right on one, but shouldn't let a mediocre score
            # masquerade as a top signal.
            score = min(100.0, primary["score"] + 5.0 * (len(agreeing_tfs) - 1))

            timestamp = df.at[idx, "timestamp"] if "timestamp" in df.columns else str(idx)
            if hasattr(timestamp, "isoformat"):
                timestamp = timestamp.isoformat()
            elif not isinstance(timestamp, str):
                timestamp = str(timestamp)
            reasons = _build_reasons(df.iloc[idx].to_dict(), direction)
            agreeing_sorted = sorted(agreeing_tfs, key=lambda t: {"5m": 0, "15m": 1}.get(t, 9))
            card = _format_card(symbol, direction, score, timestamp, plan, reasons, primary_tf, agreeing=agreeing_sorted)
            all_signals.append({
                "symbol": symbol,
                "direction": direction,
                "score": score,
                "exec_tf": primary_tf,
                "agreeing": agreeing_sorted,
                "plan": plan,
                "timestamp": timestamp,
                "reasons": reasons,
                "card": card,
            })

    all_signals.sort(key=lambda s: s["score"], reverse=True)
    return all_signals