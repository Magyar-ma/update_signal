import pandas as pd
import numpy as np


def _first_valid_tp2(candidates, fallback, reference, is_long):
    """Return the first TP2 candidate that is a genuine extension beyond ``reference``
    (TP1) in the trade's direction — strictly greater for a long, strictly lower for a
    short — skipping any that are None, non-finite, or on the wrong side. Falls back to
    the supplied ATR-based projection when no structural candidate qualifies."""
    for level in candidates:
        if level is None:
            continue
        try:
            level = float(level)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(level):
            continue
        if is_long and level > reference:
            return level
        if not is_long and level < reference:
            return level
    return fallback


def calculate_adaptive_targets(
    trade_type: str,
    order_block_low_high: tuple,
    atr: float,
    first_opposite_fvg: float = None,
    premium_discount_boundary: float = None,
    next_major_level: float = None,
    entry: float = None,
) -> dict:
    """Dynamic TP/SL combining SMC structure with market volatility.

    Rules (non-repainting, structural):
      - Stop Loss  : exact structural OB level + ``0.5 * ATR`` safety buffer
                     away from the entry side.
      - TP1        : exact start of the first opposite FVG (fast partial close).
      - TP2        : Premium/Discount zone boundary, else next major structural
                     high/low, else ``2 * ATR`` projection.

    Args:
        trade_type (str): ``'LONG'`` or ``'SHORT'``.
        order_block_low_high (tuple): ``(low, high)`` of the structural order block
            that invalidates the trade if breached.
        atr (float): current ATR value (must be > 0).
        first_opposite_fvg (float, optional): price level of the first opposite FVG
            (BEARISH FVG top for longs, BULLISH FVG bottom for shorts).
        premium_discount_boundary (float, optional): PD_HIGH for longs, PD_LOW for
            shorts. Used as TP2 when supplied.
        next_major_level (float, optional): next swing high/low as TP2 fallback.
        entry (float, optional): current entry price. Used only for sanity checks.

    Returns:
        dict:
            - ``sl`` (float): stop-loss price.
            - ``tp1`` (float): first take-profit.
            - ``tp2`` (float): second take-profit.
            - ``risk`` (float): |entry - sl| (entry is OB-adjacent, not candle close).
            - ``buffer`` (float): applied ATR buffer.
    """
    if atr is None or not np.isfinite(atr) or atr <= 0:
        raise ValueError("ATR must be a positive finite number")

    low, high = order_block_low_high
    buffer = 0.5 * atr

    targets = {}

    if trade_type.upper() == "LONG":
        sl = low - buffer
        tp1 = first_opposite_fvg if first_opposite_fvg is not None else high + atr

        if entry is not None and sl >= entry:
            sl = entry - buffer
        if entry is not None and tp1 <= entry:
            tp1 = entry + atr

        # TP2 cascade: Premium/Discount boundary -> next major structural high ->
        # 2*ATR projection. Each candidate is only accepted when it is a real
        # extension of the move (strictly above TP1 for a long); otherwise we fall
        # through to the next source so TP2 can never collapse behind TP1.
        tp2 = _first_valid_tp2(
            candidates=(premium_discount_boundary, next_major_level),
            fallback=max(high, tp1) + 2.0 * atr,
            reference=tp1,
            is_long=True,
        )
        if tp2 <= tp1:
            tp2 = tp1 + atr

        targets["sl"] = sl
        targets["tp1"] = tp1
        targets["tp2"] = tp2

    elif trade_type.upper() == "SHORT":
        sl = high + buffer
        tp1 = first_opposite_fvg if first_opposite_fvg is not None else low - atr

        if entry is not None and sl <= entry:
            sl = entry + buffer
        if entry is not None and tp1 >= entry:
            tp1 = entry - atr

        # TP2 cascade: Premium/Discount boundary -> next major structural low ->
        # 2*ATR projection. Each candidate is only accepted when it is strictly below
        # TP1 (a genuine downside extension); otherwise we fall through to the next.
        tp2 = _first_valid_tp2(
            candidates=(premium_discount_boundary, next_major_level),
            fallback=min(low, tp1) - 2.0 * atr,
            reference=tp1,
            is_long=False,
        )
        if tp2 >= tp1:
            tp2 = tp1 - atr

        targets["sl"] = sl
        targets["tp1"] = tp1
        targets["tp2"] = tp2

    else:
        raise ValueError(f"Unsupported trade_type: {trade_type!r}. Use 'LONG' or 'SHORT'.")

    entry_ref = entry if entry is not None else (low + high) / 2.0
    risk = abs(entry_ref - targets["sl"])
    targets["risk"] = risk
    targets["buffer"] = buffer
    targets["entry"] = entry_ref

    return targets
