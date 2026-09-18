import pandas as pd
import numpy as np

# --- Pillar weights, kept in sync with phase7._calculate_dynamic_scores ---
#   SMC (structure + order block + liquidity + FVG) : 50%
#   CVD / Volume Delta                              : 30%
#   Phase-2 indicators + classical patterns         : 20%
# Defined once here so the two scoring systems can never silently drift apart.
SMC_WEIGHT = 0.50
CVD_WEIGHT = 0.30
P2_WEIGHT = 0.20


def calculate_confidence_score(
    direction: str,
    row: pd.Series = None,
    *,
    market_structure: str = None,
    ob_direction: str = None,
    liquidity_sweep_direction: str = None,
    fvg_direction: str = None,
    fvg_active: bool = None,
    cvd_state: str = None,
    volume_delta: float = None,
    cvd_divergence_bull: int = None,
    cvd_divergence_bear: int = None,
    indicator_votes: float = None,
    trend_state: str = None,
    momentum_state: str = None,
) -> dict:
    """Weighted confidence scorer inspired by GainzAlgo V2 Alpha.

    Three confirmation pillars and their weights:
      - SMC (market structure + order block + liquidity hunter + FVG filled) : 50 %
      - CVD / Volume Delta (CVD state + delta sign + CVD divergence)         : 30 %
      - Phase-2 indicators + classical patterns                              : 20 %

    A signal is only authorised when the final score >= 75 / 100.

    Args:
        direction (str): ``'long'`` or ``'short'``.
        row (pd.Series, optional): A single candle row. When supplied, any
            missing keyword argument is read from the series (column name
            lookup is case-sensitive and matches the pipeline output).
        market_structure (str, optional): e.g. ``'BULLISH'``, ``'STRONG_BEARISH'``.
        ob_direction (str, optional): ``'BULLISH'`` / ``'BEARISH'`` from ``ACTIVE_OB_DIRECTION``.
        liquidity_sweep_direction (str, optional): ``'BULLISH'`` / ``'BEARISH'``.
        fvg_direction (str, optional): active FVG direction.
        fvg_active (bool, optional): True when an opposite FVG has been filled / mitigated.
        cvd_state (str, optional): ``'BULLISH'`` / ``'BEARISH'`` from ``CVD_STATE``.
        volume_delta (float, optional): raw ``VOLUME_DELTA`` value.
        cvd_divergence_bull (int, optional): 0/1 flag from ``CVD_DIV_BULL``.
        cvd_divergence_bear (int, optional): 0/1 flag from ``CVD_DIV_BEAR``.
        indicator_votes (float, optional): ``INDICATOR_VOTES`` value.
        trend_state (str, optional): ``'BULLISH'`` / ``'BEARISH'``.
        momentum_state (str, optional): ``'BULLISH'`` / ``'BEARISH'``.

    Returns:
        dict:
            - ``score`` (float): 0–100 confidence score.
            - ``passes`` (bool): True when score >= 75.
            - ``components`` (dict): raw and normalised sub-scores per pillar.
    """
    direction = (direction or "long").lower()
    is_long = direction == "long"

    if row is not None:
        market_structure = market_structure or row.get("STRUCTURE_SIGNAL") or row.get("MARKET_STRUCTURE")
        ob_direction = ob_direction or row.get("ACTIVE_OB_DIRECTION") or row.get("OB_DIRECTION")
        liquidity_sweep_direction = liquidity_sweep_direction or row.get("LIQUIDITY_SWEEP_DIRECTION")
        fvg_direction = fvg_direction or row.get("ACTIVE_FVG_DIRECTION") or row.get("FVG_DIRECTION")
        cvd_state = cvd_state or row.get("CVD_STATE")
        volume_delta = volume_delta if volume_delta is not None else row.get("VOLUME_DELTA")
        cvd_divergence_bull = cvd_divergence_bull if cvd_divergence_bull is not None else row.get("CVD_DIV_BULL")
        cvd_divergence_bear = cvd_divergence_bear if cvd_divergence_bear is not None else row.get("CVD_DIV_BEAR")
        indicator_votes = indicator_votes if indicator_votes is not None else row.get("INDICATOR_VOTES")
        trend_state = trend_state or row.get("TREND_STATE")
        momentum_state = momentum_state or row.get("MOMENTUM_STATE")

    def _safe(v):
        return v is not None and pd.notna(v)

    def _match(val, expected):
        return isinstance(val, str) and val.upper() == expected.upper()

    # ------------------------------------------------------------------
    # Pillar 1 – SMC  (max raw = 50  => weight 50 %)
    # ------------------------------------------------------------------
    smc_raw = 0.0

    if _safe(market_structure):
        if is_long and _match(market_structure, "BULLISH"):
            smc_raw += 15.0
        elif is_long and _match(market_structure, "STRONG_BULLISH"):
            smc_raw += 20.0
        elif not is_long and _match(market_structure, "BEARISH"):
            smc_raw += 15.0
        elif not is_long and _match(market_structure, "STRONG_BEARISH"):
            smc_raw += 20.0

    if _safe(ob_direction):
        if is_long and _match(ob_direction, "BULLISH"):
            smc_raw += 15.0
        elif not is_long and _match(ob_direction, "BEARISH"):
            smc_raw += 15.0

    if _safe(liquidity_sweep_direction):
        if is_long and _match(liquidity_sweep_direction, "BULLISH"):
            smc_raw += 10.0
        elif not is_long and _match(liquidity_sweep_direction, "BEARISH"):
            smc_raw += 10.0

    if fvg_active is True:
        smc_raw += 5.0
    elif _safe(fvg_direction):
        if is_long and _match(fvg_direction, "BULLISH"):
            smc_raw += 5.0
        elif not is_long and _match(fvg_direction, "BEARISH"):
            smc_raw += 5.0

    smc_score = max(0.0, min(100.0, (smc_raw / 50.0) * 100.0))

    # ------------------------------------------------------------------
    # Pillar 2 – CVD / Volume Delta  (max raw = 30  => weight 30 %)
    # ------------------------------------------------------------------
    cvd_raw = 0.0

    if _safe(cvd_state):
        if is_long and _match(cvd_state, "BULLISH"):
            cvd_raw += 10.0
        elif not is_long and _match(cvd_state, "BEARISH"):
            cvd_raw += 10.0

    if _safe(volume_delta):
        try:
            vd = float(volume_delta)
            if is_long and vd > 0:
                cvd_raw += 10.0
            elif not is_long and vd < 0:
                cvd_raw += 10.0
        except (TypeError, ValueError):
            pass

    if _safe(cvd_divergence_bull) and is_long and int(cvd_divergence_bull) > 0:
        cvd_raw += 10.0
    if _safe(cvd_divergence_bear) and not is_long and int(cvd_divergence_bear) > 0:
        cvd_raw += 10.0

    cvd_score = max(0.0, min(100.0, (cvd_raw / 30.0) * 100.0))

    # ------------------------------------------------------------------
    # Pillar 3 – Phase-2 indicators + classical patterns  (max raw = 20  => weight 20 %)
    # ------------------------------------------------------------------
    p2_raw = 0.0

    if _safe(indicator_votes):
        try:
            votes = float(indicator_votes)
            if is_long and votes >= 2:
                p2_raw += min(10.0, votes * 2.0)
            elif not is_long and votes <= -2:
                p2_raw += min(10.0, abs(votes) * 2.0)
        except (TypeError, ValueError):
            pass

    if _safe(trend_state):
        if is_long and _match(trend_state, "BULLISH"):
            p2_raw += 5.0
        elif not is_long and _match(trend_state, "BEARISH"):
            p2_raw += 5.0

    if _safe(momentum_state):
        if is_long and _match(momentum_state, "BULLISH"):
            p2_raw += 5.0
        elif not is_long and _match(momentum_state, "BEARISH"):
            p2_raw += 5.0

    p2_score = max(0.0, min(100.0, (p2_raw / 20.0) * 100.0))

    # ------------------------------------------------------------------
    # Weighted fusion  (each sub-score is already 0–100)
    # ------------------------------------------------------------------
    final_score = (SMC_WEIGHT * smc_score) + (CVD_WEIGHT * cvd_score) + (P2_WEIGHT * p2_score)
    final_score = round(max(0.0, min(100.0, final_score)), 2)

    return {
        "score": final_score,
        "passes": final_score >= 75.0,
        "components": {
            "smc": {
                "raw": round(smc_raw, 2),
                "normalised": round(smc_score, 2),
                "weight": SMC_WEIGHT,
            },
            "cvd": {
                "raw": round(cvd_raw, 2),
                "normalised": round(cvd_score, 2),
                "weight": CVD_WEIGHT,
            },
            "phase2_patterns": {
                "raw": round(p2_raw, 2),
                "normalised": round(p2_score, 2),
                "weight": P2_WEIGHT,
            },
        },
    }
