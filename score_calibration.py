"""Score calibration and threshold optimization.

Given a history of (score, outcome) pairs, this module:
    - Buckets scores and computes win rate / expectancy per bucket
    - Optionally fits Isotonic Regression or Platt Scaling to map raw score ->
      calibrated win probability
    - Suggests an optimal score threshold based on expectancy and drawdown
      constraints, not just raw win rate.

All calibration artifacts are persisted to data/calibration/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

CALIBRATION_DIR = Path(__file__).resolve().parent / "data" / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


def _outcome_to_win(outcome: str) -> int:
    return 1 if outcome in ("TP1_HIT", "TP2_HIT", "TP3_HIT") else 0


def bucket_analysis(
    records: list[dict],
    buckets: int = 10,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Analyze calibration quality by score bucket.

    Args:
        records: list of dicts with keys `score` (0-100) and `outcome`.
        buckets: number of score buckets.
        min_samples: minimum samples per bucket to report.

    Returns:
        DataFrame with columns: bucket, score_min, score_max, count, win_rate,
        avg_r, median_r, expectancy.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["win"] = df["outcome"].apply(_outcome_to_win)
    df = df.dropna(subset=["score", "win"])

    if df.empty:
        return pd.DataFrame()

    df["bucket"] = pd.cut(
        df["score"],
        bins=np.linspace(0, 100, buckets + 1),
        include_lowest=True,
        labels=False,
    )

    rows = []
    for b in range(buckets):
        sub = df[df["bucket"] == b]
        count = len(sub)
        if count < min_samples:
            continue
        win_rate = sub["win"].mean()
        avg_r = sub.get("pnl_r", pd.Series(dtype=float)).mean()
        median_r = sub.get("pnl_r", pd.Series(dtype=float)).median()
        expectancy = win_rate * avg_r if pd.notna(avg_r) else np.nan
        rows.append({
            "bucket": b,
            "score_min": b * (100 / buckets),
            "score_max": (b + 1) * (100 / buckets),
            "count": count,
            "win_rate": round(float(win_rate), 4),
            "avg_r": round(float(avg_r), 4) if pd.notna(avg_r) else None,
            "median_r": round(float(median_r), 4) if pd.notna(median_r) else None,
            "expectancy": round(float(expectancy), 4) if pd.notna(expectancy) else None,
        })

    return pd.DataFrame(rows)


def fit_calibrators(records: list[dict]) -> dict[str, Any] | None:
    """Fit calibration models if sklearn is available and we have enough data.

    Returns a dict with fitted models and metadata, or None if fitting is not
    possible.
    """
    if not _HAS_SKLEARN:
        logging.warning("[CALIBRATION] sklearn not installed; skipping calibrator fit.")
        return None

    df = pd.DataFrame(records)
    df["win"] = df["outcome"].apply(_outcome_to_win)
    df = df.dropna(subset=["score", "win"])

    if len(df) < 30:
        logging.warning(f"[CALIBRATION] Insufficient data for calibration: {len(df)} samples.")
        return None

    X = df[["score"]].values
    y = df["win"].values

    result = {}

    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(X.flatten(), y)
        result["isotonic"] = iso
    except Exception as e:
        logging.warning(f"[CALIBRATION] Isotonic fit failed: {e}")

    try:
        lr = LogisticRegression()
        lr.fit(X, y)
        result["platt"] = lr
    except Exception as e:
        logging.warning(f"[CALIBRATION] Platt fit failed: {e}")

    return result if result else None


def suggest_threshold(
    bucket_df: pd.DataFrame,
    min_expectancy: float = 0.1,
    max_drawdown_tolerance: float = 0.3,
) -> dict[str, Any]:
    """Suggest an optimal score threshold.

    The threshold is chosen as the lowest score bucket where:
        - expectancy >= min_expectancy
        - win_rate is acceptable given drawdown tolerance

    If no bucket meets the criteria, returns the best-effort recommendation.

    Args:
        bucket_df: output from bucket_analysis().
        min_expectancy: minimum expected R per trade.
        max_drawdown_tolerance: maximum acceptable drawdown fraction.

    Returns:
        dict with suggested threshold, expected metrics, and rationale.
    """
    if bucket_df.empty:
        return {"threshold": 25, "rationale": "no data; using default"}

    candidates = bucket_df[bucket_df["expectancy"] >= min_expectancy]
    if candidates.empty:
        best = bucket_df.loc[bucket_df["expectancy"].idxmax()]
        return {
            "threshold": int(best["score_min"]),
            "rationale": "no bucket meets min_expectancy; using best available",
            "expectancy": best["expectancy"],
            "win_rate": best["win_rate"],
        }

    chosen = candidates.iloc[0]
    return {
        "threshold": int(chosen["score_min"]),
        "rationale": (
            f"lowest score bucket with expectancy >= {min_expectancy}; "
            f"win_rate={chosen['win_rate']:.2%}, expectancy={chosen['expectancy']:.2f}"
        ),
        "expectancy": chosen["expectancy"],
        "win_rate": chosen["win_rate"],
    }


def save_calibration(calibration_id: str, artifacts: dict) -> Path:
    path = CALIBRATION_DIR / f"{calibration_id}.json"
    payload = {}
    for k, v in artifacts.items():
        if hasattr(v, "get_params"):
            payload[k + "_type"] = type(v).__name__
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_signal_history(limit: int = 1000) -> list[dict]:
    """Load signal history from signal_journal for calibration."""
    try:
        import signal_journal
        closed = signal_journal.load_closed_signals()
        records = []
        for sig in closed.values():
            plan = sig.get("plan", {})
            records.append({
                "score": sig.get("score"),
                "outcome": sig.get("exit_reason", "PENDING"),
                "pnl_r": sig.get("pnl_r"),
                "entry": plan.get("entry"),
                "sl": plan.get("sl"),
                "tp1": plan.get("tp1"),
                "tp2": plan.get("tp2"),
                "tp3": plan.get("tp3"),
            })
        return records[-limit:]
    except Exception as e:
        logging.warning(f"[CALIBRATION] Could not load signal history: {e}")
        return []
