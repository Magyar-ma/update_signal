import pandas as pd
import numpy as np
import logging
from pathlib import Path

import signal_journal
from phase7 import resolve_signal_state

DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_candles_since(symbol: str, tf: str, since):
    """Load every candle for ``symbol``/``tf`` that printed strictly AFTER ``since``,
    in chronological order.

    Replaces the old ``_load_latest_candles``, which only pulled the last 5-10 rows and
    then inspected a single candle (``df.iloc[-1]``). Any SL/TP that was touched between
    two tracker runs — or more than a few candles back — was silently missed. We now
    walk the full run of candles since the signal opened, so no intrabar resolution is
    skipped. When ``since`` is missing/unparseable we fall back to the full series so a
    signal is never left permanently unevaluated.
    """
    path = DATA_DIR / f"{symbol}_{tf}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if since is not None:
            since_ts = pd.to_datetime(since, errors="coerce")
            if pd.notna(since_ts):
                # IMPORTANT: strict > excludes the signal candle itself.
                df = df[df["timestamp"] > since_ts]
        return df.reset_index(drop=True)
    except Exception:
        return None


def evaluate_open_signals() -> None:
    open_signals = signal_journal.load_open_signals()
    if not open_signals:
        return

    for key, sig in list(open_signals.items()):
        plan = sig.get("plan")
        if not plan:
            continue
        symbol = sig["symbol"]
        tf = sig.get("exec_tf", "15m")
        direction = sig["direction"]
        sl = plan.get("sl")
        tp1 = plan.get("tp1")
        tp2 = plan.get("tp2")
        tp3 = plan.get("tp3")

        if any(v is None for v in [sl, tp1, tp2, tp3]):
            continue

        since = plan.get("open_time") or sig.get("timestamp")
        df = _load_candles_since(symbol, tf, since)
        if df is None or df.empty:
            continue

        end_ts = pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(hours=24)
        resolution = resolve_signal_state(df, plan, direction, end_ts)

        if resolution["status"] != "PENDING":
            exit_price = resolution["exit_price"]
            exit_reason = resolution["exit_reason"]
            exit_ts = resolution["exit_time"]
            pnl_r = resolution["pnl_r"]
            ts_str = exit_ts.isoformat() if hasattr(exit_ts, "isoformat") else str(exit_ts)
            signal_journal.close_signal(key, exit_price, exit_reason, ts_str, pnl_r=pnl_r)


def summarize_performance() -> None:
    closed = signal_journal.load_closed_signals()
    if not closed:
        logging.info("[TRACKER] No closed signals to summarize.")
        return

    total = len(closed)
    wins = sum(1 for s in closed.values() if s.get("exit_reason") in ("TP1_HIT", "TP2_HIT", "TP3_HIT"))
    losses = sum(1 for s in closed.values() if s.get("exit_reason") == "SL_HIT")
    pending = total - wins - losses

    win_rate = (wins / total * 100) if total else 0.0
    r_values = []
    for s in closed.values():
        plan = s.get("plan", {})
        entry = plan.get("entry")
        sl = plan.get("sl")
        exit_price = s.get("exit_price")
        pnl_r = s.get("pnl_r")
        if pnl_r is not None:
            r_values.append(pnl_r)
        elif all(v is not None for v in [entry, sl, exit_price]) and sl != entry:
            risk = abs(entry - sl)
            if risk > 0:
                r = (exit_price - entry) / risk if s["direction"] == "long" else (entry - exit_price) / risk
                r_values.append(r)

    avg_rr = (sum(r_values) / len(r_values)) if r_values else 0.0

    logging.info("")
    logging.info("=" * 50)
    logging.info("📊 PERFORMANCE SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Total signals : {total}")
    logging.info(f"Wins          : {wins} ({win_rate:.1f}%)")
    logging.info(f"Losses        : {losses}")
    logging.info(f"Pending       : {pending}")
    if r_values:
        logging.info(f"Avg R-multiple: {avg_rr:.2f}")
        logging.info(f"Max R         : {max(r_values):.2f}")
        logging.info(f"Min R         : {min(r_values):.2f}")
    else:
        logging.info(f"Avg R-multiple: {avg_rr:.2f}")
    logging.info("=" * 50)
