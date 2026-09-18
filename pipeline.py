"""پایپلاین یکپارچه در حافظه: تمام فازها روی DataFrameهای موجود در حافظه اجرا میشوند
بدون خواندن/نوشتن مکرر CSV در طول سیکل. کش ساده برای دادههای خام بین سیکلها."""

import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import phase2, phase3, phase4, phase5, phase6, phase7
import observability
import incremental_engine as _incr

DATA_DIR = Path(__file__).resolve().parent / "data"
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
TIMEFRAMES = ["5m", "15m", "1h", "4h"]

# کش ساده برای دادههای خام بین سیکلها (تا سایز حافظه محدود باشد)
_RAW_CACHE: dict[str, pd.DataFrame] = {}
_RAW_META: dict[str, tuple[float, pd.Timestamp, int]] = {}

# کش خروجی فاز ۲ بر اساس mtime فایل خام: تا وقتی CSV خام تغییر نکند، فاز ۲ دوباره
# محاسبه نمیشود. Phase 2 (indicators) is deterministic in the raw candles, so its
# output only needs recomputing when the underlying CSV's mtime changes.
_PHASE2_CACHE: dict[str, pd.DataFrame] = {}
_PHASE2_MTIMES: dict[str, float] = {}


def _file_signature(path):
    """Cheap cache signature; never reads the CSV just to decide cache validity."""
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _phase2_cached(symbol, tf, raw_df):
    key = f"{symbol}_{tf}"
    path = DATA_DIR / f"{key}.csv"
    sig = _file_signature(path) if path.exists() else None

    if sig is not None and _PHASE2_MTIMES.get(key) == sig and key in _PHASE2_CACHE:
        return _PHASE2_CACHE[key]

    prev = _PHASE2_CACHE.get(key)

    # True incremental path: pass only candles that were appended/replaced.
    # The old implementation passed the entire raw_df as `new_candles`, which
    # forced update_indicators_incremental() to recompute the whole dataset.
    if prev is not None:
        old_len = len(prev)
        new_len = len(raw_df)
        if new_len >= old_len and old_len > 0:
            if new_len == old_len:
                # Same row count but changed last candle: recompute a small tail.
                new_candles = raw_df.tail(1).copy()
            else:
                new_candles = raw_df.iloc[old_len:].copy()
            result = _incr.update_indicators_incremental(prev, new_candles, warmup=250)
        else:
            # History was truncated/rewritten: safe cold recomputation.
            result = phase2.run_phase_2(raw_df.copy())
    else:
        result = phase2.run_phase_2(raw_df.copy())

    _PHASE2_CACHE[key] = result
    if sig is not None:
        _PHASE2_MTIMES[key] = sig
    return result


def _load_raw(symbol, tf):
    key = f"{symbol}_{tf}"
    path = DATA_DIR / f"{key}.csv"
    if not path.exists():
        return None
    try:
        sig = _file_signature(path)
        cached_sig = _RAW_META.get(key)
        if cached_sig == sig and key in _RAW_CACHE:
            return _RAW_CACHE[key]

        df = pd.read_csv(path)
        for c in REQUIRED_COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
        if len(df) < 100:
            return None

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            invalid_timestamp = df["timestamp"].isna() | (df["timestamp"].dt.year < 2000)
            if invalid_timestamp.any() and "open_time" in df.columns:
                df.loc[invalid_timestamp, "timestamp"] = pd.to_datetime(
                    df.loc[invalid_timestamp, "open_time"], errors="coerce"
                )
        elif "open_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["open_time"], errors="coerce")
        else:
            raise ValueError(f"{key}: no timestamp/open_time column")

        df = (df.sort_values("timestamp")
                .drop_duplicates(subset=["open_time"], keep="last")
                .reset_index(drop=True))
        _RAW_CACHE[key] = df
        _RAW_META[key] = sig
        return df
    except Exception as e:
        logging.warning(f"[PIPE] Failed to load {key}: {e}")
        return None


def _run_phase4(df):
    df = phase4.detect_fvg_imbalance_and_mitigation(df)
    df = phase4.detect_order_blocks(df)
    df = phase4.detect_breaker_blocks(df)
    return df


def _run_phase5(df, deep=True):
    df = phase5.detect_candlestick_patterns(df)
    df = phase5.detect_chart_patterns(df)
    if deep:
        df = phase5.detect_harmonic_patterns(df)
        df = phase5.detect_classic_patterns(df)
        df = phase5.detect_elliott_wave(df)
    return df


def _run_phase6(df):
    df = phase6.calculate_volume_delta_and_cvd(df)
    df = phase6.calculate_obv_metrics(df)
    df = phase6.calculate_vwap_metrics(df)
    df = phase6.calculate_relative_volume(df)
    df = phase6.calculate_volume_profile(df)
    df = phase6.calculate_cvd_divergence(df)
    return df


def run_symbol(symbol, metrics=None):
    """پایپلاین کامل برای یک نماد، فقط یک بار داده را از دیسک میخواند."""
    raw_by_tf = {}
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for tf in TIMEFRAMES:
        df = _load_raw(symbol, tf)
        if df is not None:
            last_ts = df["timestamp"].iloc[-1]
            age_minutes = (now_utc - last_ts).total_seconds() / 60
            tf_seconds = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}.get(tf, 60)
            max_age = tf_seconds * 3
            if age_minutes > max_age:
                logging.warning(f"[WARN] {symbol} {tf}: data is {age_minutes:.1f} min old (max {max_age} min), skipping")
                continue
            raw_by_tf[tf] = df

    if not raw_by_tf:
        logging.warning(f"[WARN] {symbol}: no fresh data available for any TF, skipping pipeline")
        return None

    if metrics:
        metrics.start_phase("phase2")
    logging.debug(f"LOG: {symbol} entering phase2 indicators")
    # فاز ۲: اندیکاتورها (کششده بر اساس mtime فایل خام)
    t2 = time.time()
    phase2_dfs = {}
    for tf, df in raw_by_tf.items():
        phase2_dfs[tf] = _phase2_cached(symbol, tf, df)
    logging.debug(f"LOG: {symbol} phase2 done")

    if metrics:
        metrics.end_phase("phase2")
        metrics.start_phase("phase3")
    logging.debug(f"LOG: {symbol} entering phase3 structure")
    # فاز ۳: ساختار بازار + MTF
    t3 = time.time()
    phase3_dfs = {}
    for tf, df in phase2_dfs.items():
        df = phase3.detect_swings(df)
        df = phase3.classify_structure_points(df)
        df = phase3.detect_equal_levels(df)
        df = phase3.build_liquidity_pools(df)
        df = phase3.detect_liquidity_sweeps(df)
        df = phase3.classify_liquidity(df)
        df = phase3.detect_displacement(df)
        df = phase3.detect_structure_events(df)
        df = phase3.rebuild_market_structure(df)
        df = phase3.calculate_premium_discount(df)
        df = phase3.add_bars_since(df)
        df = phase3.calculate_structure_score_and_signal(df)
        phase3_dfs[tf] = df
    # MTF: نیاز به همزمان all TFs دارد
    phase3_dfs = phase3.add_mtf_context_for_symbol(symbol, phase3_dfs)
    logging.debug(f"LOG: {symbol} phase3 done")

    if metrics:
        metrics.end_phase("phase3")
        metrics.start_phase("phase4")
    logging.debug(f"LOG: {symbol} entering phase4 FVG/OB")
    # فاز ۴: sequential داخل هر symbol؛ موازی‌سازی فقط در سطح symbol انجام می‌شود.
    # Nested thread pools قبلی باعث oversubscription شدید CPU می‌شد.
    t4 = time.time()
    phase4_dfs = {tf: _run_phase4(df) for tf, df in phase3_dfs.items()}
    logging.debug(f"LOG: {symbol} phase4 done")

    # FAST PATH: اگر هیچTF کاندید ساختاری ندارد، تحلیل عمیق را رد کن
    _FAST_SCORE_THRESHOLD = 15.0
    has_candidate = False
    for tf, df in phase4_dfs.items():
        if df is None or df.empty:
            continue
        if "STRUCTURE_SIGNAL" in df.columns:
            latest_signal = df["STRUCTURE_SIGNAL"].iloc[-1] if len(df) > 0 else "NEUTRAL"
            if latest_signal != "NEUTRAL":
                has_candidate = True
                break
        if "DYNAMIC_SCORE_BULL" in df.columns or "DYNAMIC_SCORE_BEAR" in df.columns:
            bull = df.get("DYNAMIC_SCORE_BULL", pd.Series([0])).iloc[-1] if len(df) > 0 else 0
            bear = df.get("DYNAMIC_SCORE_BEAR", pd.Series([0])).iloc[-1] if len(df) > 0 else 0
            if max(float(bull), float(bear)) >= _FAST_SCORE_THRESHOLD:
                has_candidate = True
                break

    deep = has_candidate

    if metrics:
        metrics.end_phase("phase4")
        metrics.start_phase("phase5")
    logging.debug(f"LOG: {symbol} entering phase5 patterns (deep={deep})")
    # فاز ۵: الگوهای کندلی + کلاسیک (هارمونیک/الیوت فقط برای کاندیدها)
    t5 = time.time()
    phase5_dfs = {tf: _run_phase5(df, deep=deep) for tf, df in phase4_dfs.items()}
    logging.debug(f"LOG: {symbol} phase5 done")

    if metrics:
        metrics.end_phase("phase5")
        metrics.start_phase("phase6")
    logging.debug(f"LOG: {symbol} entering phase6 volume")
    # فاز ۶: حجم + CVD + VWAP + Volume Profile (موازی across TFs)
    t6 = time.time()
    phase6_dfs = {tf: _run_phase6(df) for tf, df in phase5_dfs.items()}
    logging.debug(f"LOG: {symbol} phase6 done")

    # فاز ۶ دیگر هر سیکل روی دیسک نوشته نمیشود — این فایلها هر بار بازتولید میشدند
    # و توسط پایپلاین درونحافظه مصرف نمیشوند. Disabled to cut redundant disk I/O every
    # cycle; phase6 outputs flow straight into phase7 in memory. Re-enable behind a
    # debug flag if an external tool ever needs the on-disk *_phase6.csv snapshot.

    if metrics:
        metrics.end_phase("phase6")
        metrics.start_phase("phase7")
    logging.debug(f"LOG: {symbol} entering phase7 scoring")
    # فاز ۷: امتیازدهی + سیگنال (درونحافظه)
    t7 = time.time()
    try:
        from config import USE_CLOSED_CANDLES_ONLY
    except ImportError:
        USE_CLOSED_CANDLES_ONLY = False
    signals = phase7.process_phase7(
        symbols=[symbol],
        return_objects=True,
        dataframes_by_symbol={symbol: phase6_dfs},
        use_closed_candles_only=USE_CLOSED_CANDLES_ONLY,
    )

    logging.debug(f"LOG: {symbol} phase7 done, {len(signals)} signals")
    if metrics:
        metrics.end_phase("phase7")
        if signals:
            metrics.metrics.signals_generated += len(signals)
    return signals


def run_symbol_timed(symbol):
    """Process-pool entry point with process-local timing (metrics are not shared)."""
    collector = observability.MetricsCollector()
    collector.start_cycle()
    try:
        signals = run_symbol(symbol, collector)
        metrics = collector.end_cycle()
        return {"symbol": symbol, "signals": signals or [], "metrics": metrics}
    except Exception:
        collector.end_cycle()
        raise


def run_pipeline(symbols):
    """اجرای پایپلاین برای لیستی از نمادها، سریع‌ترین شکل ممکن."""
    all_signals = []
    for symbol in symbols:
        try:
            sigs = run_symbol(symbol)
            if sigs:
                all_signals.extend(sigs)
        except Exception as e:
            logging.error(f"[PIPE] {symbol} error: {e}")
    all_signals.sort(key=lambda s: s["score"], reverse=True)
    return all_signals
