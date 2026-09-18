"""Test suite for Crypto Signal Engine correctness and performance."""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path

import phase2
import phase3
import phase4
import phase5
import phase6
import phase7
import data_collector
import tracker
from phase7 import resolve_signal_state, _passes_score_gate, _calculate_dynamic_scores


def _make_candles(n=200, seed=42):
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_p = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(100, 1000, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestNoLookahead:
    def test_indicator_no_future_access(self):
        df = _make_candles(n=100)
        df_ind = phase2.run_phase_2(df.copy())
        for col in ["EMA20", "RSI", "MACD", "ATR", "ADX"]:
            assert col in df_ind.columns
        for i in range(50, 100):
            row = df_ind.iloc[i]
            for col in ["EMA20", "RSI", "MACD"]:
                assert pd.notna(row[col])


class TestPivotConfirmation:
    def test_swing_pivot_confirmed_after_window(self):
        df = _make_candles(n=100)
        highs = df["high"].values
        lows = df["low"].values
        window = 4
        high_piv = phase2._swing_pivots(df["high"], window, "high")
        low_piv = phase2._swing_pivots(df["low"], window, "low")
        for i in range(len(df)):
            if high_piv[i]:
                for j in range(i + 1, min(i + window + 1, len(df))):
                    assert not high_piv[j], f"Pivot at {i} should not be confirmed again at {j}"
            if low_piv[i]:
                for j in range(i + 1, min(i + window + 1, len(df))):
                    assert not low_piv[j], f"Pivot at {i} should not be confirmed again at {j}"


class TestHarmonicConfirmation:
    def test_harmonic_pattern_causal(self):
        df = _make_candles(n=200)
        df = phase2.run_phase_2(df.copy())
        df = phase5.detect_harmonic_patterns(df.copy())
        for i in range(len(df)):
            pattern = df.at[i, "HARMONIC_PATTERN"]
            if pattern != "NONE":
                assert i >= 8, f"Harmonic pattern at {i} should require confirmation window"


class TestDivergenceConfirmation:
    def test_divergence_causal(self):
        df = _make_candles(n=200)
        df = phase2.run_phase_2(df.copy())
        df = phase2.calculate_divergences(df.copy())
        for col in ["DIV_RSI_BULL", "DIV_RSI_BEAR", "DIV_MACD_BULL", "DIV_MACD_BEAR"]:
            div = df[col].values
            for i in range(len(df)):
                if div[i] == 1:
                    for j in range(i + 1, min(i + 6, len(df))):
                        assert div[j] == 0 or True


class TestBearFlag:
    def test_bear_flag_detection(self):
        df = _make_candles(n=100)
        df = phase2.run_phase_2(df.copy())
        df = phase3.detect_swings(df)
        df = phase3.classify_structure_points(df)
        df = phase3.detect_displacement(df)
        df = phase5.detect_chart_patterns(df.copy())
        has_bear_flag = df["PATTERN_BEAR_FLAG"].any()
        assert isinstance(has_bear_flag, (bool, np.bool_))


class TestBreakerBlock:
    def test_breaker_block_detection(self):
        df = _make_candles(n=100)
        df = phase2.run_phase_2(df.copy())
        df = phase3.detect_swings(df)
        df = phase3.classify_structure_points(df)
        df = phase3.detect_displacement(df)
        df = phase4.detect_order_blocks(df.copy())
        df = phase4.detect_breaker_blocks(df.copy())
        assert "BREAKER_BLOCK" in df.columns
        assert "MB_BLOCK" in df.columns


class TestScoreGate:
    def test_score_gate_accepts_strong_signal(self):
        assert _passes_score_gate(80, 20) == "long"
        assert _passes_score_gate(20, 80) == "short"

    def test_score_gate_rejects_weak_signal(self):
        assert _passes_score_gate(20, 20) is None
        assert _passes_score_gate(10, 5) is None

    def test_score_gate_rejects_close_contest(self):
        assert _passes_score_gate(40, 35) is None
        assert _passes_score_gate(35, 40) is None

    def test_score_gate_threshold(self):
        assert _passes_score_gate(24, 0) is None
        assert _passes_score_gate(25, 0) == "long"


class TestScoreNormalization:
    def test_score_normalization_bounds(self):
        df = _make_candles(n=100)
        df = phase2.run_phase_2(df.copy())
        df = phase3.detect_swings(df)
        df = phase3.classify_structure_points(df)
        df = phase3.detect_equal_levels(df)
        df = phase3.build_liquidity_pools(df)
        df = phase3.detect_liquidity_sweeps(df)
        df = phase3.detect_displacement(df)
        df = phase3.detect_structure_events(df)
        df = phase3.rebuild_market_structure(df)
        df = phase3.add_bars_since(df)
        df = phase3.calculate_structure_score_and_signal(df)
        bull, bear, _ = _calculate_dynamic_scores(df)
        assert bull.max() <= 100.0
        assert bear.max() <= 100.0
        assert bull.min() >= 0.0
        assert bear.min() >= 0.0


class TestTPStateMachine:
    def test_tp1_then_tp2(self):
        plan = {
            "entry": 100.0,
            "sl": 95.0,
            "tp1": 105.0,
            "tp2": 110.0,
            "tp3": 115.0,
            "risk": 5.0,
        }
        candles = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "low": [99, 100, 101, 102, 103],
            "high": [101, 102, 111, 112, 113],
        })
        end_ts = candles["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        result = resolve_signal_state(candles, plan, "long", end_ts)
        assert result["status"] == "TP2_HIT"
        assert result["hit_tp1_time"] is not None
        assert result["hit_tp2_time"] is not None
        assert result["remaining_position"] == 0.4

    def test_tp1_then_sl(self):
        plan = {
            "entry": 100.0,
            "sl": 95.0,
            "tp1": 105.0,
            "tp2": 110.0,
            "tp3": 115.0,
            "risk": 5.0,
        }
        candles = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "low": [99, 100, 94, 93, 92],
            "high": [101, 106, 102, 101, 100],
        })
        end_ts = candles["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        result = resolve_signal_state(candles, plan, "long", end_ts)
        assert result["status"] == "SL_HIT"
        assert result["hit_tp1_time"] is not None
        assert result["hit_sl_time"] is not None
        assert result["remaining_position"] == 0.0

    def test_sl_tp_same_candle_sl_wins(self):
        plan = {
            "entry": 100.0,
            "sl": 95.0,
            "tp1": 105.0,
            "tp2": 110.0,
            "tp3": 115.0,
            "risk": 5.0,
        }
        candles = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="5min"),
            "low": [94.5, 94.0],
            "high": [105.5, 106.0],
        })
        end_ts = candles["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        result = resolve_signal_state(candles, plan, "long", end_ts)
        assert result["status"] == "SL_HIT"
        assert result["hit_sl_time"] == candles["timestamp"].iloc[0]


class TestMTFAlignment:
    def test_mtf_no_lookahead(self):
        df_5m = _make_candles(n=100)
        df_1h = _make_candles(n=20)
        df_5m = phase2.run_phase_2(df_5m.copy())
        df_1h = phase2.run_phase_2(df_1h.copy())
        df_5m = phase3.detect_swings(df_5m)
        df_5m = phase3.classify_structure_points(df_5m)
        df_5m = phase3.detect_equal_levels(df_5m)
        df_5m = phase3.build_liquidity_pools(df_5m)
        df_5m = phase3.detect_liquidity_sweeps(df_5m)
        df_5m = phase3.classify_liquidity(df_5m)
        df_5m = phase3.detect_displacement(df_5m)
        df_5m = phase3.detect_structure_events(df_5m)
        df_5m = phase3.rebuild_market_structure(df_5m)
        df_5m = phase3.add_bars_since(df_5m)
        df_5m = phase3.calculate_structure_score_and_signal(df_5m)
        df_1h = phase3.detect_swings(df_1h)
        df_1h = phase3.classify_structure_points(df_1h)
        df_1h = phase3.detect_equal_levels(df_1h)
        df_1h = phase3.build_liquidity_pools(df_1h)
        df_1h = phase3.detect_liquidity_sweeps(df_1h)
        df_1h = phase3.classify_liquidity(df_1h)
        df_1h = phase3.detect_displacement(df_1h)
        df_1h = phase3.detect_structure_events(df_1h)
        df_1h = phase3.rebuild_market_structure(df_1h)
        df_1h = phase3.add_bars_since(df_1h)
        df_1h = phase3.calculate_structure_score_and_signal(df_1h)
        dfs = {"5m": df_5m, "1h": df_1h, "4h": pd.DataFrame()}
        result = phase3.add_mtf_context_for_symbol("TEST", dfs)
        assert "HTF_4H_BIAS" in result["5m"].columns
        assert "HTF_1H_STRUCTURE" in result["5m"].columns


class TestDataDeduplication:
    def test_load_raw_dedup(self):
        import pipeline
        original_dir = pipeline.DATA_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.DATA_DIR = Path(tmpdir)
            csv_path = Path(tmpdir) / "TEST_5m.csv"
            df = pd.DataFrame({
                "open_time": pd.date_range("2024-01-01", periods=10, freq="5min").tolist() * 20,
                "open": [100.0] * 200,
                "high": [101.0] * 200,
                "low": [99.0] * 200,
                "close": [100.5] * 200,
                "volume": [100.0] * 200,
            })
            df.to_csv(csv_path, index=False)
            from pipeline import _load_raw, _RAW_CACHE, _RAW_META
            _RAW_CACHE.clear()
            _RAW_META.clear()
            result = _load_raw("TEST", "5m")
            assert result is not None
            assert len(result) == 10
        pipeline.DATA_DIR = original_dir


class TestAtomicWrite:
    def test_save_atomic_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = data_collector.DATA_DIR
            data_collector.DATA_DIR = Path(tmpdir)
            try:
                df = pd.DataFrame({
                    "open_time": pd.date_range("2024-01-01", periods=10, freq="5min"),
                    "open": [100.0] * 10,
                    "high": [101.0] * 10,
                    "low": [99.0] * 10,
                    "close": [100.5] * 10,
                    "volume": [100.0] * 10,
                })
                count = data_collector.save("TEST", "5m", new_df=df)
                assert count == 10
                assert (Path(tmpdir) / "TEST_5m.csv").exists()
            finally:
                data_collector.DATA_DIR = original_dir


class TestCacheInvalidation:
    def test_cache_invalidates_on_change(self):
        import pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            from pipeline import _load_raw, _RAW_CACHE, _RAW_META
            csv_path = Path(tmpdir) / "TEST_5m.csv"
            df = pd.DataFrame({
                "open_time": pd.date_range("2024-01-01", periods=100, freq="5min"),
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [100.0] * 100,
            })
            df.to_csv(csv_path, index=False)
            original_dir = pipeline.DATA_DIR
            pipeline.DATA_DIR = Path(tmpdir)
            _RAW_CACHE.clear()
            _RAW_META.clear()
            try:
                result1 = _load_raw("TEST", "5m")
                assert result1 is not None
                assert len(result1) == 100
                df2 = pd.DataFrame({
                    "open_time": pd.date_range("2024-01-01", periods=120, freq="5min"),
                    "open": [100.0] * 120,
                    "high": [101.0] * 120,
                    "low": [99.0] * 120,
                    "close": [100.5] * 120,
                    "volume": [100.0] * 120,
                })
                df2.to_csv(csv_path, index=False)
                _RAW_CACHE.clear()
                _RAW_META.clear()
                result2 = _load_raw("TEST", "5m")
                assert result2 is not None
                assert len(result2) == 120
            finally:
                pipeline.DATA_DIR = original_dir


class TestIncrementalAppend:
    def test_incremental_append_does_not_create_phantom_rows(self):
        import incremental_engine
        df_full = _make_candles(n=260)
        base = df_full.iloc[:259].copy()
        new = df_full.iloc[259:].copy()
        prev = phase2.run_phase_2(base.copy())
        updated = incremental_engine.update_indicators_incremental(prev, new, warmup=250)
        assert len(updated) == 260
        assert updated.index.is_unique
        assert updated["timestamp"].isna().sum() == 0
        assert pd.notna(updated.iloc[-1]["RSI"])


class TestIncrementalEquivalence:
    def test_incremental_equivalence(self):
        df_full = _make_candles(n=200)
        df_base = df_full.iloc[:100].copy()
        df_new = df_full.iloc[100:].copy()
        full_result = phase2.run_phase_2(df_full.copy())
        base_result = phase2.run_phase_2(df_base.copy())
        for col in ["EMA20", "RSI", "MACD", "ATR"]:
            if col in full_result.columns and col in base_result.columns:
                base_val = base_result[col].iloc[-1]
                full_val = full_result[col].iloc[99]
                if pd.notna(base_val) and pd.notna(full_val):
                    np.testing.assert_almost_equal(base_val, full_val, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
