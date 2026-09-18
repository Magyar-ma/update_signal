import pandas as pd
from pathlib import Path
import logging
import config
import phase2, phase3, phase4, phase5, phase6, phase7

DATA_DIR = Path("E:/PRO/trade/data")
symbol = "ACEUSDT"
raw_by_tf = {}
for tf in ["5m", "15m", "1h", "4h"]:
    f = DATA_DIR / f"{symbol}_{tf}.csv"
    if f.exists():
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        raw_by_tf[tf] = df

phase2_dfs = {tf: phase2.run_phase_2(df.copy()) for tf, df in raw_by_tf.items()}

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

phase3_dfs = phase3.add_mtf_context_for_symbol(symbol, phase3_dfs)

# Check HTF bias for 5m indices 1492-1499
df5 = scored_dfs = None
for tf, df in phase3_dfs.items():
    if tf == "5m":
        df5 = df
        break

if df5 is not None:
    logging.info("5m HTF bias and structure for idx 1492-1499:")
    for idx in range(1492, 1500):
        if idx < len(df5):
            ts = df5.at[idx, "timestamp"]
            h4 = df5.at[idx, "HTF_4H_BIAS"] if "HTF_4H_BIAS" in df5.columns else "N/A"
            h1 = df5.at[idx, "HTF_1H_STRUCTURE"] if "HTF_1H_STRUCTURE" in df5.columns else "N/A"
            regime = df5.at[idx, "MARKET_REGIME"] if "MARKET_REGIME" in df5.columns else "N/A"
            logging.info(f"  idx={idx} ts={ts} h4={h4} h1={h1} regime={regime}")
