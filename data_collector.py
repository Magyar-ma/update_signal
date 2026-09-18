import asyncio
import sys
import time
import os
import logging
import aiohttp
import pandas as pd
import requests
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    from config import SYMBOLS, TIMEFRAMES, UPDATE_SECONDS, MAX_CANDLES_PER_TF, CHECK_INTERVAL_MINUTES
except ImportError:
    SYMBOLS = ["BTC-SWAP-USDT"]
    TIMEFRAMES = ["5m", "15m", "1h", "4h"]
    UPDATE_SECONDS = 120
    MAX_CANDLES_PER_TF = {"5m": 1500, "15m": 1000, "1h": 500, "4h": 200}
    CHECK_INTERVAL_MINUTES = {"5m": 0, "15m": 7, "1h": 20, "4h": 60}

BASE_URL = "https://api.toobit.com"
DATA_DIR = Path("data")
TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}

def _parse_klines(data):
    if not data or not isinstance(data, list):
        raise ValueError("Invalid API response")
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume", "trades",
        "taker_base_volume", "taker_quote_volume"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_localize(None)
    # Keep the established candle-open timestamp. Some exchange responses expose
    # close_time in seconds, which would otherwise parse as 1970-01-01.
    df["timestamp"] = df["open_time"]
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_base_volume", "taker_quote_volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["open_time", "close_time", "close"]).sort_values(
        "close_time"
    ).reset_index(drop=True)


async def download_async(session, symbol, interval, limit):
    url = f"{BASE_URL}/quote/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for attempt in range(3):
        try:
            async def _fetch():
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    response.raise_for_status()
                    data = await response.json()
                if not data or not isinstance(data, list):
                    raise ValueError(f"Invalid API response for {symbol} {interval}")
                df = _parse_klines(data)
                if df is None or df.empty:
                    raise ValueError(f"Empty data returned for {symbol} {interval}")
                return df

            return await asyncio.wait_for(_fetch(), timeout=25)
        except asyncio.TimeoutError:
            if attempt < 2:
                sleep = 2 ** attempt
                logging.warning(f"[TIMEOUT] {symbol} {interval}: timeout after 25s, retrying in {sleep}s...")
                await asyncio.sleep(sleep)
            else:
                logging.error(f"[ERROR] Download failed for {symbol} {interval} after 3 attempts: timeout")
                raise
        except Exception as e:
            if attempt < 2:
                sleep = 2 ** attempt
                logging.warning(f"[RETRY] {symbol} {interval}: {e}, retrying in {sleep}s...")
                await asyncio.sleep(sleep)
            else:
                logging.error(f"[ERROR] Download failed for {symbol} {interval} after 3 attempts: {e}")
                raise


def download(symbol, interval, limit):
    """Synchronous compatibility wrapper for callers outside the collector."""
    url = f"{BASE_URL}/quote/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if not data or not isinstance(data, list):
                raise ValueError(f"Invalid API response for {symbol} {interval}")
            df = _parse_klines(data)
            if df is None or df.empty:
                raise ValueError(f"Empty data returned for {symbol} {interval}")
            return df
        except Exception as e:
            if attempt < 2:
                sleep = 2 ** attempt
                logging.warning(f"[RETRY] {symbol} {interval}: {e}, retrying in {sleep}s...")
                time.sleep(sleep)
            else:
                logging.error(f"[ERROR] Download failed for {symbol} {interval} after 3 attempts: {e}")
                raise


def _is_pair_due(symbol, interval):
    """Return True only when this symbol/timeframe should hit the API."""
    check_interval = CHECK_INTERVAL_MINUTES.get(interval, 0)
    if check_interval <= 0:
        return True
    marker = DATA_DIR / f"{symbol}_{interval}_last_check.txt"
    try:
        last_check = float(marker.read_text(encoding="utf-8").strip())
        if (time.time() - last_check) / 60.0 < check_interval:
            return False
    except (OSError, ValueError):
        pass
    return True


def _mark_checked(symbol, interval):
    marker = DATA_DIR / f"{symbol}_{interval}_last_check.txt"
    try:
        marker.write_text(str(time.time()), encoding="utf-8")
    except OSError as exc:
        logging.debug(f"[CHECK] Could not update {marker}: {exc}")


def _fetch_limit(symbol, interval):
    filename = DATA_DIR / f"{symbol}_{interval}.csv"
    max_candles = MAX_CANDLES_PER_TF.get(interval, 1500)
    if not filename.exists():
        return max_candles
    try:
        old_df = pd.read_csv(filename)
        time_col = "timestamp" if "timestamp" in old_df.columns else "open_time"
        last_ts = pd.to_datetime(old_df[time_col].iloc[-1], errors="coerce")
        if (pd.isna(last_ts) or last_ts.year < 2000) and "open_time" in old_df.columns:
            last_ts = pd.to_datetime(old_df["open_time"].iloc[-1], errors="coerce")
        gap_seconds = max(0, (pd.Timestamp.now(tz="UTC").tz_localize(None) - last_ts).total_seconds())
        return min(max(int(gap_seconds / TF_SECONDS.get(interval, 900)) + 2, 5), max_candles)
    except (KeyError, IndexError, ValueError, TypeError):
        return min(5, max_candles)

def save(symbol, interval, new_df=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = DATA_DIR / f"{symbol}_{interval}.csv"
    tf_seconds = TF_SECONDS.get(interval, 900)
    max_candles = MAX_CANDLES_PER_TF.get(interval, 1500)
    check_interval = CHECK_INTERVAL_MINUTES.get(interval, 0)

    last_check_file = DATA_DIR / f"{symbol}_{interval}_last_check.txt"
    old_df = None

    if filename.exists():
        try:
            old_df = pd.read_csv(filename)
            old_df["open_time"] = pd.to_datetime(old_df["open_time"], errors="coerce")
            old_df = old_df.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
            if len(old_df) == 0:
                old_df = None
            else:
                last_ts = old_df["open_time"].iloc[-1]
                now = pd.Timestamp.now(tz="UTC").tz_localize(None)
                gap_minutes = (now - last_ts).total_seconds() / 60
                if gap_minutes < tf_seconds / 60:
                    if check_interval > 0 and last_check_file.exists():
                        try:
                            last_check_time = float(last_check_file.read_text().strip())
                            elapsed_minutes = (time.time() - last_check_time) / 60
                            if elapsed_minutes < check_interval:
                                return 0
                        except (ValueError, OSError):
                            pass
        except:
            old_df = None

    if new_df is None:
        if old_df is None or len(old_df) < 100:
            limit = max_candles
            logging.info(f"[INFO] First download for {symbol} {interval}, limit = {limit}")
        else:
            last_ts = old_df["open_time"].iloc[-1]
            now = pd.Timestamp.now(tz="UTC").tz_localize(None)
            gap_seconds = (now - last_ts).total_seconds()
            gap_candles = int(gap_seconds / tf_seconds) + 1
            limit = min(max(gap_candles, 5), max_candles)
            logging.info(f"[INFO] Update {symbol} {interval}: last={last_ts}, gap={gap_candles} candles -> limit={limit}")

        try:
            new_df = download(symbol, interval, limit)
        except Exception as e:
            logging.error(f"[ERROR] Download failed for {symbol} {interval}: {e}")
            return 0

    if new_df.empty:
        _mark_checked(symbol, interval)
        return 0

    if old_df is not None:
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["open_time"], keep="last")
        combined = combined.sort_values("open_time").reset_index(drop=True)
        max_limit = MAX_CANDLES_PER_TF.get(interval, 1500)
        if len(combined) > max_limit:
            combined = combined.iloc[-max_limit:].reset_index(drop=True)
        final_df = combined
        old_times = set(pd.to_datetime(old_df["open_time"]))
        new_times = set(pd.to_datetime(new_df["open_time"]))
        new_count = len(new_times - old_times)
    else:
        final_df = new_df
        new_count = len(new_df)

    tmp_path = filename.with_suffix(filename.suffix + ".tmp")
    try:
        final_df.to_csv(tmp_path, index=False)
        os.replace(str(tmp_path), str(filename))
        _mark_checked(symbol, interval)
        return new_count
    except Exception as e:
        logging.error(f"[ERROR] Atomic write failed for {filename}: {e}")
        try:
            tmp_path.unlink()
        except:
            pass
        return 0


async def fetch_market_data_async(symbols=None, timeframes=None, session=None):
    """Fetch every symbol/timeframe concurrently, then merge each file atomically."""
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    pairs = [(symbol, interval) for symbol in symbols for interval in timeframes]
    own_session = session is None
    if own_session:
        connector = aiohttp.TCPConnector(limit=max(4, len(pairs)))
        session = aiohttp.ClientSession(connector=connector)
    try:
        due_pairs = [
            (symbol, interval)
            for symbol, interval in pairs
            if _is_pair_due(symbol, interval)
        ]
        logging.info(f"[FETCH] due pairs: {len(due_pairs)}/{len(pairs)}")
        tasks = [
            download_async(session, symbol, interval, _fetch_limit(symbol, interval))
            for symbol, interval in due_pairs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if own_session:
            await session.close()

    total_new = 0
    loop = asyncio.get_running_loop()
    for (symbol, interval), result in zip(due_pairs, results):
        if isinstance(result, Exception):
            logging.error(f"[ERROR] Download failed for {symbol} {interval}: {result}")
            continue
        if result is None or (hasattr(result, "empty") and result.empty):
            logging.warning(f"[WARN] Empty data for {symbol} {interval}, skipping")
            continue
        try:
            count = await asyncio.wait_for(
                loop.run_in_executor(None, save, symbol, interval, result),
                timeout=30,
            )
        except asyncio.TimeoutError:
            logging.error(f"[ERROR] Save timed out for {symbol} {interval}")
            continue
        except Exception as e:
            logging.error(f"[ERROR] Save failed for {symbol} {interval}: {e}")
            continue
        if count > 0:
            logging.info(f"[✓] {symbol} {interval}  →  +{count} candles")
            total_new += count
    return total_new

async def fetch_market_data(symbols=None, timeframes=None, session=None):
    return await fetch_market_data_async(symbols, timeframes, session=session)


def align_timeframes(raw_by_tf, base_tf="5m", asof=None):
    """Align closed candles without lookahead; HTF values are backward as-of joined."""
    if base_tf not in raw_by_tf or raw_by_tf[base_tf].empty:
        return pd.DataFrame()
    cutoff = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    normalized = {}
    for tf, source in raw_by_tf.items():
        frame = source.copy()
        timestamp_source = frame["timestamp"] if "timestamp" in frame.columns else frame["open_time"]
        if "open_time" in frame.columns:
            parsed_timestamp = pd.to_datetime(timestamp_source, errors="coerce")
            invalid = parsed_timestamp.isna() | (parsed_timestamp.dt.year < 2000)
            if invalid.any():
                timestamp_source = timestamp_source.copy()
                timestamp_source.loc[invalid] = frame.loc[invalid, "open_time"]
        frame["timestamp"] = pd.to_datetime(
            timestamp_source, errors="coerce"
        )
        frame = frame[frame["timestamp"] <= cutoff].dropna(subset=["timestamp"])
        normalized[tf] = frame.sort_values("timestamp").drop_duplicates(
            "timestamp", keep="last"
        ).reset_index(drop=True)
    base = normalized.get(base_tf, pd.DataFrame()).copy()
    if base.empty:
        return base
    base = base[base["timestamp"] <= cutoff].sort_values("timestamp").reset_index(drop=True)
    result = base.add_suffix(f"_{base_tf}")
    result = result.rename(columns={f"timestamp_{base_tf}": "timestamp"})
    for tf, frame in normalized.items():
        if tf == base_tf or frame.empty:
            continue
        right = frame.drop(columns=["timestamp"], errors="ignore").add_suffix(f"_{tf}")
        right["timestamp"] = frame["timestamp"].to_numpy()
        result = result.dropna(subset=["timestamp"])
        if result.empty:
            continue
        result = pd.merge_asof(
            result.sort_values("timestamp"),
            right.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        value_cols = [c for c in right.columns if c != "timestamp"]
        result[value_cols] = result[value_cols].ffill()
    return result

async def main():
    logging.info("="*60)
    logging.info("TOOBIT DATA COLLECTOR (FIXED v2)")
    logging.info("="*60)
    while True:
        logging.info(f"\nCycle: {pd.Timestamp.now().strftime('%H:%M:%S')}")
        total = await fetch_market_data()
        if total > 0:
            logging.info(f"Total new candles: {total}")
        else:
            logging.info("No new data found.")
        logging.info(f"Sleeping {UPDATE_SECONDS}s...")
        await asyncio.sleep(UPDATE_SECONDS)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())