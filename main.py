import time
import os
import asyncio
import logging
import aiohttp
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from data_collector import fetch_market_data
import notifier
import tracker
import pipeline
import observability

notifier.enable_utf8_console()

try:
    from config import SYMBOLS, TIMEFRAMES, UPDATE_SECONDS
except ImportError:
    SYMBOLS = ["BTC-SWAP-USDT"]
    TIMEFRAMES = ["5m", "15m", "1h", "4h"]
    UPDATE_SECONDS = 120


async def run_pipeline_parallel(metrics=None, executor=None):
    """اجرای موازی در سطح symbol؛ worker pool بین cycleها persistent است."""
    all_signals = []
    owns_executor = executor is None
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=min(4, len(SYMBOLS), os.cpu_count() or 2))
    loop = asyncio.get_running_loop()

    futures = [
        loop.run_in_executor(executor, pipeline.run_symbol_timed, symbol)
        for symbol in SYMBOLS
    ]
    results = await asyncio.gather(*futures, return_exceptions=True)

    # Worker metrics are process-local. For wall-clock relevance we aggregate
    # the slowest worker per phase rather than incorrectly sharing one mutable
    # MetricsCollector across concurrent workers.
    worker_metrics = []
    for symbol, result in zip(SYMBOLS, results):
        if isinstance(result, Exception):
            logging.error(f"[✗] {symbol} error: {type(result).__name__}: {result}")
            continue
        sigs = result.get("signals", [])
        worker_metrics.append(result.get("metrics"))
        if sigs:
            all_signals.extend(sigs)
            logging.info(f"[✓] {symbol} -> {len(sigs)} signal(s)")
        else:
            logging.info(f"[✓] {symbol} -> no signal")

    if metrics and worker_metrics:
        for name in ("phase2_ms", "phase3_ms", "phase4_ms", "phase5_ms", "phase6_ms", "phase7_ms"):
            setattr(metrics.metrics, name, max(getattr(m, name, 0.0) for m in worker_metrics if m is not None))
    all_signals.sort(key=lambda s: s["score"], reverse=True)
    if owns_executor:
        executor.shutdown(wait=True)
    return all_signals


async def main():
    logging.info("=" * 60)
    logging.info("🤖 TOOBIT AI SIGNAL BOT (IN-MEMORY PIPELINE)")
    logging.info(f"📊 {len(SYMBOLS)} symbols, {len(TIMEFRAMES)} timeframes")
    logging.info("=" * 60)

    metrics = observability.MetricsCollector()
    connector = aiohttp.TCPConnector(limit=max(4, len(SYMBOLS) * len(TIMEFRAMES)))
    pipeline_executor = ProcessPoolExecutor(max_workers=min(4, len(SYMBOLS), os.cpu_count() or 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            metrics.start_cycle()
            t_cycle = time.time()
            logging.info(f"\n[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] Cycle start")

            # دانلود دادهها (با منطق هوشمند)
            try:
                total_new = await fetch_market_data(session=session, symbols=SYMBOLS, timeframes=TIMEFRAMES)
                metrics.metrics.candles_downloaded += total_new or 0
            except Exception as e:
                logging.error(f"[MAIN] fetch_market_data error: {e}")
                await asyncio.sleep(5)
                continue

            if total_new is None or total_new < 0:
                logging.warning("[MAIN] Invalid data count, skipping cycle...")
                await asyncio.sleep(2)
                continue

            # اگر هیچ دادهی جدیدی اضافه نشده، از پردازش صرفنظر میکنیم
            if total_new == 0:
                logging.info("[MAIN] No new data, skipping pipeline...")
                elapsed = time.time() - t_cycle
                logging.info(f"[MAIN] Cycle done in {elapsed:.2f}s")
                await asyncio.sleep(max(1, UPDATE_SECONDS - elapsed))
                continue

            # پردازش موازی در حافظه (بدون CSV میانی)
            logging.info("LOG: Entering run_pipeline_parallel")
            try:
                signals = await run_pipeline_parallel(metrics, pipeline_executor)
            except Exception as e:
                logging.error(f"[MAIN] pipeline error: {e}")
                await asyncio.sleep(5)
                continue
            logging.info(f"LOG: run_pipeline_parallel returned {len(signals)} signals")

            logging.info("LOG: Entering notifier.deliver")
            delivered = notifier.deliver(signals)
            if delivered:
                logging.info(f"\n[MAIN] {len(delivered)} signal(s) delivered "
                      f"({len(signals) - len(delivered)} duplicate(s) suppressed).")
            elif signals:
                logging.info(f"[MAIN] {len(signals)} signal(s) suppressed as duplicates.")
            else:
                logging.info("[MAIN] No active signal.")

            elapsed = time.time() - t_cycle

            # پیگیری سیگنالهای باز: رسیدن به TP1/TP2/TP3 یا SL + گزارش عملکرد دورهای
            logging.info("LOG: Entering tracker.evaluate_open_signals")
            try:
                tracker.evaluate_open_signals()
                cycle_count = getattr(main, "_cycle", 0) + 1
                main._cycle = cycle_count
                if cycle_count % 20 == 0:
                    logging.info("LOG: Entering tracker.summarize_performance")
                    tracker.summarize_performance()
            except Exception as e:
                logging.error(f"[MAIN] tracker error: {e}")

            cycle_metrics = metrics.end_cycle()
            cycle_metrics.log_summary()

            logging.info(f"[MAIN] Cycle done in {elapsed:.2f}s")
            await asyncio.sleep(max(1, UPDATE_SECONDS - elapsed))


if __name__ == "__main__":
    asyncio.run(main())
