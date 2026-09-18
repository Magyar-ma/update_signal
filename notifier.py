import json
import sys
import time
import logging
from pathlib import Path

import pandas as pd

import signal_journal


def enable_utf8_console():
    """کنسول ویندوز بهصورت پیشفرض cp1252 است و ایموجی/فارسی را چاپ نمیکند."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


enable_utf8_console()

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "signals.log"
CSV_FILE = BASE_DIR / "signals.csv"
STATE_FILE = BASE_DIR / "data" / "signal_state.json"
TXT_FILE = BASE_DIR / "signals.txt"

# مقصد خروجی: 'txt' (پیشفرض، آماده برای تلگرام در آینده) | 'telegram' (بعداً فعال میشود)
OUTPUT_TARGET = "txt"

COOLDOWN_MINUTES = {"5m": 30, "15m": 60, "1h": 180, "4h": 480}
DEFAULT_COOLDOWN = 60

CSV_COLUMNS = [
    "sent_at", "candle_time", "symbol", "direction", "score", "exec_tf", "agreeing",
    "entry", "zone_low", "zone_high", "zone_source", "sl", "tp1", "tp2", "tp3",
    "risk_pct", "rr1", "rr2", "rr3", "leverage", "position_size", "reasons",
]


def _load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state):
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        logging.error(f"Error saving state: {e}")


def is_duplicate(signal, state):
    key = f"{signal['symbol']}|{signal['direction']}"
    prev = state.get(key)
    if not prev:
        return False
    if prev.get("candle_time") and prev["candle_time"] == signal.get("timestamp"):
        return True
    cooldown = COOLDOWN_MINUTES.get(signal["exec_tf"], DEFAULT_COOLDOWN) * 60
    if time.time() - prev.get("sent_at", 0) < cooldown:
        prev_entry = prev.get("entry")
        if prev_entry and abs(signal["plan"]["entry"] - prev_entry) / prev_entry < 0.004:
            return True
    return False


def _append_csv(rows):
    df_new = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if CSV_FILE.exists():
        df_new.to_csv(CSV_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")


def deliver(signals):
    if not signals:
        return []

    state = _load_state()
    delivered = []
    rows = []
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    for sig in signals:
        if is_duplicate(sig, state):
            continue
        plan = sig["plan"]

        # ثبت در فایل txt (مقصد نهایی — برای تلگرام در آینده آماده است)
        if OUTPUT_TARGET == "txt":
            signal_journal.append_to_txt(sig["card"])
        else:
            logging.info("\n" + sig["card"])

        # لاگ ماشینی (همیشه)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"\n[{now_str}]\n{sig['card']}\n")

        # ثبت در سیستم رهگیری برای پیگیری TP/SL
        signal_journal.register_signal(sig)

        rows.append([
            now_str, sig.get("timestamp"), sig["symbol"], sig["direction"], sig["score"],
            sig["exec_tf"], ",".join(sig["agreeing"]),
            round(plan["entry"], 8), round(plan["zone_low"], 8), round(plan["zone_high"], 8),
            plan["zone_source"], round(plan["sl"], 8), round(plan["tp1"], 8),
            round(plan["tp2"], 8), round(plan["tp3"], 8), round(plan["risk_pct"], 3),
            round(plan["rr1"], 2), round(plan["rr2"], 2), round(plan["rr3"], 2),
            plan["leverage"], round(plan.get("position_size", 0), 6), " | ".join(sig["reasons"]),
        ])

        state[f"{sig['symbol']}|{sig['direction']}"] = {
            "sent_at": time.time(),
            "candle_time": sig.get("timestamp"),
            "entry": plan["entry"],
            "score": sig["score"],
        }
        delivered.append(sig)

    if rows:
        _append_csv(rows)
        _save_state(state)

    return delivered
