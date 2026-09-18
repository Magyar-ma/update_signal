import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TXT_FILE = BASE_DIR / "signals.txt"
OPEN_SIGNALS_FILE = DATA_DIR / "signals_open.json"
CLOSED_SIGNALS_FILE = DATA_DIR / "signals_closed.json"


def append_to_txt(text: str) -> None:
    with TXT_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n{text}\n")


def register_signal(sig: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    open_signals = _load_json(OPEN_SIGNALS_FILE)
    key = f"{sig['symbol']}|{sig['direction']}|{sig.get('timestamp', '')}"
    record = {
        "symbol": sig["symbol"],
        "direction": sig["direction"],
        "score": sig.get("score"),
        "exec_tf": sig.get("exec_tf"),
        "timestamp": sig.get("timestamp"),
        "plan": sig.get("plan"),
        "reasons": sig.get("reasons", []),
        "registered_at": time.time(),
        "status": "OPEN",
    }
    open_signals[key] = record
    _save_json(OPEN_SIGNALS_FILE, open_signals)


def load_open_signals() -> dict:
    return _load_json(OPEN_SIGNALS_FILE)


def load_closed_signals() -> dict:
    return _load_json(CLOSED_SIGNALS_FILE)


def close_signal(key: str, exit_price: float, exit_reason: str, exit_time, pnl_r: float = None) -> dict:
    open_signals = _load_json(OPEN_SIGNALS_FILE)
    closed = _load_json(CLOSED_SIGNALS_FILE)
    if key in open_signals:
        sig = open_signals.pop(key)
        sig["status"] = "CLOSED"
        sig["exit_price"] = exit_price
        sig["exit_reason"] = exit_reason
        sig["exit_time"] = exit_time
        if pnl_r is not None:
            sig["pnl_r"] = pnl_r
        closed[key] = sig
        _save_json(OPEN_SIGNALS_FILE, open_signals)
        _save_json(CLOSED_SIGNALS_FILE, closed)
        return sig
    return {}


def _load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
