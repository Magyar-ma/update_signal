"""Import smoke test for the signal engine."""
from __future__ import annotations

import importlib
import sys

MODULES = [
    "config", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7",
    "incremental_indicators", "incremental_engine", "pipeline",
    "data_collector", "tracker", "notifier", "observability",
    "signal_journal", "confidence_scoring", "adaptive_targets",
    "market_regime", "score_calibration",
]

def main() -> int:
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"[OK] {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    if failed:
        print("\nImport smoke test failed.")
        return 1
    print("\nImport smoke test passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
