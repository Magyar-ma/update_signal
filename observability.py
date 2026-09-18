"""Cycle-level observability for the signal engine.

Tracks timing, counts, and quality metrics for each pipeline cycle.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import config


@dataclass
class CycleMetrics:
    fetch_ms: float = 0.0
    cache_ms: float = 0.0
    phase2_ms: float = 0.0
    phase3_ms: float = 0.0
    phase4_ms: float = 0.0
    phase5_ms: float = 0.0
    phase6_ms: float = 0.0
    phase7_ms: float = 0.0
    tracker_ms: float = 0.0
    total_ms: float = 0.0
    api_requests: int = 0
    candles_downloaded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    signals_generated: int = 0
    signals_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def log_summary(self) -> None:
        logging.info("")
        logging.info("=" * 60)
        logging.info("📊 CYCLE METRICS")
        logging.info("=" * 60)
        logging.info(f"  Total cycle time : {self.total_ms:.1f} ms")
        logging.info(f"  Fetch            : {self.fetch_ms:.1f} ms ({self.fetch_ms / max(self.total_ms, 1) * 100:.1f}%)")
        logging.info(f"  Cache            : {self.cache_ms:.1f} ms ({self.cache_ms / max(self.total_ms, 1) * 100:.1f}%)")
        logging.info(f"  Phase 2 (indicators) : {self.phase2_ms:.1f} ms")
        logging.info(f"  Phase 3 (structure)  : {self.phase3_ms:.1f} ms")
        logging.info(f"  Phase 4 (FVG/OB)     : {self.phase4_ms:.1f} ms")
        logging.info(f"  Phase 5 (patterns)   : {self.phase5_ms:.1f} ms")
        logging.info(f"  Phase 6 (volume)     : {self.phase6_ms:.1f} ms")
        logging.info(f"  Phase 7 (scoring)    : {self.phase7_ms:.1f} ms")
        logging.info(f"  Tracker            : {self.tracker_ms:.1f} ms")
        logging.info(f"  API requests       : {self.api_requests}")
        logging.info(f"  Candles downloaded : {self.candles_downloaded}")
        logging.info(f"  Cache hits/misses  : {self.cache_hits}/{self.cache_misses}")
        logging.info(f"  Signals generated  : {self.signals_generated}")
        logging.info(f"  Signals rejected   : {self.signals_rejected}")
        if self.rejection_reasons:
            for reason, count in sorted(self.rejection_reasons.items(), key=lambda x: -x[1]):
                logging.info(f"    - {reason}: {count}")
        logging.info("=" * 60)


class MetricsCollector:
    def __init__(self) -> None:
        self._start: float = 0.0
        self._phase_starts: dict[str, float] = {}
        self.metrics = CycleMetrics()

    def start_cycle(self) -> None:
        self._start = time.perf_counter()
        self.metrics = CycleMetrics()

    def start_phase(self, name: str) -> None:
        self._phase_starts[name] = time.perf_counter()

    def end_phase(self, name: str) -> None:
        start = self._phase_starts.pop(name, time.perf_counter())
        elapsed = (time.perf_counter() - start) * 1000.0
        attr = f"{name}_ms"
        if hasattr(self.metrics, attr):
            setattr(self.metrics, attr, elapsed)

    def end_cycle(self) -> CycleMetrics:
        self.metrics.total_ms = (time.perf_counter() - self._start) * 1000.0
        return self.metrics

    def record_rejection(self, reason: str) -> None:
        self.metrics.signals_rejected += 1
        self.metrics.rejection_reasons[reason] = (
            self.metrics.rejection_reasons.get(reason, 0) + 1
        )
