"""Rate, staleness and underrun accounting (§4.4).

The loop owns the clock. Every step records how long it took, how old the
action it applied was, and whether the source had a fresh answer at all. A
source that consistently underruns is a configuration error the harness
reports, not a mystery.
"""

from __future__ import annotations

import numpy as np


class LatencyTracker:
    def __init__(self) -> None:
        self._dts: list[float] = []
        self._staleness_ms: list[float] = []
        self.underruns = 0
        self.steps = 0

    def reset(self) -> None:
        self._dts.clear()
        self._staleness_ms.clear()
        self.underruns = 0
        self.steps = 0

    def record(
        self, dt: float, staleness_ms: float | None = None, underrun: bool = False
    ) -> None:
        self.steps += 1
        self._dts.append(float(dt))
        if staleness_ms is not None:
            self._staleness_ms.append(float(staleness_ms))
        if underrun:
            self.underruns += 1

    def staleness_stats(self) -> dict[str, float]:
        if not self._staleness_ms:
            return {}
        values = np.asarray(self._staleness_ms, dtype=np.float64)
        return {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    def stats(self) -> dict[str, float]:
        out: dict[str, float] = {"steps": float(self.steps), "underruns": float(self.underruns)}
        if self._dts:
            dts = np.asarray(self._dts, dtype=np.float64)
            out["mean_dt"] = float(dts.mean())
            out["max_dt"] = float(dts.max())
        return out
