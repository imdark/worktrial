"""Time, behind an interface.

Wall time is correct on hardware and wasteful in simulation, where we want to
step faster than real time and where tests want determinism. Everything that
needs a timestamp or a rate takes a ``Clock``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np


class Clock(ABC):
    @abstractmethod
    def now(self) -> float:
        """Monotonic seconds."""

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Advance time by at least ``seconds``."""


class WallClock(Clock):
    def now(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class ManualClock(Clock):
    """Deterministic clock: time moves only when advanced.

    Makes loop and watchdog tests exact and instantaneous.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += max(0.0, float(seconds))

    def advance(self, seconds: float) -> None:
        self._t += float(seconds)


class RateLimiter:
    """Holds a loop to a fixed rate and records how well it managed.

    Jitter statistics are not decoration: a control loop that silently drifts
    off its nominal rate produces a dataset whose action timing does not match
    the robot it will later be deployed on.
    """

    def __init__(self, hz: float, clock: Clock) -> None:
        if hz <= 0:
            raise ValueError(f"rate must be positive, got {hz}")
        self.hz = float(hz)
        self.period = 1.0 / self.hz
        self._clock = clock
        self._last: float | None = None
        self._periods: list[float] = []
        self._overruns = 0

    def reset(self) -> None:
        self._last = None
        self._periods.clear()
        self._overruns = 0

    def sleep(self) -> float:
        """Block until the next tick. Returns the realised period in seconds."""
        now = self._clock.now()
        if self._last is None:
            self._last = now
            return self.period

        elapsed = now - self._last
        remaining = self.period - elapsed
        if remaining > 0:
            self._clock.sleep(remaining)
        else:
            self._overruns += 1

        tick = self._clock.now()
        realised = tick - self._last
        self._last = tick
        self._periods.append(realised)
        return realised

    @property
    def overruns(self) -> int:
        """Ticks that took longer than the nominal period."""
        return self._overruns

    def stats(self) -> dict[str, float]:
        if not self._periods:
            return {"mean_hz": 0.0, "mean_period": 0.0, "max_period": 0.0, "jitter": 0.0}
        periods = np.asarray(self._periods, dtype=np.float64)
        return {
            "mean_hz": float(1.0 / periods.mean()),
            "mean_period": float(periods.mean()),
            "max_period": float(periods.max()),
            "jitter": float(np.abs(periods - self.period).max()),
        }
