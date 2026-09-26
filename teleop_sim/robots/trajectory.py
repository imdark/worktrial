"""Time-scaling helpers shared by scripted motions and safety moves."""

from __future__ import annotations


def min_jerk(tau: float) -> float:
    """Minimum-jerk progress 0..1: starts and ends with zero velocity AND zero
    acceleration, unlike a cosine ease, whose acceleration jumps at each end
    (felt on the rig as a kick at every reversal). Peaks at 1.875x the average
    speed."""
    tau = min(max(tau, 0.0), 1.0)
    return tau**3 * (10 - 15 * tau + 6 * tau**2)
