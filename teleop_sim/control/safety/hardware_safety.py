"""Real-hardware safety. Stage 11.

Deliberately present and deliberately unimplemented. In sim a safety trip
wastes an episode; on hardware it can break the arm, which is why this is a
separate implementation rather than a flag on the simulation watchdog. It adds
force/current limits, e-stop integration, and an active safe stop on trip.
"""

from __future__ import annotations

from teleop_sim.core.protocols import SafetyMonitor, SafetyVerdict
from teleop_sim.core.registry import SAFETY, register
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Observation


@register(SAFETY, "hardware_safety")
class HardwareSafetyMonitor(SafetyMonitor):
    def __init__(self, spec: RobotSpec | None = None, **kwargs: object) -> None:
        raise NotImplementedError(
            "HardwareSafetyMonitor lands at Stage 11, with force/current limits, "
            "e-stop integration and safe-stop-on-trip. Note that the startup check "
            "(control/compat.py) already refuses a configured force limit on a robot "
            "whose spec declares no such sensor, so that limit cannot silently never "
            "fire. Until Stage 11 use safety type 'sim_watchdog'."
        )

    def reset(self) -> None:
        raise NotImplementedError

    def check(self, obs: Observation, step: int) -> SafetyVerdict | None:
        raise NotImplementedError
