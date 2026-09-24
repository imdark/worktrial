"""A Task is a composition, not an interface (§2.1, §3.3).

v2 gave one Task object four responsibilities -- scene randomisation, reset,
success detection and autonomous reset -- and those four split cleanly along
the sim/real line. Bundling them hid a missing work item: hardware success
detection, which Stage 11's exit criterion depends on.

Task survives as the convenience object that names an experiment and bundles
the pieces. It carries no interface of its own, and the control loop takes the
pieces directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from teleop_sim.core.protocols import ResetStrategy, SuccessDetector


@dataclass
class Task:
    name: str
    success: SuccessDetector
    reset: ResetStrategy
    scene: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
