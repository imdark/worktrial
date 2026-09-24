"""No-op reset: the robot's own homing is the whole reset.

Correct whenever there is no scene to restore -- a bare arm, a smoke test, or
a task whose objects are fixed. Sim re-randomisation arrives at Stage 4 and
the physical strategies at Stage 11.
"""

from __future__ import annotations

import numpy as np

from teleop_sim.core.protocols import ResetStrategy
from teleop_sim.core.registry import RESETS, register


@register(RESETS, "none")
class NoResetStrategy(ResetStrategy):
    def reset(self, rng: np.random.Generator) -> None:
        return None
