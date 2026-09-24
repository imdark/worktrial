"""No success criterion: the episode ends when the safety monitor says so.

Honest for smoke runs and open-ended data collection, where "did it work" is a
question for a human watching the replay. Task-specific detectors arrive with
the first real task at Stage 4.
"""

from __future__ import annotations

from teleop_sim.core.protocols import SuccessDetector
from teleop_sim.core.registry import SUCCESS, register
from teleop_sim.core.types import Observation


@register(SUCCESS, "never")
class NeverSucceeds(SuccessDetector):
    def is_success(self, obs: Observation) -> bool:
        return False

    def failure_tag(self, obs: Observation) -> str | None:
        return "no_success_criterion"
