"""Success detection without privileged state. Stage 11a.

In simulation success is read straight out of mj_data. On hardware there is no
such state, and Stage 11's exit criterion -- N consecutive autonomous episodes
-- is unmeasurable without one of these. Stubbed from Stage 0 so the gap is
visible in the interface rather than discovered late.
"""

from __future__ import annotations

from teleop_sim.core.protocols import SuccessDetector
from teleop_sim.core.registry import SUCCESS, register
from teleop_sim.core.types import Observation

_STAGE_11A = (
    "lands at Stage 11a. Sim reads privileged simulator state; hardware cannot, so "
    "real autonomous evaluation needs a learned classifier, an instrumented fixture, "
    "or a human. Until then use a simulation success detector."
)


@register(SUCCESS, "learned")
class LearnedSuccessDetector(SuccessDetector):
    """Vision classifier over the final frames of an episode."""

    def __init__(self, **kwargs: object) -> None:
        raise NotImplementedError(f"LearnedSuccessDetector {_STAGE_11A}")

    def is_success(self, obs: Observation) -> bool:
        raise NotImplementedError


@register(SUCCESS, "fixture")
class FixtureSuccessDetector(SuccessDetector):
    """Instrumented jig: a switch, a load cell, or a light gate."""

    def __init__(self, **kwargs: object) -> None:
        raise NotImplementedError(f"FixtureSuccessDetector {_STAGE_11A}")

    def is_success(self, obs: Observation) -> bool:
        raise NotImplementedError
