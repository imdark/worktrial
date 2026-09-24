"""Off-thread inference for policies that cannot answer within a tick. Stage 2.

Deliberately present and deliberately unimplemented. Three things in the plan
break the assumption that inference fits inside a control tick: the Stage 12
VLM (~200-500 ms at 1-5 Hz), a remote policy over ZMQ, and Stage 13 latent
planning. One wrapper covers all of them.

When implemented it runs ``predict`` on a worker thread, serves the freshest
buffered action every tick, and lets the loop compute staleness from
``Action.obs_timestamp``. Buffer exhaustion is an explicit, counted underrun
whose fallback (hold the last action, or decay toward it) is a config field
rather than a silent behaviour.
"""

from __future__ import annotations

from teleop_sim.core.clock import Clock
from teleop_sim.core.protocols import ActionSource, Policy
from teleop_sim.core.registry import SOURCES, register
from teleop_sim.core.types import Action, Observation


@register(SOURCES, "async_policy")
class AsyncPolicySource(ActionSource):
    def __init__(
        self,
        policy: Policy,
        clock: Clock,
        on_underrun: str = "hold_last",
        max_staleness_ms: float = 150.0,
    ) -> None:
        raise NotImplementedError(
            "AsyncPolicySource lands at Stage 2, alongside per-step staleness and "
            "underrun logging. It will run predict() on a worker thread so a slow "
            "policy cannot stall the control loop. Until then use source type "
            "'policy', which calls predict() synchronously when its chunk buffer "
            "empties -- correct only for policies that fit inside a tick."
        )

    def reset(self, obs: Observation | None = None) -> None:
        raise NotImplementedError

    def get_action(self, obs: Observation) -> Action:
        raise NotImplementedError
