"""A policy, presented to the control loop as an action source.

Owns chunk buffering so that Policy implementations do not each reinvent it.
Synchronous: ``predict`` is called on the control thread when the buffer
empties, which is correct for scripted policies and for small checkpoints that
fit inside a tick. Anything slower belongs behind AsyncPolicySource.
"""

from __future__ import annotations

from collections import deque

from teleop_sim.core.clock import Clock
from teleop_sim.core.policy_spec import PolicySpec
from teleop_sim.core.protocols import ActionSource, Policy
from teleop_sim.core.registry import SOURCES, register
from teleop_sim.core.types import Action, ActionOrigin, Observation


@register(SOURCES, "policy")
class PolicySource(ActionSource):
    def __init__(
        self,
        policy: Policy,
        clock: Clock,
        origin: ActionOrigin | str = ActionOrigin.POLICY,
    ) -> None:
        self.policy = policy
        self.clock = clock
        self.origin = ActionOrigin(origin)
        self._buffer: deque[Action] = deque()

    @property
    def name(self) -> str:
        return f"policy({type(self.policy).__name__})"

    def reset(self, obs: Observation | None = None) -> None:
        self.policy.reset()
        self._buffer.clear()

    def get_action(self, obs: Observation) -> Action:
        if not self._buffer:
            chunk = self.policy.predict(obs)
            self._buffer.extend(chunk.actions)

        action = self._buffer.popleft()
        # The chunk was computed once; each action is *issued* when it is
        # popped. obs_timestamp is left alone -- that is what makes the loop's
        # staleness measurement meaningful across a replayed chunk.
        action.timestamp = self.clock.now()
        action.source = self.origin
        return action

    def policy_spec(self) -> PolicySpec | None:
        return getattr(self.policy, "spec", None)
