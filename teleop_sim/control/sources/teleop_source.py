"""A human, presented to the control loop as an action source."""

from __future__ import annotations

from teleop_sim.core.protocols import ActionSource, Retargeter, Teleoperator
from teleop_sim.core.registry import SOURCES, register
from teleop_sim.core.types import Action, ActionOrigin, Observation


@register(SOURCES, "teleop")
class TeleopSource(ActionSource):
    origin = ActionOrigin.HUMAN_TELEOP

    def __init__(self, teleop: Teleoperator, retargeter: Retargeter) -> None:
        self.teleop = teleop
        self.retargeter = retargeter
        self._events: set[str] = set()

    @property
    def name(self) -> str:
        return f"teleop({type(self.teleop).__name__})"

    def reset(self, obs: Observation | None = None) -> None:
        self._events = set()
        self.retargeter.reset()
        if not self.teleop.is_connected:
            self.teleop.connect()

    def get_action(self, obs: Observation) -> Action:
        command = self.teleop.get_command()
        # Buttons are captured here, during the action call, so that the loop
        # can inspect them before deciding whether to send the action at all.
        # An e-stop that only took effect on the next tick would not be one.
        self._events = command.pressed_events()
        self.teleop.send_feedback(obs)
        return self.retargeter(command, obs)

    def poll_events(self) -> set[str]:
        return set(self._events)

    def close(self) -> None:
        if self.teleop.is_connected:
            self.teleop.disconnect()
