"""Cartesian -> joint retargeting. Stage 8.

Deliberately present and deliberately unimplemented. A VR headset or a
SpaceMouse emits an end-effector pose; turning that into joint targets is the
only thing standing between those devices and the rest of the system. Keeping
the stub here from Stage 0 forces the Retargeter interface to stay honest
about a command kind it cannot yet serve.
"""

from __future__ import annotations

from typing import Any

from teleop_sim.core.protocols import Retargeter
from teleop_sim.core.registry import RETARGETERS, register
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Action, Observation, TeleopCommand


@register(RETARGETERS, "ik")
class IKRetargeter(Retargeter):
    def __init__(self, spec: RobotSpec, solver: str = "mink", **kwargs: Any) -> None:
        raise NotImplementedError(
            "IKRetargeter lands at Stage 8, alongside VR and SpaceMouse input. "
            "It will map ee_pose / ee_delta commands onto joint targets via "
            f"{solver}. Until then use retargeter type 'identity' with a "
            "joint-space device such as a leader arm."
        )

    def __call__(self, command: TeleopCommand, obs: Observation) -> Action:
        raise NotImplementedError
