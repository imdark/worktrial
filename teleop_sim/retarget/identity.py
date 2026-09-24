"""Straight-through joint retargeting.

The minimum viable Retargeter: a leader arm whose kinematics match the
follower exactly. The calibrated affine version (per-joint offset and scale,
loaded from a calibration file) arrives with the real leader arm at Stage 8.
"""

from __future__ import annotations

from teleop_sim.core.protocols import Retargeter
from teleop_sim.core.registry import RETARGETERS, register
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Action, ActionOrigin, Observation, TeleopCommand


class RetargetError(ValueError):
    """A teleop command cannot be mapped onto this robot."""


@register(RETARGETERS, "identity")
class IdentityRetargeter(Retargeter):
    def __init__(self, spec: RobotSpec) -> None:
        self.spec = spec

    def __call__(self, command: TeleopCommand, obs: Observation) -> Action:
        if command.kind != "joint":
            raise RetargetError(
                f"IdentityRetargeter handles 'joint' commands, got {command.kind!r}. "
                "Cartesian devices need an IKRetargeter."
            )
        if command.values.shape != (self.spec.dof,):
            raise RetargetError(
                f"command has {command.values.shape[0]} joint values but robot "
                f"{self.spec.name!r} has {self.spec.dof} joints"
            )
        return Action(
            mode=self.spec.default_control_mode,
            values=self.spec.clip_joints(command.values),
            gripper=command.gripper,
            timestamp=command.timestamp,
            obs_timestamp=obs.timestamp,
            source=ActionOrigin.HUMAN_TELEOP,
        )
