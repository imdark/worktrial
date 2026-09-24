"""The smallest possible autonomous policy.

Exists for one reason: to prove the control loop runs with no human attached,
from the same config file a teleoperated run uses. That is the Stage 0 exit
criterion, and it is the claim the whole autonomy track rests on.
"""

from __future__ import annotations

import numpy as np

from teleop_sim.core.clock import Clock
from teleop_sim.core.policy_spec import PolicySpec
from teleop_sim.core.protocols import Policy
from teleop_sim.core.registry import POLICIES, register
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Action, ActionChunk, ActionOrigin, Observation


@register(POLICIES, "constant")
class ConstantPolicy(Policy):
    """Commands a fixed joint target -- by default, hold wherever we started."""

    def __init__(
        self,
        spec: RobotSpec,
        clock: Clock,
        target: list[float] | None = None,
        gripper: float = 0.0,
        horizon: int = 1,
    ) -> None:
        self.robot_spec = spec
        self.clock = clock
        self.horizon = int(horizon)
        self.gripper = float(gripper)
        self._configured = None if target is None else np.asarray(target, dtype=np.float64)
        self._target: np.ndarray | None = None

        self.spec = PolicySpec(
            policy_id="constant",
            control_mode=spec.default_control_mode,
            obs_keys=["joint_pos", "gripper"],
            action_horizon=self.horizon,
            expected_control_hz=spec.control_hz,
            trained_against_robot_spec=spec.content_hash(),
        )

    def reset(self) -> None:
        self._target = None if self._configured is None else self._configured.copy()

    def predict(self, obs: Observation) -> ActionChunk:
        if self._target is None:
            self._target = obs.joint_pos.copy()
        values = self.robot_spec.clip_joints(self._target)
        now = self.clock.now()
        actions = [
            Action(
                mode=self.robot_spec.default_control_mode,
                values=values,
                gripper=self.gripper,
                timestamp=now,
                obs_timestamp=obs.timestamp,
                source=ActionOrigin.POLICY,
            )
            for _ in range(self.horizon)
        ]
        return ActionChunk(
            actions=actions,
            obs_timestamp=obs.timestamp,
            horizon_hz=self.robot_spec.control_hz,
        )
