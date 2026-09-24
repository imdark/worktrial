"""The shared contract every Robot implementation must satisfy (§5.1).

Principle 3 -- radians in, radians out; gripper normalised to [0, 1] -- was
stated in v2 and enforced nowhere. A unit bug in a new driver costs a week and
looks exactly like bad policy performance, because a policy trained on
degrees-shaped data simply performs badly rather than raising.

Subclass, implement ``base_spec`` and ``make_robot``, and pytest collects the
inherited tests. FakeRobot, MujocoRobot and ZMQRobotClient run this in CI; the
real driver runs it on a bench.
"""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import numpy as np
import pytest

from teleop_sim.core.protocols import Robot, UnsupportedControlMode
from teleop_sim.core.spec import RobotSpec, SensingSpec
from teleop_sim.core.types import Action, ControlMode

_TOL = 1e-9


class RobotConformance:
    #: False for a real driver, whose sensors cannot be conjured by editing a spec.
    supports_sensing_variants = True

    #: Steps allowed for a command to take effect. Instant for fakes, not for hardware.
    settle_steps = 50

    def base_spec(self) -> RobotSpec:
        raise NotImplementedError

    def make_robot(self, spec: RobotSpec) -> Robot:
        raise NotImplementedError

    # ------------------------------------------------------------- helpers

    def _robot(self, spec: RobotSpec | None = None) -> Robot:
        return self.make_robot(spec if spec is not None else self.base_spec())

    @staticmethod
    def _command(
        robot: Robot,
        values: np.ndarray,
        gripper: float = 0.0,
        mode: ControlMode | None = None,
    ) -> Action:
        return Action(
            mode=mode or robot.spec.default_control_mode,
            values=values,
            gripper=gripper,
            timestamp=0.0,
            obs_timestamp=0.0,
        )

    def _settle(self, robot: Robot, action: Action) -> None:
        for _ in range(self.settle_steps):
            robot.send_action(action)

    @staticmethod
    def _within_limits(obs_joints: np.ndarray, spec: RobotSpec) -> bool:
        return bool(
            np.all(obs_joints >= spec.lower_limits - _TOL)
            and np.all(obs_joints <= spec.upper_limits + _TOL)
        )

    # -------------------------------------------------------------- tests

    def test_reset_produces_a_valid_observation(self):
        robot = self._robot()
        obs = robot.reset(seed=0)
        assert obs.dof == robot.spec.dof
        assert self._within_limits(obs.joint_pos, robot.spec)
        assert 0.0 <= obs.gripper <= 1.0

    def test_joint_targets_are_radians_and_converge(self):
        """A driver interpreting the command as degrees fails here: it either
        saturates at a limit or barely moves."""
        robot = self._robot()
        spec = robot.spec
        obs = robot.reset(seed=0)

        # Relative to home, not to the range midpoint: for a joint like YAM's
        # joint2 ([0, 3.67]) the midpoint folds the arm into the table.
        target = spec.home_joints() + 0.25 * (spec.upper_limits - spec.home_joints())
        start_error = float(np.max(np.abs(obs.joint_pos - target)))
        assert start_error > 0, "target must differ from the home pose for this to test anything"

        self._settle(robot, self._command(robot, target))
        obs = robot.get_observation()

        end_error = float(np.max(np.abs(obs.joint_pos - target)))
        assert end_error < start_error
        assert self._within_limits(obs.joint_pos, spec)

    def test_out_of_limit_command_clips_rather_than_raising(self):
        robot = self._robot()
        robot.reset(seed=0)
        sent = robot.send_action(
            self._command(robot, np.full(robot.spec.dof, 1e3))
        )
        np.testing.assert_allclose(sent.values, robot.spec.upper_limits)
        assert self._within_limits(robot.get_observation().joint_pos, robot.spec)

    def test_send_action_returns_the_clipped_action_with_provenance_intact(self):
        robot = self._robot()
        robot.reset(seed=0)
        requested = self._command(robot, np.full(robot.spec.dof, 1e3))
        sent = robot.send_action(requested)

        assert sent.mode is requested.mode
        assert sent.source is requested.source
        assert sent.obs_timestamp == requested.obs_timestamp
        assert not np.allclose(sent.values, requested.values), "expected clipping"

    def test_gripper_normalisation_at_both_extremes(self):
        robot = self._robot()
        robot.reset(seed=0)
        home = robot.spec.home_joints()

        self._settle(robot, self._command(robot, home, gripper=0.0))
        assert robot.get_observation().gripper == 0.0

        self._settle(robot, self._command(robot, home, gripper=1.0))
        assert robot.get_observation().gripper == 1.0

        self._settle(robot, self._command(robot, home, gripper=0.5))
        middle = robot.get_observation().gripper
        assert 0.0 < middle < 1.0

    def test_timestamps_strictly_increase(self):
        robot = self._robot()
        robot.reset(seed=0)
        stamps = [robot.get_observation().timestamp for _ in range(20)]
        assert all(later > earlier for earlier, later in pairwise(stamps))

    def test_declared_control_modes_are_accepted(self):
        robot = self._robot()
        robot.reset(seed=0)
        for mode in robot.spec.supported_modes:
            robot.send_action(self._command(robot, robot.spec.home_joints(), mode=mode))

    def test_undeclared_control_mode_is_rejected(self):
        robot = self._robot()
        robot.reset(seed=0)
        undeclared = next(
            (mode for mode in ControlMode if not robot.spec.supports(mode)), None
        )
        if undeclared is None:
            pytest.skip("this robot declares every control mode")
        with pytest.raises(UnsupportedControlMode):
            robot.send_action(
                self._command(robot, robot.spec.home_joints(), mode=undeclared)
            )

    def test_unsensed_fields_are_absent_from_the_observation(self):
        robot = self._robot()
        obs = robot.reset(seed=0)
        for key in ("joint_current", "joint_torque", "ee_wrench"):
            declared = robot.spec.sensing.provides(key)
            present = getattr(obs, key) is not None
            assert present == declared, (
                f"spec declares sensing.{key}={declared} but the observation "
                f"{'carries' if present else 'omits'} it"
            )

    def test_declared_sensing_is_actually_populated(self):
        if not self.supports_sensing_variants:
            pytest.skip("sensors cannot be added by editing a spec on this robot")
        spec = dataclasses.replace(
            self.base_spec(),
            sensing=SensingSpec(
                joint_current=True, joint_torque=True, ee_wrench=True, estop=True
            ),
        )
        obs = self._robot(spec).reset(seed=0)
        assert obs.joint_current is not None
        assert obs.joint_torque is not None
        assert obs.ee_wrench is not None
        assert obs.ee_wrench.shape == (6,)
