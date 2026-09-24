"""MujocoRobot against the shared Robot contract.

The Stage 1 exit criterion. The same ten tests FakeRobot passes, so anything
downstream -- loop, recorder, policy -- cannot tell the two apart.
"""

from __future__ import annotations

import pytest

from teleop_sim.core.clock import WallClock
from teleop_sim.core.protocols import Robot
from teleop_sim.core.spec import RobotSpec, SensingSpec
from teleop_sim.robots.sim.mujoco_robot import ModelMismatch, MujocoRobot
from tests.conformance.robot import RobotConformance
from tests.conftest import SPEC_PATH

pytest.importorskip("mujoco", reason="needs the 'sim' extra")


def _spec(**overrides) -> RobotSpec:
    import dataclasses

    spec = RobotSpec.from_yaml(SPEC_PATH)
    return dataclasses.replace(spec, **overrides) if overrides else spec


def _robot(spec: RobotSpec) -> MujocoRobot:
    robot = MujocoRobot(spec, WallClock(), image_size=(64, 48))
    robot.connect()
    return robot


class TestMujocoRobotConformance(RobotConformance):
    # Sensing here is a property of the MJCF, not of a spec flag: MuJoCo has no
    # motor current, and a wrench needs force/torque sensors in the model. The
    # driver refuses to fake either -- see the tests below.
    supports_sensing_variants = False

    # One send_action advances 17 physics substeps; a position servo needs a
    # little longer than the fake's instant tracking.
    settle_steps = 80

    def base_spec(self) -> RobotSpec:
        return _spec()

    def make_robot(self, spec: RobotSpec) -> Robot:
        return _robot(spec)


class TestYamConformance(RobotConformance):
    """A second, real arm through the identical contract -- the hardware-as-data
    claim tested rather than asserted. Composed (arm + stock gripper) and bare:
    the view of the robot the real driver has. Its gripper is a coupled
    two-finger design with a deadband, where SO-101's is a single jaw."""

    supports_sensing_variants = False
    settle_steps = 80

    def base_spec(self) -> RobotSpec:
        return RobotSpec.from_yaml(SPEC_PATH.parent / "yam.yaml")

    def make_robot(self, spec: RobotSpec) -> Robot:
        return _robot(spec)


class TestYamTestJawConformance(TestYamConformance):
    """Same arm, a different end effector: swapping the gripper is one line in
    the robot description, and the contract still holds."""

    def base_spec(self) -> RobotSpec:
        return RobotSpec.from_yaml(
            SPEC_PATH.parent.parent.parent.parent / "tests" / "assets" / "yam_test_jaw.yaml"
        )


def test_declared_joint_torque_is_populated():
    """The one sensing field MuJoCo can honestly provide."""
    obs = _robot(_spec(sensing=SensingSpec(joint_torque=True))).reset(seed=0)
    assert obs.joint_torque is not None
    assert obs.joint_torque.shape == (5,)
    assert obs.joint_current is None


@pytest.mark.parametrize(
    "flags, message",
    [
        ({"joint_current": True}, "does not model motor current"),
        ({"ee_wrench": True}, "no force/torque sensor"),
    ],
)
def test_unmodellable_sensing_is_refused_at_construction(flags, message):
    """Better to refuse than to put fabricated numbers into a dataset that a
    real arm would later have to reproduce."""
    with pytest.raises(ModelMismatch, match=message):
        _robot(_spec(sensing=SensingSpec(**flags)))


def test_missing_joint_in_the_model_is_named():
    from teleop_sim.core.spec import JointSpec

    spec = _spec()
    broken = _spec(joints=[*spec.joints, JointSpec("seventh_axis", -1.0, 1.0)])
    with pytest.raises(ModelMismatch, match="seventh_axis"):
        _robot(broken)


def test_sim_time_advances_with_control_steps():
    robot = _robot(_spec())
    obs = robot.reset(seed=0)
    assert obs.extra["sim_time"] == 0.0

    action = RobotConformance._command(robot, robot.spec.neutral_joints())
    robot.send_action(action)
    after = robot.get_observation()

    expected = robot.substeps * robot.model.opt.timestep
    assert after.extra["sim_time"] == pytest.approx(expected)
    assert after.extra["sim_time"] == pytest.approx(1.0 / robot.spec.control_hz, abs=1e-3)
