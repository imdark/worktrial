"""FakeRobot against the shared Robot contract.

The first implementation through the suite. MujocoRobot joins it at Stage 1,
ZMQRobotClient at Stage 9, and the real driver on a bench.
"""

from __future__ import annotations

from teleop_sim.core.clock import WallClock
from teleop_sim.core.protocols import Robot
from teleop_sim.core.spec import RobotSpec
from tests.conformance.robot import RobotConformance
from tests.conftest import SPEC_PATH
from tests.fakes import FakeRobot


class TestFakeRobotConformance(RobotConformance):
    settle_steps = 5

    def base_spec(self) -> RobotSpec:
        return RobotSpec.from_yaml(SPEC_PATH)

    def make_robot(self, spec: RobotSpec) -> Robot:
        # A real clock: "timestamps strictly increase" is a contract about the
        # robot, and a ManualClock would let a broken driver pass it.
        robot = FakeRobot(spec, WallClock())
        robot.connect()
        return robot
