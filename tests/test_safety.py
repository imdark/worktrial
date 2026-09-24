"""Validation: every autonomous episode terminates on its own.

Autonomy removes the human who would notice a stalled policy or an arm leaving
the table, so the guarantee has to be structural.
"""

from __future__ import annotations

import numpy as np
import pytest

from teleop_sim.control.loop import ControlLoop
from teleop_sim.control.safety import HardwareSafetyMonitor, SimWatchdog
from teleop_sim.control.sources import PolicySource, TeleopSource
from teleop_sim.core.types import Observation, Outcome
from teleop_sim.policies.constant import ConstantPolicy
from teleop_sim.retarget.identity import IdentityRetargeter
from tests.fakes import CountingReset, FakeRobot, FakeSuccessDetector, FakeTeleop


def _loop(spec, clock, robot, source, safety):
    return ControlLoop(
        robot,
        source,
        FakeSuccessDetector(success_after=None),
        CountingReset(),
        safety,
        clock=clock,
    )


def _outside(spec) -> Observation:
    return Observation(
        images={},
        joint_pos=np.zeros(spec.dof),
        ee_pose=[9.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        gripper=0.0,
        timestamp=0.0,
    )


def test_stalled_robot_trips_the_watchdog(spec, clock):
    robot = FakeRobot(spec, clock, stall=True)
    source = PolicySource(ConstantPolicy(spec, clock), clock=clock)
    safety = SimWatchdog(spec, max_steps=500, stall_steps=5)

    result = _loop(spec, clock, robot, source, safety).run_episode(seed=0)

    assert result.outcome is Outcome.WATCHDOG
    assert result.failure_tag == "stalled"
    assert result.steps == 5


def test_a_moving_robot_does_not_trip_stall_detection(spec, clock):
    robot = FakeRobot(spec, clock)
    source = TeleopSource(FakeTeleop(spec, clock), IdentityRetargeter(spec))
    safety = SimWatchdog(spec, max_steps=50, stall_steps=5)

    result = _loop(spec, clock, robot, source, safety).run_episode(seed=0)

    assert result.outcome is Outcome.TIMEOUT
    assert result.failure_tag == "max_steps"


def test_stall_detection_is_off_by_default(spec, clock):
    """A human pausing mid-demonstration looks exactly like a stalled policy,
    so teleop configs must not get stall detection unless they ask for it."""
    robot = FakeRobot(spec, clock, stall=True)
    source = PolicySource(ConstantPolicy(spec, clock), clock=clock)

    result = _loop(spec, clock, robot, source, SimWatchdog(spec, max_steps=10)).run_episode(0)

    assert result.outcome is Outcome.TIMEOUT
    assert result.failure_tag == "max_steps"


def test_wall_clock_timeout(spec, clock):
    robot = FakeRobot(spec, clock)
    source = TeleopSource(FakeTeleop(spec, clock), IdentityRetargeter(spec))
    safety = SimWatchdog(spec, max_steps=10_000, max_seconds=1.0)

    result = _loop(spec, clock, robot, source, safety).run_episode(seed=0)

    assert result.outcome is Outcome.TIMEOUT
    assert result.failure_tag == "max_seconds"
    assert result.steps == pytest.approx(30, abs=2)


def test_workspace_violation_is_detected(spec):
    safety = SimWatchdog(spec, enforce_workspace=True)
    safety.reset()
    verdict = safety.check(_outside(spec), step=0)
    assert verdict is not None
    assert verdict.outcome is Outcome.WATCHDOG
    assert verdict.tag == "out_of_workspace"


def test_workspace_check_is_opt_in(spec):
    safety = SimWatchdog(spec)
    safety.reset()
    assert safety.check(_outside(spec), step=0) is None


def test_invalid_config_rejected(spec):
    with pytest.raises(ValueError, match="max_steps"):
        SimWatchdog(spec, max_steps=0)
    with pytest.raises(ValueError, match="stall_steps"):
        SimWatchdog(spec, stall_steps=-1)


def test_hardware_safety_is_present_but_honest(spec):
    """Stage 11 lives behind a stub, not behind a silent gap."""
    with pytest.raises(NotImplementedError, match="Stage 11"):
        HardwareSafetyMonitor(spec, max_current_a=1.8)
