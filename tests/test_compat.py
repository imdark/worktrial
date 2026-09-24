"""Validation: incompatible combinations refuse to start (§4.3, principle 11).

The central risk in "swap the model" is that a checkpoint's requirements --
camera names, resolutions, control mode, rate, which observation fields it
consumes -- are implicit. A mismatch then surfaces as a KeyError at step 1, or
as quiet degradation that looks like a bad policy.
"""

from __future__ import annotations

import dataclasses

import pytest

from teleop_sim.control.compat import Incompatible, check_compatibility
from teleop_sim.core.policy_spec import CameraRequirement, PolicySpec
from teleop_sim.core.spec import SensingSpec
from teleop_sim.core.types import ControlMode

CAMERAS = {"wrist": (640, 480), "front": (640, 480)}


def _policy(**overrides) -> PolicySpec:
    base = {
        "policy_id": "act_v1",
        "control_mode": ControlMode.JOINT_POSITION,
        "cameras": [CameraRequirement("wrist", height=480, width=640)],
        "obs_keys": ["images", "joint_pos", "gripper"],
        "expected_control_hz": 30.0,
    }
    base.update(overrides)
    return PolicySpec(**base)


def _sensing(spec, **flags):
    return dataclasses.replace(spec, sensing=SensingSpec(**flags))


def test_a_matching_combination_passes_without_warnings(spec):
    report = check_compatibility(_policy(), spec, CAMERAS)
    assert report.warnings == []


def test_unsupported_control_mode_is_rejected(spec):
    policy = _policy(control_mode=ControlMode.EE_POSE_DELTA)
    with pytest.raises(Incompatible, match="ee_pose_delta"):
        check_compatibility(policy, spec, CAMERAS)


def test_missing_camera_is_rejected(spec):
    policy = _policy(cameras=[CameraRequirement("overhead", height=480, width=640)])
    with pytest.raises(Incompatible, match="hard contract"):
        check_compatibility(policy, spec, CAMERAS)


def test_camera_resolution_mismatch_is_rejected(spec):
    policy = _policy(cameras=[CameraRequirement("wrist", height=240, width=320)])
    with pytest.raises(Incompatible, match="320x240"):
        check_compatibility(policy, spec, CAMERAS)


def test_consuming_an_unsensed_field_is_rejected(spec):
    """so101 declares no current sensing, so a policy trained on current
    cannot run on it -- and must say so before step 0."""
    policy = _policy(obs_keys=["joint_pos", "joint_current"])
    with pytest.raises(Incompatible, match="sensing.joint_current: false"):
        check_compatibility(policy, spec, CAMERAS)


def test_consuming_a_sensed_field_passes(spec):
    policy = _policy(obs_keys=["joint_pos", "joint_current"])
    sensed = _sensing(spec, joint_current=True)
    assert check_compatibility(policy, sensed, CAMERAS).warnings == []


def test_rate_disagreement_is_a_warning_not_an_error(spec):
    report = check_compatibility(_policy(expected_control_hz=50.0), spec, CAMERAS)
    assert any("50" in w and "30" in w for w in report.warnings)


def test_spec_lineage_drift_is_a_warning(spec):
    """Expected after the Stage 9 spec revision -- and exactly the line you
    want in the log when evaluation numbers move."""
    policy = _policy(trained_against_robot_spec="deadbeefdeadbeef")
    report = check_compatibility(policy, spec, CAMERAS)
    assert any("trained against RobotSpec" in w for w in report.warnings)


def test_matching_spec_lineage_is_silent(spec):
    policy = _policy(trained_against_robot_spec=spec.content_hash())
    assert check_compatibility(policy, spec, CAMERAS).warnings == []


def test_force_limit_without_a_sensor_is_rejected(spec):
    """The v2 hole: a configured limit with nothing behind it never fires."""
    with pytest.raises(Incompatible, match="could never fire"):
        check_compatibility(None, spec, CAMERAS, safety_config={"max_current_a": 1.8})


def test_force_limit_with_a_sensor_passes(spec):
    sensed = _sensing(spec, joint_current=True)
    check_compatibility(None, sensed, CAMERAS, safety_config={"max_current_a": 1.8})


def test_safety_limits_are_checked_even_without_a_policy(spec):
    with pytest.raises(Incompatible, match="sensing.estop"):
        check_compatibility(None, spec, CAMERAS, safety_config={"estop": "/dev/tty.estop"})


def test_every_problem_is_reported_at_once(spec):
    policy = _policy(
        control_mode=ControlMode.JOINT_VELOCITY,
        cameras=[CameraRequirement("overhead", height=480, width=640)],
        obs_keys=["joint_torque"],
    )
    with pytest.raises(Incompatible) as excinfo:
        check_compatibility(policy, spec, CAMERAS, safety_config={"max_current_a": 2.0})
    message = str(excinfo.value)
    assert message.count("\n  - ") == 4, message


def test_camera_config_defaults_to_the_robot_spec(spec):
    assert check_compatibility(_policy(), spec).warnings == []
