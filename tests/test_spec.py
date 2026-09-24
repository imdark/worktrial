"""Validation rung 2: a spec loads, validates, and fails loudly when wrong."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from teleop_sim.core.spec import (
    CameraSpec,
    GripperSpec,
    JointSpec,
    RobotSpec,
    SpecError,
    WorkspaceBounds,
)
from teleop_sim.core.types import ControlMode


def test_spec_loads_from_yaml(spec):
    assert spec.name == "so101"
    assert spec.dof == 5
    assert spec.joint_names[0] == "shoulder_pan"
    assert spec.camera_names == ["wrist", "front"]
    assert spec.default_control_mode is ControlMode.JOINT_POSITION
    assert spec.workspace is not None


def test_clip_joints_respects_limits(spec):
    clipped = spec.clip_joints(np.full(spec.dof, 100.0))
    np.testing.assert_allclose(clipped, spec.upper_limits)


def test_clip_joints_rejects_wrong_dof(spec):
    with pytest.raises(SpecError, match="expected 5 joint values"):
        spec.clip_joints(np.zeros(3))


def test_neutral_joints_inside_limits(spec):
    q = spec.neutral_joints()
    assert np.all(q >= spec.lower_limits) and np.all(q <= spec.upper_limits)


def test_gripper_normalisation_round_trip(spec):
    low, high = spec.gripper.open_pos, spec.gripper.closed_pos
    for native in (low, (low + high) / 2, high):
        assert spec.gripper.denormalize(spec.gripper.normalize(native)) == pytest.approx(
            native
        )
    assert spec.gripper.normalize(low - 10.0) == 0.0
    assert spec.gripper.normalize(high + 10.0) == 1.0


def test_camera_intrinsics_shape(spec):
    k = spec.camera("wrist").intrinsics()
    assert k.shape == (3, 3)
    assert k[0, 2] == pytest.approx(320.0)


def test_unknown_camera_lists_alternatives(spec):
    with pytest.raises(SpecError, match=r"\['wrist', 'front'\]"):
        spec.camera("overhead")


def test_degrees_in_limits_are_caught():
    # lower/upper inverted is the classic symptom of degrees-vs-radians.
    with pytest.raises(SpecError, match="RADIANS"):
        JointSpec(name="j0", lower=90.0, upper=-90.0)


def test_duplicate_joint_names_rejected():
    joints = [JointSpec("j0", -1.0, 1.0), JointSpec("j0", -1.0, 1.0)]
    with pytest.raises(SpecError, match="duplicate joint names"):
        RobotSpec(
            name="dup",
            joints=joints,
            ee_link="ee",
            gripper=GripperSpec("g", 0.0, 1.0),
        )


def test_invalid_fov_rejected():
    with pytest.raises(SpecError, match="fovy_deg"):
        CameraSpec(name="bad", fovy_deg=0.0)


def test_invalid_pose_length_rejected():
    with pytest.raises(SpecError, match="7 values"):
        CameraSpec(name="bad", pose=(0.0, 0.0, 0.0))


def test_workspace_contains():
    bounds = WorkspaceBounds(lower=(-1.0, -1.0, 0.0), upper=(1.0, 1.0, 1.0))
    assert bounds.contains(np.array([0.0, 0.0, 0.5]))
    assert not bounds.contains(np.array([2.0, 0.0, 0.5]))


def test_missing_spec_file_names_the_path(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SpecError, match="nope.yaml"):
        RobotSpec.from_yaml(missing)


def test_unknown_field_is_reported(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: bad\n"
        "ee_link: ee\n"
        "typo_field: 3\n"
        "joints: [{name: j0, lower: -1.0, upper: 1.0}]\n"
        "gripper: {joint: g, open_pos: 0.0, closed_pos: 1.0}\n"
    )
    with pytest.raises(SpecError, match="unknown or missing field"):
        RobotSpec.from_yaml(path)


# ------------------------------------------------ v3: modes, sensing, hashing


def test_supported_modes_default_to_the_default_mode(spec):
    assert spec.supported_modes == [ControlMode.JOINT_POSITION]
    assert spec.supports(ControlMode.JOINT_POSITION)
    assert not spec.supports(ControlMode.EE_POSE_DELTA)


def test_default_mode_must_be_supported():
    with pytest.raises(SpecError, match="not in supported_modes"):
        RobotSpec(
            name="bad",
            joints=[JointSpec("j0", -1.0, 1.0)],
            ee_link="ee",
            gripper=GripperSpec("g", 0.0, 1.0),
            default_control_mode=ControlMode.JOINT_POSITION,
            supported_modes=[ControlMode.EE_POSE_DELTA],
        )


def test_sensing_defaults_to_nothing_measurable(spec):
    """Position-controlled hobby servos report neither force nor current, and
    the spec must say so -- a safety limit with no sensor never fires."""
    assert not spec.sensing.joint_current
    assert not spec.sensing.joint_torque
    assert not spec.sensing.ee_wrench
    assert not spec.sensing.provides("estop")


def test_content_hash_is_stable_and_ignores_file_location(spec, tmp_path):
    """A hash that moves when a checkout moves cannot tell you whether two
    datasets were recorded against the same robot."""
    copy = tmp_path / "elsewhere.yaml"
    text = spec.source_path.read_text().replace(
        "../../../assets/so101/scene.xml", str(Path(spec.mjcf_path))
    )
    copy.write_text(text)
    relocated = RobotSpec.from_yaml(copy)

    assert relocated.source_path != spec.source_path
    assert relocated.mjcf_path == spec.mjcf_path
    assert relocated.content_hash() == spec.content_hash()


def test_content_hash_changes_when_content_changes(spec):
    import dataclasses

    nudged = dataclasses.replace(spec, control_hz=50.0)
    assert nudged.content_hash() != spec.content_hash()


def test_content_hash_follows_mjcf_includes(spec, tmp_path):
    """scene.xml's own bytes say nothing about the arm it includes, so editing
    a joint limit in so101.xml must still move the hash."""
    import shutil

    assets = tmp_path / "so101"
    shutil.copytree(Path(spec.mjcf_path).parent, assets)
    copy = tmp_path / "spec.yaml"
    copy.write_text(
        spec.source_path.read_text().replace(
            "../../../assets/so101/scene.xml", str(assets / "scene.xml")
        )
    )
    before = RobotSpec.from_yaml(copy).content_hash()

    included = assets / "so101.xml"
    included.write_text(included.read_text().replace('range="-1.92 1.92"', 'range="-1.5 1.5"'))
    after = RobotSpec.from_yaml(copy).content_hash()

    assert before != after
