"""The MJCF and the RobotSpec must describe the same robot.

Two files declare the same joint limits, the same gripper travel and the same
cameras. When they drift, nothing raises: commands clip against a bound the
model does not have, and it looks exactly like bad tracking. This is the test
the Stage 1 note asks for -- "correct the limits against the MJCF" made
permanent instead of done once.
"""

from __future__ import annotations

import numpy as np
import pytest

from teleop_sim.core.spec import RobotSpec
from tests.conftest import SPEC_PATH

mujoco = pytest.importorskip("mujoco", reason="needs the 'sim' extra")

LIMIT_TOL = 1e-6
POSE_TOL = 1e-3  # 1 mm


SIM_SPECS = [SPEC_PATH, SPEC_PATH.parent / "yam.yaml"]


@pytest.fixture(scope="module", params=SIM_SPECS, ids=lambda p: p.stem)
def sim_spec(request):
    return RobotSpec.from_yaml(request.param)


@pytest.fixture(scope="module")
def model(sim_spec):
    return mujoco.MjModel.from_xml_path(sim_spec.mjcf_path)


def _jid(model, name):
    ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert ident >= 0, f"joint {name!r} is in the spec but not in the model"
    return ident


def _aid(model, name):
    ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    assert ident >= 0, f"actuator {name!r} is in the spec but not in the model"
    return ident


def test_every_spec_joint_has_matching_limits(model, sim_spec):
    for joint in sim_spec.joints:
        low, high = model.jnt_range[_jid(model, joint.name)]
        assert low == pytest.approx(joint.lower, abs=LIMIT_TOL), joint.name
        assert high == pytest.approx(joint.upper, abs=LIMIT_TOL), joint.name


def test_every_spec_joint_has_an_actuator_over_the_same_range(model, sim_spec):
    """A ctrlrange narrower than the joint range silently caps the arm."""
    for joint in sim_spec.joints:
        low, high = model.actuator_ctrlrange[_aid(model, joint.name)]
        assert low == pytest.approx(joint.lower, abs=LIMIT_TOL), joint.name
        assert high == pytest.approx(joint.upper, abs=LIMIT_TOL), joint.name


def test_gripper_range_lies_within_the_joint_range(model, sim_spec):
    """The spec's open/closed must be positions the jaw can physically occupy."""
    gripper = sim_spec.gripper
    low, high = model.jnt_range[_jid(model, gripper.joint)]
    for value in (gripper.open_pos, gripper.closed_pos):
        assert low - LIMIT_TOL <= value <= high + LIMIT_TOL


def test_gripper_actuator_can_command_both_extremes(model, sim_spec):
    """ctrlrange must cover the spec's open/closed, or one end is unreachable
    and the normalised reading can never get there. Where it overshoots the
    joint range (YAM's opens to 0.041 against a 0.0375 stop), commanding that
    extreme drives the jaw into a hard stop -- which is what makes the reading
    saturate exactly. The other end relies on GripperSpec.deadband."""
    gripper = sim_spec.gripper
    ctrl_low, ctrl_high = model.actuator_ctrlrange[_aid(model, gripper.actuator_name)]
    for value in (gripper.open_pos, gripper.closed_pos):
        assert ctrl_low - LIMIT_TOL <= value <= ctrl_high + LIMIT_TOL


def test_home_matches_the_model_keyframe_when_there_is_one(model, sim_spec):
    """A spec home that quietly disagrees with the model's own home keyframe is
    two definitions of 'where the robot starts'."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0 or sim_spec.home is None:
        pytest.skip("no home keyframe, or no declared home")
    qpos = model.key_qpos[kid]
    for joint, value in zip(sim_spec.joints, sim_spec.home, strict=True):
        adr = model.jnt_qposadr[_jid(model, joint.name)]
        assert qpos[adr] == pytest.approx(value, abs=1e-3), joint.name


def test_end_effector_site_exists_when_declared(model, sim_spec):
    if sim_spec.ee_site is None:
        pytest.skip("no ee_site declared")
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sim_spec.ee_site) >= 0


def test_end_effector_body_exists(model, sim_spec):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sim_spec.ee_link) >= 0


def test_every_spec_camera_exists_with_matching_fov(model, sim_spec):
    for camera in sim_spec.cameras:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)
        assert cid >= 0, f"camera {camera.name!r} is in the spec but not in the model"
        assert model.cam_fovy[cid] == pytest.approx(camera.fovy_deg, abs=1e-6)


def test_camera_mounts_and_poses_agree(model, sim_spec):
    """Full pose, including orientation.

    Stage 9's hand-eye calibration will correct these against the real rig; the
    point of the check is that the correction lands in both files or in
    neither. A spec whose camera pose quietly disagrees with the model is a
    sim2real gap nobody is looking for."""
    for camera in sim_spec.cameras:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.cam_bodyid[cid])
        assert body == camera.mount, f"camera {camera.name!r} mount"
        np.testing.assert_allclose(
            model.cam_pos[cid], camera.pose[:3], atol=POSE_TOL, err_msg=camera.name
        )
        np.testing.assert_allclose(
            model.cam_quat[cid], camera.pose[3:], atol=1e-5, err_msg=camera.name
        )


def test_control_rate_divides_into_the_physics_timestep(sim_spec, model):
    """A control period that is not a whole number of substeps makes every
    episode's timing subtly irreproducible."""
    period = 1.0 / sim_spec.control_hz
    substeps = period / model.opt.timestep
    assert substeps == pytest.approx(round(substeps), abs=1e-9), (
        f"control_hz {sim_spec.control_hz} over timestep {model.opt.timestep} "
        f"gives {substeps} substeps"
    )
