"""The YAM glasses environment admits the task it is built for.

Everything here was first found by looking and measuring, and each test pins
one of those findings so it cannot quietly regress.
"""

from __future__ import annotations

import numpy as np
import pytest

from teleop_sim.core.clock import WallClock
from teleop_sim.core.parts import SceneSpec
from teleop_sim.core.spec import RobotSpec
from tests.conftest import PACKAGE_ROOT

mujoco = pytest.importorskip("mujoco", reason="needs the 'sim' extra")

from teleop_sim.envs.scripted_grasp import GraspScript  # noqa: E402
from teleop_sim.robots.sim.mujoco_robot import MujocoRobot  # noqa: E402

YAM_SPEC = PACKAGE_ROOT / "robots" / "specs" / "yam.yaml"
GLASSES_TABLE = PACKAGE_ROOT / "scenes" / "glasses_table.yaml"
GLASSES = ("glass_left", "glass_right")


@pytest.fixture
def robot():
    spec = RobotSpec.from_yaml(YAM_SPEC).with_scene(SceneSpec.from_yaml(GLASSES_TABLE))
    robot = MujocoRobot(spec, WallClock(), render_cameras=[])
    robot.reset(seed=0)
    return robot


def _body(robot, name):
    return mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, name)


def _settle(robot, steps=600):
    for _ in range(steps):
        mujoco.mj_step(robot.model, robot.data)


# ------------------------------------------------------------------ the glasses


def test_two_glasses_stand_upright_on_the_table(robot):
    _settle(robot)
    script = GraspScript(robot)
    for glass in GLASSES:
        assert script.body_pos(glass)[2] == pytest.approx(0.0, abs=1e-3), glass
        assert script.body_tilt_deg(glass) < 0.5, glass


def test_glasses_are_near_each_other_without_touching(robot):
    script = GraspScript(robot)
    left, right = (script.body_pos(g)[:2] for g in GLASSES)
    centre_gap = float(np.linalg.norm(left - right))
    wall_gap = centre_gap - 2 * 0.0325
    assert 0.02 < wall_gap < 0.08, f"{wall_gap * 1000:.0f} mm between the walls"


def test_glasses_weigh_what_a_tumbler_weighs(robot):
    for glass in GLASSES:
        mass = robot.model.body_subtreemass[_body(robot, glass)]
        assert 0.15 < mass < 0.30, f"{glass}: {mass * 1000:.0f} g"


def test_a_glass_fits_between_the_open_fingers(robot):
    """The pads open to ~76 mm; the glasses are sized to leave clearance."""
    m, d = robot.model, robot.data
    pads = {
        side: [g for g in range(m.ngeom)
               if m.geom_bodyid[g] == _body(robot, f"{side}_down")
               and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        for side in ("lf", "rf")
    }
    fromto = np.zeros(6)
    opening = min(
        mujoco.mj_geomDistance(m, d, a, b, 1.0, fromto) for a in pads["lf"] for b in pads["rf"]
    )
    glass_width = 2 * 0.0325
    assert opening > glass_width + 2 * 0.004, f"opening {opening * 1000:.1f} mm"


# ----------------------------------------------------------------- arm and camera


def test_home_pose_puts_the_camera_on_top_and_the_fingers_level(robot):
    """The finger slide axis is link_6 x and the camera sits on its +y face.
    Measured, after inferring it from body positions got it backwards."""
    r6 = robot.data.xmat[_body(robot, "yam_linear")].reshape(3, 3)
    assert r6[:, 1] @ [0, 0, 1] > 0.99, "camera face should point up"
    assert abs(r6[:, 0] @ [0, 0, 1]) < 0.01, "fingers should close horizontally"


def test_wrist_camera_looks_along_the_fingers(robot):
    m, d = robot.model, robot.data
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    view = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
    tool = d.xmat[_body(robot, "yam_linear")].reshape(3, 3)[:, 2]
    assert view @ tool > 0.8
    assert d.cam_xpos[cam][2] > d.xpos[_body(robot, "link_6")][2], "camera above the wrist"


def test_camera_housing_touches_nothing_at_rest(robot):
    m, d = robot.model, robot.data
    housing = _body(robot, "wrist_d405")
    touching = [
        c for c in range(d.ncon)
        if housing in (m.geom_bodyid[d.contact[c].geom1], m.geom_bodyid[d.contact[c].geom2])
    ]
    assert not touching


def test_friction_cone_is_elliptic(robot):
    """With the default pyramidal cone a held glass creeps out of the grip."""
    assert robot.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert robot.model.opt.impratio >= 10


# ------------------------------------------------------------------- the task


@pytest.mark.parametrize("glass", GLASSES)
def test_both_glasses_are_reachable_with_level_fingers(robot, glass):
    assert GraspScript(robot).plan(glass) is not None


def test_a_scripted_side_grasp_lifts_a_glass(robot):
    """The environment admits the task: reach, close, lift 100 mm, upright,
    without disturbing the neighbour. Margins are thin -- this is a proof
    that a grasp exists, not a robust grasp policy (that is Stage 4)."""
    script = GraspScript(robot)
    neighbour_before = script.body_pos("glass_left")[:2]

    assert script.pick("glass_right")

    assert script.body_pos("glass_right")[2] > 0.08, "glass should be lifted"
    assert script.body_tilt_deg("glass_right") < 10.0, "and still upright"
    moved = np.linalg.norm(script.body_pos("glass_left")[:2] - neighbour_before)
    assert moved < 0.005, f"neighbour moved {moved * 1000:.1f} mm"


# ------------------------------------------------------------------ appearance

#: Mean pixel brightness right after reset (home pose, gripper open), measured
#: on the model as it was before the arm / end effector / scene split. A policy
#: learns from these pixels, so a lost light is a real regression -- and the
#: split did lose one (upstream's wrist-tracking spotlight: -13%) while every
#: kinematic and dynamic test stayed green. The fixed replacement lands within
#: +0.3% (top) and -2.9% (wrist, which the tracking light favoured).
REFERENCE_BRIGHTNESS = {"top": 171.05, "wrist": 82.24}


@pytest.mark.parametrize("camera", sorted(REFERENCE_BRIGHTNESS))
def test_scene_lighting_matches_the_reference(camera):
    spec = RobotSpec.from_yaml(YAM_SPEC).with_scene(SceneSpec.from_yaml(GLASSES_TABLE))
    robot = MujocoRobot(spec, WallClock(), image_size=(320, 240), render_cameras=[camera])
    image = robot.reset(seed=0).images[camera]
    assert image.mean() == pytest.approx(REFERENCE_BRIGHTNESS[camera], rel=0.05)
