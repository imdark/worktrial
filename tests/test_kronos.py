"""The Kronos model agrees with the hardware evidence it was built from.

Each number here was established from the CAD, the slicer projects, the
datasheet or the teleop video, and is pinned so the model cannot drift from it
without a test saying so.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from teleop_sim.core.clock import WallClock
from teleop_sim.core.parts import SceneSpec
from teleop_sim.core.spec import RobotSpec
from tests.conftest import KRONOS_DIR, PACKAGE_ROOT, REPO_ROOT, requires_kronos

pytestmark = requires_kronos

mujoco = pytest.importorskip("mujoco", reason="needs the 'sim' extra")

from teleop_sim.envs.kronos_assets import (  # noqa: E402
    OPERATING_OPEN_RAD,
    build_kronos,
    load_extract,
)
from teleop_sim.envs.scripted_grasp import KRONOS_GRASP, GraspScript  # noqa: E402
from teleop_sim.robots.sim.mujoco_robot import MujocoRobot  # noqa: E402

KRONOS = KRONOS_DIR
YAM_KRONOS = PACKAGE_ROOT / "robots" / "specs" / "yam_kronos.yaml"
GLASSES = PACKAGE_ROOT / "scenes" / "glasses_table.yaml"


@pytest.fixture(scope="module")
def extract():
    return load_extract(KRONOS / "kronos_extract.json")


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(KRONOS / "kronos.xml"))


def _joint(model, side):
    return model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"kronos_finger_{side}")
    ]


def test_model_is_current_with_its_extraction(extract, model):
    spec = importlib.util.spec_from_file_location(
        "bkm", REPO_ROOT / "scripts" / "build_kronos_model.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    closed = build.measure_closing_angle(extract)
    assert (KRONOS / "kronos.xml").read_text() == build_kronos(extract, closed_rad=closed), (
        "kronos.xml is stale -- run `python scripts/build_kronos_model.py`"
    )


def test_pivots_are_the_servo_shafts_31mm_apart(extract):
    left, right = (np.array(extract["pivots_cad_mm"][s]) for s in ("left", "right"))
    assert np.linalg.norm(left - right) == pytest.approx(31.0, abs=0.1)
    assert left[1] == pytest.approx(right[1]) and left[2] == pytest.approx(right[2])


def test_masses_come_from_the_slicer_and_the_datasheet(extract, model):
    bodies = extract["bodies"]
    assert bodies["base"]["mass_kg"] * 1000 == pytest.approx(258.7, abs=0.2)
    for side in ("left", "right"):
        assert bodies[f"finger_{side}"]["mass_kg"] * 1000 == pytest.approx(38.0, abs=0.1)
    # base + 2 fingers + the (assumed) 60 g camera
    assert model.body_subtreemass[0] * 1000 == pytest.approx(394.7, abs=0.5)


def test_inertias_are_physical(extract):
    for name, body in extract["bodies"].items():
        eig = np.linalg.eigvalsh(np.array(body["inertia"]))
        assert (eig > 0).all(), name
        assert eig[2] <= eig[0] + eig[1] + 1e-12, f"{name}: violates the triangle inequality"


def test_torque_limit_is_two_xl430s_at_stall(model):
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "kronos_grip")
    assert model.actuator_forcerange[a][1] == pytest.approx(2 * 1.4)


def test_fingers_move_in_exact_mirror(model):
    """The finger hubs are meshed 1:1 gears (and the video agrees: +13.9 / -13.2 deg)."""
    d = mujoco.MjData(model)
    d.qpos[[_joint(model, "left"), _joint(model, "right")]] = OPERATING_OPEN_RAD
    d.ctrl[0] = (OPERATING_OPEN_RAD + model.jnt_range[0][1]) / 2
    for _ in range(2000):
        mujoco.mj_step(model, d)
    assert d.qpos[_joint(model, "left")] == pytest.approx(d.qpos[_joint(model, "right")], abs=1e-3)


def test_swing_matches_the_video(model):
    """Open (widest in the video) to closed (pads meet): 25.5 deg per finger in
    the video, 24.6 from the STL geometry. The model's pad shapes are slightly
    fuller than the real pads, so allow a few degrees."""
    closed = model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "kronos_finger_left")
    ][1]
    assert math.degrees(closed - OPERATING_OPEN_RAD) == pytest.approx(25.5, abs=3.5)


def test_every_mesh_fits_mujocos_limit():
    for path in (KRONOS / "meshes").glob("*.stl"):
        faces = int.from_bytes(path.read_bytes()[80:84], "little")
        assert faces < 200_000, path.name


@pytest.fixture
def robot():
    spec = RobotSpec.from_yaml(YAM_KRONOS).with_scene(SceneSpec.from_yaml(GLASSES))
    robot = MujocoRobot(spec, WallClock(), render_cameras=[])
    robot.reset(seed=0)
    return robot


def test_the_wrist_camera_looks_at_the_fingertips(robot):
    m, d = robot.model, robot.data
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "kronos_grasp")
    view = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
    to_tips = d.site_xpos[site] - d.cam_xpos[cam]
    angle = math.degrees(math.acos(view @ to_tips / np.linalg.norm(to_tips)))
    assert angle < 12.0, f"camera is {angle:.1f} deg off the fingertips"


def test_the_neck_is_up_and_the_fingers_close_sideways_at_home(robot):
    r = robot.data.xmat[mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "kronos")].reshape(
        3, 3
    )
    assert r[:, 1] @ [0, 0, 1] > 0.99, "camera neck should point up"
    assert abs(r[:, 0] @ [0, 0, 1]) < 0.01, "fingers should close horizontally"


def test_kronos_lifts_a_glass(robot):
    """The side grasp found by sweep. Kronos lifts in 12 of 16 configurations;
    tilt is its failure mode (a V-gripper pinches near its tips on one line)."""
    script = GraspScript(robot)
    neighbour = script.body_pos("glass_left")[:2]
    assert script.pick("glass_right", KRONOS_GRASP)
    assert script.body_pos("glass_right")[2] > 0.08
    assert script.body_tilt_deg("glass_right") < 10.0
    assert np.linalg.norm(script.body_pos("glass_left")[:2] - neighbour) < 0.005
