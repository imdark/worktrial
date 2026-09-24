"""Scene object handles -- concrete and sim-only by design."""

from __future__ import annotations

import numpy as np
import pytest

from teleop_sim.core.clock import WallClock
from teleop_sim.core.spec import RobotSpec
from teleop_sim.envs.scene import Scene, SceneError
from tests.conftest import SPEC_PATH

pytest.importorskip("mujoco", reason="needs the 'sim' extra")


@pytest.fixture
def scene():
    from teleop_sim.robots.sim.mujoco_robot import MujocoRobot

    robot = MujocoRobot(RobotSpec.from_yaml(SPEC_PATH), WallClock(), render_cameras=[])
    robot.reset(seed=0)
    return Scene(robot.model, robot.data)


def test_only_free_bodies_are_movable(scene):
    assert scene.movable_bodies == ["cube_red"]


def test_set_and_read_back_a_pose(scene):
    target = np.array([0.22, -0.05, 0.02])
    scene.set_body_pose("cube_red", target)
    pos, quat = scene.body_pose("cube_red")
    np.testing.assert_allclose(pos, target, atol=1e-9)
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_a_placed_object_settles_onto_the_table(scene):
    scene.set_body_pose("cube_red", [0.20, 0.0, 0.25])
    scene.settle(steps=600)
    pos, _ = scene.body_pose("cube_red")
    assert pos[2] == pytest.approx(0.018, abs=3e-3), "cube should rest on its half-height"
    assert np.linalg.norm(pos[:2] - np.array([0.20, 0.0])) < 0.05


def test_teleporting_clears_stale_velocity(scene):
    """A body moved without zeroing its velocity shoots off on the next step."""
    scene.set_body_pose("cube_red", [0.20, 0.0, 0.30])
    scene.settle(steps=300)
    scene.set_body_pose("cube_red", [0.20, 0.0, 0.05])
    pos_before, _ = scene.body_pose("cube_red")
    scene.settle(steps=1)
    pos_after, _ = scene.body_pose("cube_red")
    assert abs(pos_after[2] - pos_before[2]) < 1e-3


def test_welded_body_is_refused_with_alternatives(scene):
    with pytest.raises(SceneError, match=r"movable bodies are \['cube_red'\]"):
        scene.body_pose("gripper_link")
