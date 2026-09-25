"""The VLM-guided pick / lift / place MVP, tested without a network or an arm.

* Camera geometry round-trips pixels and table points.
* The task picks, lifts and puts back a glass in MuJoCo with oracle vision.
* The same task, through the real config and the robots_realtime bridge,
  drives a MuJoCo rig speaking gem13's bus protocol.
* Claude answers are parsed and escalated correctly (stub client, no API).
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teleop_sim.core.types import TaskEvent
from teleop_sim.perception.camera import CameraModel
from teleop_sim.robots.kinematics import rot_z
from tests.conftest import requires_kronos

PACKAGE = Path(__file__).resolve().parent.parent / "teleop_sim"
SIM_CONFIG = PACKAGE / "configs" / "yam_kronos_pick_sim.yaml"
REAL_CONFIG = PACKAGE / "configs" / "yam_kronos_pick_real.yaml"
FAST = {"max_joint_speed": 0.8, "approach_speed": 0.3, "settle_seconds": 0.3, "hold_seconds": 0.5}


# ------------------------------------------------------------------ geometry


def test_a_pixel_round_trips_through_the_table_plane():
    rng = np.random.default_rng(0)
    K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])
    down = np.column_stack([[0.0, -1, 0], [1.0, 0, 0], [0, 0, 1.0]])
    for _ in range(20):
        cam = CameraModel(K, np.array([0.2, 0.0, 0.5]), rot_z(rng.uniform(-1, 1)) @ down, 640, 480)
        point = np.array([rng.uniform(0.1, 0.3), rng.uniform(-0.1, 0.1), 0.0])
        u, v = cam.project(point)
        assert cam.in_image(u, v)
        np.testing.assert_allclose(cam.intersect_z(u, v, 0.0), point, atol=1e-9)


def test_points_behind_the_camera_do_not_project():
    down = np.column_stack([[0.0, -1, 0], [1.0, 0, 0], [0, 0, 1.0]])
    cam = CameraModel(np.eye(3), np.array([0, 0, 0.5]), down, 640, 480)
    assert cam.project(np.array([0, 0, 1.0])) is None
    assert cam.intersect_z(0, 0, 1.0) is None


def test_task_events_carry_the_failure_tag():
    assert TaskEvent.failure("unreachable") == "task_failure:unreachable"
    assert TaskEvent.failure("") == TaskEvent.FAILURE


# ------------------------------------------------------------------ in sim

mujoco = pytest.importorskip("mujoco")

from teleop_sim.core.clock import ManualClock  # noqa: E402
from teleop_sim.core.config import RunConfig, build_system  # noqa: E402
from teleop_sim.core.types import Outcome  # noqa: E402
from teleop_sim.perception.vision import Detection, TargetPlan, Verdict, Vision  # noqa: E402


def _glass(robot, name: str) -> tuple[np.ndarray, float]:
    bid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, name)
    up = robot.data.xmat[bid].reshape(3, 3)[2, 2]
    return robot.data.xpos[bid].copy(), float(np.degrees(np.arccos(np.clip(up, -1, 1))))


def _sim_system(vision: dict, **policy):
    config = RunConfig.from_yaml(SIM_CONFIG)
    config.source["policy"].update(vision=vision, **policy)
    return build_system(config, clock=ManualClock())


@requires_kronos  # yam_kronos needs the private Kronos model
@pytest.mark.parametrize(
    "target,neighbour", [("glass_right", "glass_left"), ("glass_left", "glass_right")]
)
def test_the_task_picks_lifts_and_puts_back_a_glass(target, neighbour):
    system = _sim_system({"type": "oracle", "target_body": target})
    policy = system.source.policy
    policy.bind(robot=system.robot)
    start = {n: _glass(system.robot, n) for n in (target, neighbour)}

    result = system.loop.run_episode(seed=0)

    assert result.outcome is Outcome.SUCCESS, result.failure_tag
    pos, tilt = _glass(system.robot, target)
    assert np.linalg.norm(pos[:2] - start[target][0][:2]) < 0.025, "put back near where it was"
    assert tilt < 3.0, "and upright"
    assert pos[2] == pytest.approx(0.0, abs=2e-3), "and standing on the table"
    moved = np.linalg.norm(_glass(system.robot, neighbour)[0][:2] - start[neighbour][0][:2])
    assert moved < 0.002, "the other glass is untouched"


class NeverFinds(Vision):
    def plan(self, views, instruction):
        return TargetPlan(description="a glass")

    def locate(self, image, camera, target):
        return Detection(found=False, note="nothing there")

    def verify(self, views, question):
        return Verdict(ok=False, confidence=1.0)


@requires_kronos  # yam_kronos needs the private Kronos model
def test_a_task_that_cannot_find_its_target_ends_as_a_tagged_failure():
    system = _sim_system({"type": "oracle", "target_body": "glass_right"}, max_attempts=2, **FAST)
    policy = system.source.policy
    policy.bind(robot=system.robot)
    policy.vision = NeverFinds()

    result = system.loop.run_episode(seed=0)

    assert result.outcome is Outcome.FAILURE
    assert result.failure_tag == "attempts_exhausted"


# ------------------------------------------------------- the real path, against a sim rig

pytest.importorskip("zmq")
pytest.importorskip("msgpack_numpy")

from teleop_sim.robots.real.rr_wire import decode, encode  # noqa: E402
from teleop_sim.robots.real.sim_rig import SimRig  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_wire_format_round_trips_numpy():
    parts = encode(
        "hud/teleop/yam_right/joint_pos", {"joint_pos": np.arange(7.0)}, src="x", ts=12.5
    )
    topic, envelope = decode(parts)
    assert topic == "hud/teleop/yam_right/joint_pos"
    assert envelope["ts"] == 12.5 and envelope["src"] == "x"
    np.testing.assert_array_equal(envelope["data"]["joint_pos"], np.arange(7.0))


@requires_kronos  # yam_kronos needs the private Kronos model
def test_the_real_config_drives_a_rig_speaking_gem13s_protocol():
    pub, sub = _free_port(), _free_port()
    rig = SimRig(pub_port=pub, sub_port=sub).start()
    try:
        config = RunConfig.from_yaml(REAL_CONFIG)
        config.robot.update(host="127.0.0.1", pub_port=pub, sub_port=sub)
        # The real config is tuned to gem13's cell; the rig here is the sim
        # glasses table, so use its cell geometry. What is under test is the
        # bus path, not the grasp tuning.
        config.source["policy"].update(
            vision={"type": "oracle", "target_body": "glass_right"},
            survey_center=[0.40, 0.0],
            glass_height=0.10,
            grasp={"past_centre": 0.02, "height": 0.09, "pitch_deg": 45.0},
            workspace_xy={"lower": [0.2, -0.3], "upper": [0.6, 0.3]},
            **FAST,
        )
        system = build_system(config)
        system.source.policy.bind(robot=rig.robot)
        start = _glass(rig.robot, "glass_right")[0]

        result = system.loop.run_episode(seed=0)
        system.robot.disconnect()

        assert result.outcome is Outcome.SUCCESS, result.failure_tag
        assert rig.commands_applied > 100
        assert rig.last_command.shape == (7,)
        assert rig.last_command[6] == pytest.approx(1.0), "rr gripper 1.0 = open, after release"
        pos, tilt = _glass(rig.robot, "glass_right")
        assert np.linalg.norm(pos[:2] - start[:2]) < 0.025 and tilt < 3.0
    finally:
        rig.stop()


def test_the_bridge_refuses_to_start_without_a_rig():
    from teleop_sim.core.clock import WallClock
    from teleop_sim.core.spec import RobotSpec
    from teleop_sim.robots.real.rr_bridge import RobotLinkLost, RobotsRealtimeRobot

    spec = RobotSpec.from_yaml(PACKAGE / "robots" / "specs" / "yam_kronos.yaml")
    robot = RobotsRealtimeRobot(
        spec,
        WallClock(),
        host="127.0.0.1",
        pub_port=_free_port(),
        sub_port=_free_port(),
        cameras={},
        connect_timeout_s=0.5,
    )
    with pytest.raises(RobotLinkLost, match="rr-session running"):
        robot.reset()
    robot.disconnect()


# ------------------------------------------------------- Claude answers (stub client)

from teleop_sim.perception.claude_vision import ClaudeVision  # noqa: E402


class StubClient:
    """Answers messages.create with queued JSON, recording which model was asked."""

    def __init__(self, answers: list[dict]) -> None:
        self.answers = list(answers)
        self.models: list[str] = []
        self.messages = self

    def with_options(self, **_):
        return self

    def create(self, model, output_config, **_):
        assert output_config["format"]["type"] == "json_schema"
        self.models.append(model)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.answers.pop(0)))],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            _request_id="req_stub",
        )


def _locate_answer(confidence: float, found: bool = True) -> dict:
    return {
        "found": found,
        "base_x": 320.0,
        "base_y": 250.0,
        "rim_x": 318.0,
        "rim_y": 200.0,
        "confidence": confidence,
        "note": "",
    }


def _camera() -> CameraModel:
    return CameraModel(np.eye(3), np.zeros(3), np.eye(3), 640, 480)


def test_a_confident_fast_answer_is_used_as_is():
    client = StubClient([_locate_answer(0.9)])
    vision = ClaudeVision(client=client)
    det = vision.locate(np.zeros((480, 640, 3), np.uint8), _camera(), "the glass")
    assert det.found and det.base.u == 320.0 and det.rim.v == 200.0
    assert client.models == ["claude-sonnet-5"]


def test_an_unsure_fast_answer_is_escalated_to_the_smart_model():
    client = StubClient([_locate_answer(0.2), _locate_answer(0.8)])
    vision = ClaudeVision(client=client)
    det = vision.locate(np.zeros((480, 640, 3), np.uint8), _camera(), "the glass")
    assert client.models == ["claude-sonnet-5", "claude-opus-5-5"]
    assert det.model == "claude-opus-5-5"


def test_a_pixel_outside_the_image_is_not_a_detection():
    answer = _locate_answer(0.9) | {"base_x": 900.0}
    client = StubClient([answer, answer])
    det = ClaudeVision(client=client).locate(np.zeros((480, 640, 3), np.uint8), _camera(), "g")
    assert not det.found


def test_planning_asks_the_smart_model():
    client = StubClient([{"feasible": True, "target_description": "the right glass", "reason": ""}])
    plan = ClaudeVision(client=client).plan(
        SimpleNamespace(images={"wrist": np.zeros((4, 4, 3), np.uint8)}), "pick up a glass"
    )
    assert plan.description == "the right glass"
    assert client.models == ["claude-opus-5-5"]
