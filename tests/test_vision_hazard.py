"""The vision safety loop, with a fake model in place of laya-vision."""

from __future__ import annotations

import time

import numpy as np
import pytest

import teleop_sim.builtins  # noqa: F401
from teleop_sim.control.safety.vision_hazard import (
    DEFAULT_CHECKS,
    HazardCheck,
    HazardTracker,
    VisionHazardMonitor,
    with_hazard_monitor,
)
from teleop_sim.core.registry import SAFETY, build
from teleop_sim.core.types import Observation, Outcome
from teleop_sim.perception.vision import HazardReview

CLEAR = np.zeros((8, 8, 3), dtype=np.uint8)
HAND = np.full((8, 8, 3), 200, dtype=np.uint8)  # the fake model "sees" a hand in bright frames


def obs(wrist=CLEAR, overview=CLEAR, t=0.0) -> Observation:
    return Observation(
        images={"wrist": wrist, "overview": overview},
        joint_pos=np.zeros(6),
        gripper=0.0,
        timestamp=t,
    )


def fake_model(image, questions):
    p = 0.95 if image.mean() > 100 else 0.1
    return {name: p for name in questions}


class FakeRobot:
    stopped = False
    looks = 0
    frame = CLEAR

    def safe_stop(self):
        self.stopped = True

    def get_observation(self):
        self.looks += 1
        return obs(overview=self.frame)


class FakeReviewer:
    """Answers with the next of ``decisions``: (decision, hazard, confidence)."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.calls = []

    def __call__(self, images, context):
        self.calls.append((sorted(images), context))
        if not self.decisions:
            raise AssertionError("reviewer asked more often than expected")
        decision, hazard, confidence = self.decisions.pop(0)
        if decision == "error":
            raise TimeoutError("API down")
        return HazardReview(decision, hazard, confidence, reason=f"fake {decision}", model="fake")


def reviewed(reviewer, robot=None, **kwargs):
    now = [0.0]

    def sleep(s):
        now[0] += s

    monitor = VisionHazardMonitor(
        asker=fake_model,
        reviewer=reviewer,
        clock=lambda: now[0],
        sleep=sleep,
        max_blind_s=1e9,
        **kwargs,
    )
    monitor.bind(robot=robot, context="Picking up the frosted cup.")
    monitor.reset()
    return monitor


def drive_hand_until_review(monitor, reviewer, steps=2000):
    """Show a hand to the overview camera until Claude has been asked (or a verdict)."""
    for step in range(steps):
        verdict = monitor.check(obs(overview=HAND), step)
        if verdict is not None or reviewer.calls:
            return verdict
        time.sleep(0.002)
    raise AssertionError("no alarm was raised")


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_it_takes_three_confident_frames_in_a_row_to_raise_an_alarm():
    tracker = HazardTracker([HazardCheck("hand", "overview", "hand?", trip_above=0.6)])
    assert tracker.update("overview", {"hand": 0.9}) is None
    assert tracker.update("overview", {"hand": 0.9}) is None
    assert tracker.update("overview", {"hand": 0.2}) is None  # broken streak
    assert tracker.update("overview", {"hand": 0.9}) is None
    assert tracker.update("overview", {"hand": 0.8}) is None
    trip = tracker.update("overview", {"hand": 0.7})
    assert trip is not None and trip.check == "hand" and trip.history == [0.9, 0.8, 0.7]


def test_answers_for_one_camera_do_not_count_for_another():
    tracker = HazardTracker(DEFAULT_CHECKS)
    assert set(tracker.questions_for("wrist")) == {"hand_wrist"}
    assert set(tracker.questions_for("overview")) == {"hand_overview", "reach_overview"}
    assert tracker.update("wrist", {"hand_overview": 0.99}) is None
    assert tracker.update("wrist", {"hand_overview": 0.99}) is None


def test_duplicate_check_names_are_refused():
    with pytest.raises(ValueError, match="unique"):
        HazardTracker([HazardCheck("a", "wrist", "q"), HazardCheck("a", "overview", "q")])


def test_a_hand_in_view_stops_the_arm(tmp_path):
    monitor = VisionHazardMonitor(asker=fake_model, log_dir=str(tmp_path))
    monitor.reset()
    try:
        for step in range(3):
            assert monitor.check(obs(), step) is None
        assert wait_for(lambda: monitor.n_answers >= 2)
        verdict = None
        for step in range(3, 400):
            verdict = monitor.check(obs(overview=HAND), step)
            if verdict is not None:
                break
            time.sleep(0.005)
        assert verdict is not None
        assert verdict.outcome is Outcome.WATCHDOG and verdict.safe_stop
        assert verdict.tag in {"hazard:hand_overview", "hazard:reach_overview"}
        robot = FakeRobot()
        monitor.on_trip(robot, verdict)
        assert robot.stopped
        log = (tmp_path / "hazard_checks.jsonl").read_text()
        assert '"event": "trip"' in log
        assert list(tmp_path.glob("*_alarm_hand_overview.jpg"))
    finally:
        monitor.close()


def test_a_model_that_stops_answering_trips_as_blind():
    now = [0.0]
    calls = []

    def dies_after_preflight(image, questions):
        calls.append(1)
        if len(calls) > 1:
            raise TimeoutError("server hung")
        return fake_model(image, questions)

    monitor = VisionHazardMonitor(asker=dies_after_preflight, max_blind_s=2.0, clock=lambda: now[0])
    monitor.reset()
    try:
        assert monitor.check(obs(), 0) is None
        assert wait_for(lambda: len(calls) > 1)
        now[0] = 1.0
        assert monitor.check(obs(), 1) is None
        now[0] = 2.5
        verdict = monitor.check(obs(), 2)
        assert verdict is not None and verdict.tag == "hazard_monitor_blind" and verdict.safe_stop
        assert "server hung" in verdict.detail
    finally:
        monitor.close()


def test_an_unreachable_model_refuses_to_start():
    def down(image, questions):
        raise ConnectionError("refused")

    monitor = VisionHazardMonitor(asker=down)
    with pytest.raises(RuntimeError, match="laya_server.py"):
        monitor.reset()


def test_the_inner_watchdog_still_applies():
    monitor = build(
        SAFETY,
        {"type": "vision_hazard", "inner": {"type": "sim_watchdog", "max_steps": 5}},
        spec=None,
    )
    monitor.ask = fake_model
    monitor.reset()
    try:
        verdict = monitor.check(obs(), 5)
        assert verdict is not None and verdict.tag == "max_steps"
    finally:
        monitor.close()


def test_claude_calls_a_false_alarm_and_the_arm_carries_on():
    reviewer = FakeReviewer(("resume", False, 0.9))
    monitor = reviewed(reviewer, robot=FakeRobot())
    try:
        assert drive_hand_until_review(monitor, reviewer) is None
        images, context = reviewer.calls[0]
        assert "overview (frame that raised the alarm)" in images and "wrist (now)" in images
        assert "frosted cup" in context and "3 frames in a row" in context
        # the streak was cleared: clear frames keep it running
        for step in range(20):
            assert monitor.check(obs(), 3000 + step) is None
    finally:
        monitor.close()


def test_claude_waits_for_the_hand_to_leave_then_resumes():
    robot = FakeRobot()
    reviewer = FakeReviewer(("wait", True, 0.9), ("wait", True, 0.9), ("resume", False, 0.95))
    monitor = reviewed(reviewer, robot=robot, review={"wait_s": 3.0})
    try:
        assert drive_hand_until_review(monitor, reviewer) is None
        assert len(reviewer.calls) == 3 and robot.looks == 2  # fresh frames each look
        assert not robot.stopped
    finally:
        monitor.close()


def test_an_unsure_resume_is_a_wait():
    robot = FakeRobot()
    reviewer = FakeReviewer(("resume", False, 0.5), ("resume", False, 0.9))
    monitor = reviewed(reviewer, robot=robot, review={"resume_confidence": 0.7})
    try:
        assert drive_hand_until_review(monitor, reviewer) is None
        assert len(reviewer.calls) == 2 and robot.looks == 1
    finally:
        monitor.close()


@pytest.mark.parametrize(
    "decisions, why",
    [
        ([("abort", True, 0.9)], "Claude: fake abort"),
        ([("error", False, 0.0)], "could not review"),
        ([("wait", True, 0.9)] * 30, "still unclear after 10s"),
    ],
)
def test_the_arm_stops_when_claude_says_so_cannot_answer_or_the_hazard_stays(decisions, why):
    reviewer = FakeReviewer(*decisions)
    monitor = reviewed(reviewer, robot=FakeRobot(), review={"wait_s": 3.0, "max_hold_s": 10.0})
    try:
        verdict = drive_hand_until_review(monitor, reviewer)
        assert verdict is not None and verdict.safe_stop and verdict.tag.startswith("hazard:")
        assert why in verdict.detail
    finally:
        monitor.close()


def test_the_real_config_runs_without_it_unless_asked():
    from pathlib import Path

    from teleop_sim.core.config import RunConfig

    root = Path(__file__).resolve().parent.parent / "teleop_sim" / "configs"
    config = RunConfig.from_yaml(root / "yam_kronos_pick_real.yaml")
    assert config.safety["type"] == "sim_watchdog"
    wrapped = with_hazard_monitor(config.safety)
    assert wrapped["type"] == "vision_hazard" and wrapped["inner"] == config.safety
    assert wrapped["review"]["type"] == "claude"
