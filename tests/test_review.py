"""The shared stop-and-ask-Claude step: what counts toward the per-episode limit."""

from __future__ import annotations

import numpy as np

from teleop_sim.control.safety.review import ClaudeReview
from teleop_sim.core.types import Observation
from teleop_sim.perception.vision import HazardReview

FRAME = np.zeros((4, 4, 3), np.uint8)


def obs() -> Observation:
    return Observation(images={"wrist": FRAME}, joint_pos=np.zeros(6), gripper=0.0, timestamp=0.0)


class Robot:
    def get_observation(self):
        return obs()


def test_a_wait_then_resume_is_one_alarm_and_the_limit_counts_alarms():
    answers = []

    def reviewer(images, context):
        answers.append(1)
        decision = "wait" if len(answers) % 2 else "resume"  # wait, resume, wait, resume...
        return HazardReview(decision, decision == "wait", 0.9, "fake", "fake")

    review = ClaudeReview({"max_alarms": 3}, reviewer, sleep=lambda s: None)
    review.bind(robot=Robot())
    review.reset()
    for _ in range(3):
        assert review.decide({}, obs(), ["wrist"], lambda held: "alarm") is None
    assert len(answers) == 6  # two reviews per alarm, all allowed
    why = review.decide({}, obs(), ["wrist"], lambda held: "alarm")
    assert why is not None and "max_alarms" in why
    assert len(answers) == 6  # the 4th alarm stops without asking


def test_claude_sees_the_start_of_the_run_for_reference():
    seen = {}

    def reviewer(images, context):
        seen.update(images=sorted(images), context=context)
        return HazardReview("resume", False, 0.9, "fake", "fake")

    review = ClaudeReview({}, reviewer, sleep=lambda s: None)
    review.reset()
    review.note_baseline(obs(), ["wrist"])
    review.note_baseline(
        Observation(images={"wrist": FRAME + 1}, joint_pos=np.zeros(6), gripper=0, timestamp=1),
        ["wrist"],
    )
    assert review.baseline["wrist"].max() == 0  # only the first frames are kept
    assert review.decide({}, obs(), ["wrist"], lambda held: "alarm") is None
    assert "wrist (at the start of the run, for reference)" in seen["images"]
    assert "belongs to the scene" in seen["context"]
