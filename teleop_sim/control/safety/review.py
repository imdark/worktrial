"""Stop, let Claude look, then resume, wait or abort: shared by the safety monitors.

Both the vision hazard monitor and the motor-torque contact monitor end the
same way. The arm is already holding (the monitor's ``check`` blocks, so the
loop sends nothing and the rig holds its pose). Claude is shown the frame
from the moment of the alarm and fresh frames, told what the detector saw,
and decides:

* resume -- no hazard, no collision: the policy carries on where it stopped;
* wait   -- something is there but may go away: hold, look again in wait_s;
* abort  -- stop for good: the monitor safe-stops and the episode ends.

A "resume" below ``resume_confidence`` counts as a wait. Claude failing to
answer, still unclear after ``max_hold_s`` (one alarm), or more than
``max_alarms`` alarms in one episode all abort: with no second opinion, the arm stays
stopped.

    review:
      type: claude
      fast_model: claude-sonnet-5     # decides; an unsure "resume" goes to Opus
      smart_model: claude-opus-5-5
      escalate_below: 0.8
      resume_confidence: 0.7
      wait_s: 3.0
      max_hold_s: 60.0
      max_alarms: 6
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from teleop_sim.core.protocols import Robot
from teleop_sim.core.types import Observation
from teleop_sim.perception.vision import HazardReview
from teleop_sim.runlog import RunLog

#: (images, context) -> HazardReview; raises on failure.
Reviewer = Callable[[dict[str, np.ndarray], str], HazardReview]

DEFAULT_CONTEXT = (
    "The robot is picking up a cup from the table, lifting it and putting it back. Cups "
    "on the table, the robot's own gripper and its cables are expected."
)


def task_context(instruction: str) -> str:
    """What the reviewer should expect to see, for a pick-lift-place instruction."""
    return (
        f"The robot's task: {instruction}. It picks that object up, lifts it and puts it "
        "down again. Other objects already on the table, the robot's own gripper, its "
        "cables and mounts are expected; nothing else should be in or near its path."
    )


def claude_reviewer(run_log: RunLog | None = None, **vision_kwargs: Any) -> Reviewer:
    from teleop_sim.perception.claude_vision import ClaudeVision

    vision = ClaudeVision(run_log=run_log, **vision_kwargs)
    return vision.review_hazard


class ClaudeReview:
    """The pause-and-decide step. ``decide`` returns None to resume, or why to abort."""

    def __init__(
        self,
        review: dict[str, Any] | None,
        reviewer: Reviewer | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        name: str = "safety",
    ) -> None:
        cfg = dict(review) if review is not None else None
        self.wait_s = float((cfg or {}).pop("wait_s", 3.0))
        self.max_hold_s = float((cfg or {}).pop("max_hold_s", 60.0))
        # How many alarms one episode may have before the arm stays stopped. Each
        # alarm may take several reviews (wait, wait, resume); max_hold_s bounds those.
        self.max_alarms = int((cfg or {}).pop("max_alarms", 6))
        self.resume_confidence = float((cfg or {}).pop("resume_confidence", 0.7))
        if cfg is not None:
            kind = cfg.pop("type", "claude")
            if kind != "claude":
                raise ValueError(f"{name}: unknown review type {kind!r}")
        self.vision_cfg = cfg  # what is left goes to ClaudeVision
        self._reviewer = reviewer
        self._own_reviewer = reviewer is None
        self.clock, self.sleep, self.name = clock, sleep, name
        self.robot: Robot | None = None
        self.run_log = RunLog(None)
        self.context = DEFAULT_CONTEXT
        self.n_alarms = 0
        #: Frames from the start of the episode: what the workspace normally holds.
        self.baseline: dict[str, np.ndarray] | None = None

    @property
    def enabled(self) -> bool:
        return self._reviewer is not None or self.vision_cfg is not None

    def bind(
        self,
        run_log: RunLog | None = None,
        robot: Robot | None = None,
        context: str | None = None,
    ) -> None:
        if run_log is not None:
            self.run_log = run_log
        if robot is not None:
            self.robot = robot
        if context:
            self.context = context

    def reset(self) -> None:
        self.n_alarms = 0
        self.baseline = None
        if self.vision_cfg is not None and self._own_reviewer:  # log to this run
            self._reviewer = claude_reviewer(run_log=self.run_log, **self.vision_cfg)

    def note_baseline(self, obs: Observation, cameras: list[str]) -> None:
        """Keep the first frames of the episode as the reference scene (once)."""
        if self.baseline is None:
            images = {c: obs.images[c].copy() for c in cameras if c in obs.images}
            if images:
                self.baseline = images

    def decide(
        self,
        alarm_images: dict[str, np.ndarray],
        now: Observation,
        cameras: list[str],
        describe: Callable[[float], str],
        log: Callable[[dict[str, Any]], None] = lambda line: None,
    ) -> str | None:
        """Hold and ask Claude until it says resume (-> None) or the arm must stay
        stopped (-> the reason). ``describe(seconds_held)`` says what the detector
        saw; ``alarm_images`` are frames from the moment of the alarm."""
        if self._reviewer is None:
            return "no reviewer configured"
        if self.n_alarms >= self.max_alarms:
            return f"{self.n_alarms} alarms already this episode (max_alarms)"
        self.n_alarms += 1
        paused_at = self.clock()
        latest = now
        while True:
            images = {
                f"{c} (at the start of the run, for reference)": img
                for c, img in (self.baseline or {}).items()
            }
            images.update(alarm_images)
            images.update({f"{c} (now)": latest.images[c] for c in cameras if c in latest.images})
            context = f"{self.context}\n\n{describe(self.clock() - paused_at)}"
            if self.baseline:
                context += (
                    "\n\nThe images marked 'at the start of the run' show the workspace before "
                    "anything happened. Everything visible there (bottles, cups, tools, cables) "
                    "belongs to the scene and is not a new hazard, even if it is near the arm. "
                    "Judge only what is different now: resume when whatever caused the alarm "
                    "is gone and the scene is back to how it started."
                )
            try:
                review = self._reviewer(images, context)
            except Exception as exc:  # no second opinion -> stay stopped
                return f"Claude could not review ({exc})"
            fields = {
                "decision": review.decision,
                "hazard": review.hazard,
                "confidence": review.confidence,
                "reason": review.reason,
                "model": review.model,
            }
            self.run_log.event(f"{self.name}_review", **fields)
            log({"event": "review", **fields})
            if (
                review.decision == "resume"
                and not review.hazard
                and review.confidence >= self.resume_confidence
            ):
                print(f"[safety] RESUME ({review.model}): {review.reason}", flush=True)
                return None
            if review.decision == "abort":
                return f"Claude: {review.reason}"
            # "wait", or a "resume" Claude is not sure of: hold and look again
            if self.robot is None:
                return f"cannot look again without a robot; {review.reason}"
            if self.clock() - paused_at + self.wait_s > self.max_hold_s:
                return f"still unclear after {self.max_hold_s:.0f}s; {review.reason}"
            print(
                f"[safety] HOLD ({review.decision}, {review.confidence:.2f}): {review.reason}",
                flush=True,
            )
            self.sleep(self.wait_s)
            latest = self.robot.get_observation()
