"""The one control loop.

Teleoperated collection, scripted rollouts, autonomous evaluation and (later)
shared control are all this function with different objects passed in. It
depends on the seams and on nothing concrete at all.

The loop owns the clock (principle 7). It measures how long each step took and
how stale the action it applied was, and it records both. A source that cannot
answer inside a tick returns its freshest answer; staleness is what makes that
visible instead of mysterious.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teleop_sim.control.compat import CompatibilityReport, check_compatibility
from teleop_sim.control.latency import LatencyTracker
from teleop_sim.core.clock import Clock, RateLimiter, WallClock
from teleop_sim.core.hashing import code_version
from teleop_sim.core.protocols import (
    ActionSource,
    Recorder,
    ResetStrategy,
    Robot,
    SafetyMonitor,
    SuccessDetector,
)
from teleop_sim.core.types import Button, EpisodeResult, Outcome, TaskEvent

# Events that end an episode immediately, mapped to (outcome, tag, discard).
_TERMINAL_EVENTS: dict[str, tuple[Outcome, str, bool]] = {
    Button.ESTOP: (Outcome.ABORTED, "estop", False),
    Button.DISCARD: (Outcome.ABORTED, "discard", True),
    Button.RESET: (Outcome.ABORTED, "reset", False),
}
# A source's own verdict on the task: TaskEvent.SUCCESS, or TaskEvent.failure(tag).
_TASK_EVENT_PREFIX = "task_"


@dataclass
class LoopConfig:
    control_hz: float
    # A backstop only. The SafetyMonitor is what is meant to end episodes; if
    # this bound is ever what stops a run, the monitor is misconfigured.
    max_steps: int = 100_000


class ControlLoop:
    def __init__(
        self,
        robot: Robot,
        source: ActionSource,
        success: SuccessDetector,
        reset: ResetStrategy,
        safety: SafetyMonitor,
        clock: Clock | None = None,
        recorder: Recorder | None = None,
        config: LoopConfig | None = None,
        safety_config: dict[str, object] | None = None,
        check_compat: bool = True,
    ) -> None:
        self.robot = robot
        self.source = source
        self.success = success
        self.reset_strategy = reset
        self.safety = safety
        self.clock = clock or WallClock()
        self.recorder = recorder
        self.config = config or LoopConfig(control_hz=robot.spec.control_hz)
        self.rate = RateLimiter(self.config.control_hz, self.clock)
        self.latency = LatencyTracker()

        self._robot_spec_hash = robot.spec.content_hash()
        self._code_version = code_version()

        # Compatibility fails at construction, not at step 1 (principle 11).
        self.compatibility = CompatibilityReport()
        if check_compat:
            self.compatibility = check_compatibility(
                source.policy_spec(),
                robot.spec,
                camera_config=robot.camera_config(),
                safety_config=safety_config,
            )

    def run_episode(self, seed: int | None = None) -> EpisodeResult:
        rng = np.random.default_rng(seed)
        self.reset_strategy.reset(rng)
        obs = self.robot.reset(seed)
        self.source.reset(obs)
        self.success.reset()
        self.safety.reset()
        self.rate.reset()
        self.latency.reset()
        if self.recorder is not None:
            self.recorder.start_episode(seed)

        started = self.clock.now()
        steps = 0
        human_steps = 0
        discard = False
        outcome: Outcome | None = None
        tag: str | None = None

        while True:
            if steps > 0:
                obs = self.robot.get_observation()

            if self.success.is_success(obs):
                outcome = Outcome.SUCCESS
                break

            verdict = self.safety.check(obs, steps)
            if verdict is not None:
                outcome, tag = verdict.outcome, verdict.tag
                self.safety.on_trip(self.robot, verdict)
                break

            if steps >= self.config.max_steps:
                outcome, tag = Outcome.TIMEOUT, "loop_max_steps"
                break

            action = self.source.get_action(obs)

            events = self.source.poll_events()
            terminal = next((e for e in events if e in _TERMINAL_EVENTS), None)
            if terminal is not None:
                outcome, tag, discard = _TERMINAL_EVENTS[terminal]
                break
            verdict_event = next((e for e in events if e.startswith(_TASK_EVENT_PREFIX)), None)
            if verdict_event is not None:
                if verdict_event == TaskEvent.SUCCESS:
                    outcome = Outcome.SUCCESS
                else:
                    outcome = Outcome.FAILURE
                    tag = verdict_event.partition(":")[2] or TaskEvent.FAILURE
                break

            sent = self.robot.send_action(action)
            staleness = sent.staleness_ms(self.clock.now())
            if sent.source.is_human:
                human_steps += 1
            if self.recorder is not None:
                self.recorder.record(obs, sent)

            steps += 1
            dt = self.rate.sleep()
            self.latency.record(dt, staleness_ms=staleness)

        if outcome is Outcome.SUCCESS:
            tag = None
        elif tag is None:
            tag = self.success.failure_tag(obs)

        policy_spec = self.source.policy_spec()
        result = EpisodeResult(
            outcome=outcome or Outcome.FAILURE,
            steps=steps,
            duration=self.clock.now() - started,
            seed=seed,
            failure_tag=tag,
            intervention_frac=(human_steps / steps) if steps else 0.0,
            robot_spec_hash=self._robot_spec_hash,
            policy_spec_hash=None if policy_spec is None else policy_spec.content_hash(),
            code_version=self._code_version,
            staleness_ms=self.latency.staleness_stats(),
            underruns=self.source.underruns(),
            extra={"source": self.source.name, "rate": self.rate.stats()},
        )

        if self.recorder is not None:
            if discard:
                self.recorder.discard_episode()
            else:
                self.recorder.end_episode(result)
        return result

    def run(self, episodes: int, seed: int = 0) -> list[EpisodeResult]:
        """Run consecutive episodes. The Stage 6 harness wraps this with
        reporting and autonomous scene reset."""
        return [self.run_episode(seed=seed + i) for i in range(episodes)]
