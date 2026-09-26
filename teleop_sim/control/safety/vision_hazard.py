"""A vision safety loop: stop the arm when laya-vision sees a hazard around it.

Wraps another monitor (normally ``sim_watchdog``) and adds, in a background
thread, typed yes/no questions to laya-vision about the latest wrist and
overhead frames -- "is a person's hand in view?" -- at a few Hz per camera
(~150 ms per answer on an M5). ``check`` never blocks the control tick while
all is clear: it hands the thread the newest images and picks up its answers.

One confident "yes" never stops anything. Three in a row (~1 s) raise an alarm
and pause the arm: ``check`` holds the tick, so nothing is commanded and gem13
holds its pose. Claude (Sonnet 5; an unsure "resume" goes to Opus 5.5) is shown
the alarm frame and fresh frames and decides: resume (a false alarm, or the
hazard has gone -- the motion carries on), wait (hold, look again in wait_s),
or abort (safe stop, episode over). No answer from Claude, still unclear after
max_hold_s, or more than max_alarms alarms in an episode aborts. Without a
``review`` section an alarm aborts at once.

Off by default. ``--hazard-monitor`` on scripts/run_pick.py and
scripts/collect_real_picks.py wraps the run's safety monitor with the settings
in configs/safety/vision_hazard.yaml (see ``with_hazard_monitor``); by hand:

    safety:
      type: vision_hazard
      inner: {type: sim_watchdog, max_steps: 30000, max_seconds: 900}
      endpoint: http://127.0.0.1:8765     # scripts/laya_server.py
      review: {type: claude, wait_s: 3.0, max_hold_s: 60.0}
      # checks: [...]                     # default: DEFAULT_CHECKS

This is a supplementary layer, not a certified one: the e-stop and
robots_realtime's own limits stay the real guarantee. It is built to fail
closed -- a server that stops answering for ``max_blind_s`` while frames are
waiting trips as ``hazard_monitor_blind``, and an unreachable server at the
start of an episode is an error, not a silent no-op.

What the questions can and cannot do, measured zero-shot 2026-09-25 on the
1,885-tick, 10-episode clean batch (runs/collect_20260925-133615; see
scripts/eval_hazard_monitor.py):

* Person / hand questions stay quiet on clean frames (1-2 % of frames above
  0.5, none above 0.8), so they ship enabled by default.
* "Is there an object other than the cup in the gripper's path?" does not:
  87-99 % of clean frames score above 0.5, because zero-shot the model cannot
  tell the target cup, the spare cup or the gripper itself from an obstacle.
  Foreign-object checks need fine-tuning on this rig's frames with labelled
  obstacles first (scripts/watch_hazards.py --label collects them); add them
  to ``checks`` with the fine-tuned server once they measure clean.

Detection rate is not yet measured: no frame in the batch has a hazard in it.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from teleop_sim.control.safety.review import ClaudeReview, Reviewer
from teleop_sim.core.protocols import Robot, SafetyMonitor, SafetyVerdict
from teleop_sim.core.registry import SAFETY, build, register
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Observation, Outcome
from teleop_sim.runlog import RunLog

HAND_QUESTION = "Is a person's hand or arm visible in the image?"
REACH_QUESTION = "Is a person reaching towards the robot arm?"


@dataclass(frozen=True)
class HazardCheck:
    """One yes/no question about one camera, and when a "yes" trips.

    ``consecutive`` answers in a row at or above ``trip_above`` raise an
    alarm: one confident frame is never enough; three, ~1 s of frames, are.
    """

    name: str
    camera: str
    question: str
    trip_above: float = 0.6
    consecutive: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.trip_above <= 1.0:
            raise ValueError(f"{self.name}: trip_above must be in (0, 1], got {self.trip_above}")
        if self.consecutive < 1:
            raise ValueError(f"{self.name}: consecutive must be >= 1, got {self.consecutive}")


DEFAULT_CHECKS = (
    HazardCheck("hand_wrist", "wrist", HAND_QUESTION, trip_above=0.7),
    HazardCheck("hand_overview", "overview", HAND_QUESTION),
    HazardCheck("reach_overview", "overview", REACH_QUESTION),
)


@dataclass
class HazardTrip:
    check: str
    camera: str
    p: float
    history: list[float]
    image: np.ndarray | None = field(default=None, repr=False)


class HazardTracker:
    """Debounces answers into trips. Pure: no threads, no I/O -- test and replay it."""

    def __init__(self, checks: tuple[HazardCheck, ...] | list[HazardCheck]) -> None:
        names = [c.name for c in checks]
        if len(set(names)) != len(names):
            raise ValueError(f"hazard check names must be unique, got {names}")
        self.checks = tuple(checks)
        self._history: dict[str, list[float]] = {c.name: [] for c in self.checks}

    def reset(self) -> None:
        for h in self._history.values():
            h.clear()

    def cameras(self) -> list[str]:
        return sorted({c.camera for c in self.checks})

    def questions_for(self, camera: str) -> dict[str, dict[str, str]]:
        return {
            c.name: {"type": "noul", "instructions": c.question}
            for c in self.checks
            if c.camera == camera
        }

    def update(self, camera: str, answers: dict[str, float]) -> HazardTrip | None:
        """Feed one camera's P(yes) per check; return the first check that trips."""
        for c in self.checks:
            if c.camera != camera or c.name not in answers:
                continue
            h = self._history[c.name]
            h.append(float(answers[c.name]))
            del h[: -max(c.consecutive, 8)]
            recent = h[-c.consecutive :]
            if len(recent) == c.consecutive and min(recent) >= c.trip_above:
                return HazardTrip(c.name, camera, recent[-1], list(recent))
        return None


#: (image, questions) -> {check name: P(yes)}; raises on failure.
Asker = Callable[[np.ndarray, dict[str, dict[str, str]]], dict[str, float]]
#: The opt-in settings --hazard-monitor applies (run_pick.py, collect_real_picks.py).
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "safety" / "vision_hazard.yaml"


def with_hazard_monitor(
    safety: dict[str, Any] | None, config: str | Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    """A safety config that wraps ``safety`` (the run's own monitor) in vision_hazard."""
    import yaml

    wrapped = dict(yaml.safe_load(Path(config).read_text()))
    wrapped["inner"] = dict(safety or {"type": "sim_watchdog"})
    return wrapped


def laya_asker(endpoint: str, timeout_s: float) -> Asker:
    from teleop_sim.perception.claude_vision import encode_image
    from teleop_sim.perception.laya_vision import post_predict

    def ask(image: np.ndarray, questions: dict[str, dict[str, str]]) -> dict[str, float]:
        data, _ = encode_image(image)
        answer = post_predict(endpoint, data, questions, timeout_s)
        return {k: float(v["noul"]) for k, v in answer["answers"].items()}

    return ask


@register(SAFETY, "vision_hazard")
class VisionHazardMonitor(SafetyMonitor):
    def __init__(
        self,
        spec: RobotSpec | None = None,
        inner: dict[str, Any] | None = None,
        endpoint: str = "http://127.0.0.1:8765",
        checks: list[dict[str, Any]] | None = None,
        timeout_s: float = 1.0,
        max_blind_s: float = 2.0,
        log_dir: str | None = None,
        save_images_above: float = 0.4,
        review: dict[str, Any] | None = None,
        asker: Asker | None = None,
        reviewer: Reviewer | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner: SafetyMonitor | None = (
            build(SAFETY, dict(inner), spec=spec) if inner is not None else None
        )
        self.tracker = HazardTracker(
            tuple(HazardCheck(**c) for c in checks) if checks is not None else DEFAULT_CHECKS
        )
        self.endpoint = endpoint
        self.ask = asker or laya_asker(endpoint, timeout_s)
        self.max_blind_s = float(max_blind_s)
        self.save_images_above = float(save_images_above)
        self.clock, self.sleep = clock, sleep
        self.log_dir = Path(log_dir) if log_dir else None
        # review: None -> an alarm ends the episode at once (no second opinion).
        self.review = ClaudeReview(review, reviewer, clock=clock, sleep=sleep, name="hazard")
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._slot: dict[str, np.ndarray] | None = None
        #: When the oldest frame not yet followed by a good answer was handed over.
        self._unanswered_since: float | None = None
        self._trip: HazardTrip | None = None
        self._error: str | None = None
        self.last_answers: dict[str, dict[str, float]] = {}
        self.n_answers = 0
        self._n_images = 0

    # ------------------------------------------------------------ binding

    def bind(
        self,
        run_log: RunLog | None = None,
        robot: Robot | None = None,
        context: str | None = None,
        **kw: Any,
    ) -> None:
        """The run's log (hazard_checks.jsonl, frames, Claude's reviews go there), the
        robot to read fresh frames from while paused, and what the task expects to see."""
        if run_log is not None:
            self.log_dir = run_log.root
        self.review.bind(run_log=run_log, robot=robot, context=context)
        if self.inner is not None and hasattr(self.inner, "bind"):
            self.inner.bind(run_log=run_log, robot=robot, context=context, **kw)

    # ------------------------------------------------------------ SafetyMonitor

    def reset(self) -> None:
        if self.inner is not None:
            self.inner.reset()
        self._halt_thread()
        self.tracker.reset()
        self._slot, self._unanswered_since, self._trip, self._error = None, None, None, None
        self.last_answers, self.n_answers = {}, 0
        self.review.reset()
        self._preflight()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision-hazard", daemon=True)
        self._thread.start()

    def check(self, obs: Observation, step: int) -> SafetyVerdict | None:
        if self.inner is not None:
            verdict = self.inner.check(obs, step)
            if verdict is not None:
                return verdict
        now = self.clock()
        self.review.note_baseline(obs, self.tracker.cameras())
        images = {c: obs.images[c] for c in self.tracker.cameras() if c in obs.images}
        with self._lock:
            trip, error = self._trip, self._error
            if images:
                self._slot = images
                if self._unanswered_since is None:
                    self._unanswered_since = now
                self._wake.set()
            waiting_since = self._unanswered_since
        if trip is not None:
            alarm = f"laya-vision: {trip.check} on {trip.camera} P(yes)=" + "/".join(
                f"{p:.2f}" for p in trip.history
            )
            if trip.image is not None:
                self._save_image(f"alarm_{trip.check}", trip.image)
            if not self.review.enabled:
                return SafetyVerdict(Outcome.WATCHDOG, f"hazard:{trip.check}", alarm, True)
            return self._pause_and_review(obs, trip, alarm)
        if waiting_since is not None and now - waiting_since > self.max_blind_s:
            return SafetyVerdict(
                Outcome.WATCHDOG,
                "hazard_monitor_blind",
                f"no laya-vision answer for {now - waiting_since:.1f}s"
                + (f" (last error: {error})" if error else ""),
                safe_stop=True,
            )
        return None

    def _pause_and_review(
        self, obs: Observation, trip: HazardTrip, alarm: str
    ) -> SafetyVerdict | None:
        """Hold the arm and let Claude decide: resume, keep waiting, or abort.

        Blocking here is the pause: while check() does not return, the loop sends
        no command, and gem13's RobotNode holds its last pose (in sim, nothing
        steps). The policy hands out waypoints one per tick, so on resume the
        motion carries on from where it stopped.
        """
        print(f"[safety] PAUSE: {alarm}; asking Claude", flush=True)
        self.review.run_log.event(
            "hazard_pause",
            check=trip.check,
            camera=trip.camera,
            p=[round(p, 3) for p in trip.history],
        )
        self._log({"event": "pause", "check": trip.check, "detail": alarm})
        alarm_images = (
            {f"{trip.camera} (frame that raised the alarm)": trip.image}
            if trip.image is not None
            else {}
        )

        def describe(held_s: float) -> str:
            return (
                f"The alarm came from the camera detector: {trip.check} -- asked "
                f"{self._question(trip.check)!r} of the {trip.camera} camera, it answered "
                "yes with probability "
                + "/".join(f"{p:.2f}" for p in trip.history)
                + f" on {len(trip.history)} frames in a row. The arm has been holding for "
                f"{held_s:.0f}s."
            )

        why = self.review.decide(alarm_images, obs, self.tracker.cameras(), describe, log=self._log)
        if why is not None:
            return self._abort(trip, f"{why}; {alarm}")
        with self._lock:
            self.tracker.reset()
            self._trip, self._unanswered_since = None, None
        self._log({"event": "resume"})
        return None

    def _abort(self, trip: HazardTrip, detail: str) -> SafetyVerdict:
        return SafetyVerdict(Outcome.WATCHDOG, f"hazard:{trip.check}", detail, safe_stop=True)

    def _question(self, check: str) -> str:
        return next(c.question for c in self.tracker.checks if c.name == check)

    def on_trip(self, robot: Robot, verdict: SafetyVerdict) -> None:
        if verdict.safe_stop:
            robot.safe_stop()
        print(f"[safety] TRIP {verdict.tag}: {verdict.detail}", flush=True)
        self._log({"event": "trip", "tag": verdict.tag, "detail": verdict.detail})
        self._halt_thread()

    def close(self) -> None:
        self._halt_thread()

    # ------------------------------------------------------------ worker

    def _preflight(self) -> None:
        """Fail at episode start, not mid-motion, if the model cannot be asked."""
        probe = np.zeros((64, 64, 3), dtype=np.uint8)
        camera = self.tracker.cameras()[0]
        try:
            self.ask(probe, self.tracker.questions_for(camera))
        except Exception as exc:
            raise RuntimeError(
                f"vision_hazard: laya-vision did not answer at {self.endpoint} ({exc}). Start "
                "it: third_party/laya-vision/.venv/bin/python scripts/laya_server.py"
            ) from exc

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(0.1):
                continue
            with self._lock:
                self._wake.clear()
                if self._slot is None:
                    continue
                images, self._slot = self._slot, None
            for camera, image in images.items():
                if self._stop.is_set():
                    return
                try:
                    answers = self.ask(image, self.tracker.questions_for(camera))
                except Exception as exc:  # stay blind; check() trips after max_blind_s
                    with self._lock:
                        self._error = f"{type(exc).__name__}: {exc}"
                    self._log({"event": "error", "camera": camera, "error": self._error})
                    continue
                self._record(camera, image, answers)

    def _record(self, camera: str, image: np.ndarray, answers: dict[str, float]) -> None:
        trip = self.tracker.update(camera, answers)
        with self._lock:
            self.last_answers[camera] = answers
            self.n_answers += 1
            self._error = None
            self._unanswered_since = None
            if trip is not None and self._trip is None:
                trip.image = image
                self._trip = trip
        line: dict[str, Any] = {
            "event": "answer",
            "camera": camera,
            "p": {k: round(v, 3) for k, v in answers.items()},
        }
        if max(answers.values(), default=0.0) >= self.save_images_above:
            line["image"] = self._save_image(f"hazard_{camera}", image)
        self._log(line)

    def _halt_thread(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None

    # ------------------------------------------------------------ logging

    def _log(self, line: dict[str, Any]) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / "hazard_checks.jsonl", "a") as fh:
            fh.write(json.dumps({"t": time.time(), **line}) + "\n")

    def _save_image(self, name: str, rgb: np.ndarray) -> str | None:
        if self.log_dir is None:
            return None
        import cv2

        self._n_images += 1
        fname = f"{self._n_images:04d}_{name}.jpg"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.log_dir / fname), np.ascontiguousarray(rgb[:, :, ::-1]))
        return fname
