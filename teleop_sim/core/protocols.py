"""The eight seams (§2.1).

An interface exists only where an implementation will genuinely be swapped and
where the second implementation can already be named. Everything else is
written concretely.

``ActionSource`` is not a seam -- it is the adapter that lets one control loop
serve teleoperation, scripted control, autonomy and shared control. Neither is
``Task``, which is a concrete composition of a SuccessDetector, a
ResetStrategy and scene code (see teleop_sim/envs/task.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from teleop_sim.core.policy_spec import PolicySpec
from teleop_sim.core.spec import CameraSpec, RobotSpec
from teleop_sim.core.types import (
    Action,
    ActionChunk,
    ActionOrigin,
    EpisodeResult,
    Observation,
    Outcome,
    TeleopCommand,
)


class UnsupportedControlMode(ValueError):
    """A robot was commanded in a mode its spec does not declare."""


# --------------------------------------------------------------- seam 1: Robot


class Robot(ABC):
    """A physical or simulated arm. Owns unit conversion and joint safety.

    Every implementation must pass tests/conformance/ -- that suite, not
    convention, is what enforces radians-in/radians-out and gripper
    normalisation across drivers.
    """

    spec: RobotSpec

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_observation(self) -> Observation: ...

    @abstractmethod
    def send_action(self, action: Action) -> Action:
        """Command the robot.

        Returns the action *actually sent* after clipping to joint limits. The
        recorder logs this return value, never the requested action -- a
        dataset of unreachable targets teaches a policy to command them.

        Must raise UnsupportedControlMode for a mode absent from
        ``spec.supported_modes``, rather than silently reinterpreting it.
        """

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """Return to a home pose and produce the first observation."""

    def safe_stop(self) -> None:
        """Bring the robot to a standstill. Overridden by hardware drivers."""
        return None

    def camera_config(self) -> dict[str, tuple[int, int]]:
        """Camera names and the resolutions actually served, for the startup check.

        Defaults to what the spec declares. A driver that serves something else
        -- a downscaled stream, a subset of cameras -- must override this, or
        the compatibility check validates a fiction.
        """
        return {camera.name: camera.resolution for camera in self.spec.cameras}

    def __enter__(self) -> Robot:
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disconnect()


# -------------------------------------------------- seams 2-3: human input path


class Teleoperator(ABC):
    """A human input device. Knows nothing about the robot it will drive."""

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_command(self) -> TeleopCommand: ...

    def send_feedback(self, obs: Observation) -> None:
        """Optional haptics / force feedback. No-op by default."""
        return None


class Retargeter(ABC):
    """Maps a device-shaped command onto one robot's action space.

    The seam that makes VR a later config change: a leader arm emits joint
    targets and uses an affine map, a headset emits an end-effector pose and
    uses IK, and nothing downstream can tell the difference.
    """

    @abstractmethod
    def __call__(self, command: TeleopCommand, obs: Observation) -> Action: ...

    def reset(self) -> None:
        return None


# -------------------------------------------------------------- seam 4: Camera


class Camera(ABC):
    spec: CameraSpec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def resolution(self) -> tuple[int, int]:
        return self.spec.resolution

    @abstractmethod
    def read(self) -> np.ndarray:
        """One HxWx3 uint8 frame."""

    @property
    def intrinsics(self) -> np.ndarray:
        return self.spec.intrinsics()


# -------------------------------------------------------------- seam 5: Policy


class Policy(ABC):
    """Observation -> action chunk.

    ``predict`` may be slow and may be called off the control thread. Buffering
    and concurrency belong to the source, not here, so that each has exactly
    one implementation rather than one per policy.
    """

    spec: PolicySpec

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def predict(self, obs: Observation) -> ActionChunk: ...


# ----------------------------------------------- seams 6-7: success and reset


class SuccessDetector(ABC):
    """Did the episode achieve its goal?

    A seam because sim reads privileged simulator state and hardware cannot:
    on a real robot this becomes a learned classifier, an instrumented
    fixture, or a human. Stage 11's exit criterion is unmeasurable without it.
    """

    def reset(self) -> None:
        return None

    @abstractmethod
    def is_success(self, obs: Observation) -> bool: ...

    def failure_tag(self, obs: Observation) -> str | None:
        """Coarse label for why an episode failed, if known."""
        return None


class ResetStrategy(ABC):
    """Return the world to a start state between episodes.

    In sim: teleport and re-randomise. On hardware: a fixture, a scripted
    re-place routine, or a human between batches.
    """

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None: ...


# ------------------------------------------------------- seam 8: SafetyMonitor


@dataclass(frozen=True)
class SafetyVerdict:
    outcome: Outcome
    tag: str
    detail: str = ""
    safe_stop: bool = False


class SafetyMonitor(ABC):
    """Ends episodes that would otherwise run forever or do damage.

    In teleoperation the human is the monitor: they notice a stall, a runaway,
    an arm heading off the table. Autonomy removes that human, so the
    guarantee has to be made structural.
    """

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def check(self, obs: Observation, step: int) -> SafetyVerdict | None: ...

    def on_trip(self, robot: Robot, verdict: SafetyVerdict) -> None:
        """React to a trip. Hardware implementations command a safe stop."""
        if verdict.safe_stop:
            robot.safe_stop()


# ------------------------------------------------------ adapter: ActionSource


class ActionSource(ABC):
    """Whatever is driving the robot this episode -- human, policy, or both.

    ``get_action`` MUST NOT block past a control tick. A source that cannot
    answer in time returns its freshest answer; the loop measures how stale
    that answer was.
    """

    origin: ActionOrigin = ActionOrigin.POLICY

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def reset(self, obs: Observation | None = None) -> None: ...

    @abstractmethod
    def get_action(self, obs: Observation) -> Action: ...

    def poll_events(self) -> set[str]:
        """Events raised during the most recent ``get_action``.

        Button names from ``Button``. A policy source raises none, so the loop
        needs no branch on whether a human is present.
        """
        return set()

    def underruns(self) -> int:
        """Ticks served without a fresh action. Always 0 for synchronous sources."""
        return 0

    def policy_spec(self) -> PolicySpec | None:
        """The spec of the policy behind this source, if any (for provenance)."""
        return None

    def close(self) -> None:
        return None


class Recorder(ABC):
    """Dataset writer. Implemented at Stage 3; the loop tolerates ``None``."""

    @abstractmethod
    def start_episode(self, seed: int | None = None) -> None: ...

    @abstractmethod
    def record(self, obs: Observation, action: Action) -> None: ...

    @abstractmethod
    def end_episode(self, result: EpisodeResult) -> None: ...

    @abstractmethod
    def discard_episode(self) -> None: ...
