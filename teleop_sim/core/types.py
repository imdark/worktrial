"""Canonical types exchanged at every seam.

Three conventions are load-bearing and are enforced here rather than by
documentation:

* joint values are RADIANS,
* gripper values are normalised to [0, 1] (0 = open, 1 = closed),
* timestamps are monotonic seconds (``time.perf_counter`` domain).

Native units -- servo ticks, degrees, metres of finger travel -- exist only
inside a driver and never cross a seam. Conformance (tests/conformance/) is
what enforces this on every Robot implementation; this module only enforces it
on the types themselves.

Morphology assumption, stated once and deliberately: these types encode one
kinematic chain with one scalar-DoF end effector. Bimanual arms and dexterous
hands require revising Observation and Action. The robot_spec_hash carried by
every episode is what makes that migration detectable rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

_EPS = 1e-6


class TypeValidationError(ValueError):
    """A canonical type was constructed with out-of-contract data."""


class ControlMode(str, Enum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    EE_POSE_ABS = "ee_pose_abs"
    EE_POSE_DELTA = "ee_pose_delta"


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    WATCHDOG = "watchdog"
    ABORTED = "aborted"


class ActionOrigin(str, Enum):
    """Who produced an action.

    Finer-grained than a human/policy boolean because Stage 7 needs base
    demonstrations separable from corrections (they are weighted differently at
    training time), and Stage 12 needs to know which level of the hierarchy
    emitted a given action.
    """

    HUMAN_TELEOP = "human_teleop"
    HUMAN_CORRECTION = "human_correction"
    SCRIPTED = "scripted"
    POLICY = "policy"
    POLICY_HIGH_LEVEL = "policy_high_level"
    POLICY_LOW_LEVEL = "policy_low_level"

    @property
    def is_human(self) -> bool:
        return self in (ActionOrigin.HUMAN_TELEOP, ActionOrigin.HUMAN_CORRECTION)


class Button:
    """Semantic names for teleoperator buttons."""

    RECORD = "record"
    RESET = "reset"
    DISCARD = "discard"
    ESTOP = "estop"
    TAKEOVER = "takeover"

    ALL = (RECORD, RESET, DISCARD, ESTOP, TAKEOVER)


class TaskEvent:
    """Events an action source raises to end an episode on the task's own verdict.

    A scripted or VLM-guided task knows when it has finished -- or has failed
    and why -- before any SuccessDetector could. ``failure(tag)`` carries the
    reason into ``EpisodeResult.failure_tag``.
    """

    SUCCESS = "task_success"
    FAILURE = "task_failure"

    @staticmethod
    def failure(tag: str) -> str:
        return f"{TaskEvent.FAILURE}:{tag}" if tag else TaskEvent.FAILURE


TeleopKind = Literal["joint", "ee_pose", "ee_delta"]


def _as_vector(values: Any, name: str, size: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise TypeValidationError(f"{name} must be 1-D, got shape {arr.shape}")
    if size is not None and arr.shape[0] != size:
        raise TypeValidationError(f"{name} must have {size} entries, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise TypeValidationError(f"{name} contains non-finite values: {arr}")
    return arr


def _check_normalised(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise TypeValidationError(f"{name} must be finite, got {value}")
    if value < -_EPS or value > 1.0 + _EPS:
        raise TypeValidationError(
            f"{name} must be normalised to [0, 1] (0 = open, 1 = closed), got {value}. "
            "Native gripper units belong inside the driver."
        )
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class Observation:
    """Everything a policy or a human sees at one instant.

    The sensing fields are explicitly nullable because most position-controlled
    hobby servos cannot report force or current. A robot must populate exactly
    the fields its RobotSpec declares -- conformance checks both directions.
    """

    images: dict[str, np.ndarray]
    joint_pos: np.ndarray
    gripper: float
    timestamp: float
    joint_vel: np.ndarray | None = None
    joint_current: np.ndarray | None = None
    joint_torque: np.ndarray | None = None
    ee_wrench: np.ndarray | None = None
    ee_pose: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    #: Observation fields a PolicySpec may name in ``obs_keys``.
    OBS_KEYS = (
        "images",
        "joint_pos",
        "joint_vel",
        "joint_current",
        "joint_torque",
        "ee_wrench",
        "ee_pose",
        "gripper",
    )

    def __post_init__(self) -> None:
        self.joint_pos = _as_vector(self.joint_pos, "joint_pos")
        dof = self.joint_pos.shape[0]
        for name in ("joint_vel", "joint_current", "joint_torque"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _as_vector(value, name, size=dof))
        if self.ee_wrench is not None:
            self.ee_wrench = _as_vector(self.ee_wrench, "ee_wrench", size=6)
        if self.ee_pose is not None:
            self.ee_pose = _as_vector(self.ee_pose, "ee_pose", size=7)

        self.gripper = _check_normalised(self.gripper, "gripper")
        self.timestamp = float(self.timestamp)

        for name, img in self.images.items():
            if not isinstance(img, np.ndarray):
                raise TypeValidationError(f"image {name!r} must be a numpy array")
            if img.ndim != 3 or img.shape[2] != 3:
                raise TypeValidationError(f"image {name!r} must be HxWx3, got {img.shape}")
            if img.dtype != np.uint8:
                raise TypeValidationError(f"image {name!r} must be uint8, got {img.dtype}")

    @property
    def dof(self) -> int:
        return int(self.joint_pos.shape[0])

    def has(self, key: str) -> bool:
        """Whether this observation actually carries ``key``."""
        if key == "images":
            return bool(self.images)
        if key not in self.OBS_KEYS:
            raise TypeValidationError(
                f"unknown observation key {key!r}; known keys: {list(self.OBS_KEYS)}"
            )
        return getattr(self, key) is not None

    def to_record(self) -> dict[str, Any]:
        """Flat, JSON-safe view for the dataset writer.

        Images are excluded: the video backend owns them, referenced by frame
        index rather than inlined.
        """
        record: dict[str, Any] = {
            "joint_pos": self.joint_pos.tolist(),
            "gripper": self.gripper,
            "timestamp": self.timestamp,
            "camera_names": sorted(self.images),
        }
        for name in ("joint_vel", "joint_current", "joint_torque", "ee_wrench", "ee_pose"):
            value = getattr(self, name)
            record[name] = None if value is None else value.tolist()
        return record

    @classmethod
    def from_record(
        cls, record: dict[str, Any], images: dict[str, np.ndarray] | None = None
    ) -> Observation:
        return cls(
            images=images or {},
            joint_pos=record["joint_pos"],
            joint_vel=record.get("joint_vel"),
            joint_current=record.get("joint_current"),
            joint_torque=record.get("joint_torque"),
            ee_wrench=record.get("ee_wrench"),
            ee_pose=record.get("ee_pose"),
            gripper=record["gripper"],
            timestamp=record["timestamp"],
        )


@dataclass
class Action:
    """A command destined for the robot.

    ``obs_timestamp`` is what makes staleness measurable: with asynchronous
    inference the action applied at time T may have been computed from an
    observation several ticks old, and that number is the single most useful
    diagnostic for a policy that evaluates well and fails live.
    """

    mode: ControlMode
    values: np.ndarray
    gripper: float
    timestamp: float
    obs_timestamp: float | None = None
    source: ActionOrigin = ActionOrigin.POLICY

    def __post_init__(self) -> None:
        self.mode = ControlMode(self.mode)
        self.values = _as_vector(self.values, "values")
        self.gripper = _check_normalised(self.gripper, "gripper")
        self.timestamp = float(self.timestamp)
        if self.obs_timestamp is not None:
            self.obs_timestamp = float(self.obs_timestamp)
        self.source = ActionOrigin(self.source)

    @property
    def dof(self) -> int:
        return int(self.values.shape[0])

    def staleness_ms(self, now: float) -> float | None:
        """Milliseconds between the observation used and ``now``."""
        if self.obs_timestamp is None:
            return None
        return (float(now) - self.obs_timestamp) * 1000.0

    def replace_values(self, values: np.ndarray) -> Action:
        """Copy with new values, preserving mode, provenance and timestamps.

        Drivers use this to report what was actually sent after clipping.
        """
        return Action(
            mode=self.mode,
            values=values,
            gripper=self.gripper,
            timestamp=self.timestamp,
            obs_timestamp=self.obs_timestamp,
            source=self.source,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "values": self.values.tolist(),
            "gripper": self.gripper,
            "timestamp": self.timestamp,
            "obs_timestamp": self.obs_timestamp,
            "source": self.source.value,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Action:
        return cls(
            mode=ControlMode(record["mode"]),
            values=record["values"],
            gripper=record["gripper"],
            timestamp=record["timestamp"],
            obs_timestamp=record.get("obs_timestamp"),
            source=ActionOrigin(record.get("source", ActionOrigin.POLICY)),
        )


@dataclass
class TeleopCommand:
    """Raw output of a human input device, before retargeting.

    Deliberately knows nothing about the robot it will drive -- that mapping is
    the Retargeter's job, and it is what makes a VR headset a drop-in
    replacement for a leader arm.
    """

    kind: TeleopKind
    values: np.ndarray
    gripper: float
    timestamp: float
    buttons: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("joint", "ee_pose", "ee_delta"):
            raise TypeValidationError(f"unknown teleop command kind {self.kind!r}")
        self.values = _as_vector(self.values, "values")
        self.gripper = _check_normalised(self.gripper, "gripper")
        self.timestamp = float(self.timestamp)

    def pressed(self, button: str) -> bool:
        return bool(self.buttons.get(button, False))

    def pressed_events(self) -> set[str]:
        return {name for name, down in self.buttons.items() if down}


@dataclass
class ActionChunk:
    """A policy's output: several future actions plus the observation they came from.

    Chunking lives here rather than inside Policy so that buffering and
    concurrency have exactly one implementation each (PolicySource and
    AsyncPolicySource) instead of one per policy.
    """

    actions: list[Action]
    obs_timestamp: float
    horizon_hz: float

    def __post_init__(self) -> None:
        if not self.actions:
            raise TypeValidationError("an ActionChunk must contain at least one action")
        self.obs_timestamp = float(self.obs_timestamp)
        self.horizon_hz = float(self.horizon_hz)
        if self.horizon_hz <= 0:
            raise TypeValidationError(f"horizon_hz must be positive, got {self.horizon_hz}")
        for action in self.actions:
            if action.obs_timestamp is None:
                action.obs_timestamp = self.obs_timestamp

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class EpisodeResult:
    """How an episode ended, and under what provenance.

    Every autonomous episode terminates on its own -- success, failure,
    timeout, or safety trip -- never "until someone notices".
    """

    outcome: Outcome
    steps: int
    duration: float
    seed: int | None = None
    failure_tag: str | None = None
    intervention_frac: float = 0.0
    robot_spec_hash: str | None = None
    policy_spec_hash: str | None = None
    code_version: str | None = None
    staleness_ms: dict[str, float] = field(default_factory=dict)
    underruns: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.outcome = Outcome(self.outcome)
        self.steps = int(self.steps)
        self.duration = float(self.duration)
        self.underruns = int(self.underruns)
        self.intervention_frac = _check_normalised(
            self.intervention_frac, "intervention_frac"
        )

    @property
    def succeeded(self) -> bool:
        return self.outcome is Outcome.SUCCESS

    @property
    def autonomous(self) -> bool:
        return self.intervention_frac == 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "steps": self.steps,
            "duration": self.duration,
            "seed": self.seed,
            "failure_tag": self.failure_tag,
            "intervention_frac": self.intervention_frac,
            "robot_spec_hash": self.robot_spec_hash,
            "policy_spec_hash": self.policy_spec_hash,
            "code_version": self.code_version,
            "staleness_ms": dict(self.staleness_ms),
            "underruns": self.underruns,
            **self.extra,
        }
