"""Hardware described as data.

A ``RobotSpec`` drives the simulated robot, the real driver, the safety layer
and the renderer from one file. Swapping arms is a new YAML plus a driver; it
is never a change to the loop, the policy or the recorder.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from teleop_sim.core.hashing import content_hash
from teleop_sim.core.types import ControlMode

_INCLUDE = re.compile(r"""<include\s+file\s*=\s*["']([^"']+)["']""")


def _file_tree_digest(path: str | Path | None) -> str | None:
    """Hash a model file together with everything it includes.

    MuJoCo's <include> is textual, so scene.xml's own bytes say nothing about
    the arm it pulls in. Without following includes, editing a joint limit in
    so101.xml would leave the spec hash unchanged.
    """
    if not path:
        return None
    root = Path(path)
    if not root.is_file():
        return None

    seen: set[Path] = set()
    parts: list[str] = []

    def walk(current: Path) -> None:
        current = current.resolve()
        if current in seen or not current.is_file():
            return
        seen.add(current)
        text = current.read_text()
        parts.append(f"{current.name}:{content_hash(text)}")
        for included in sorted(_INCLUDE.findall(text)):
            walk(current.parent / included)

    walk(root)
    return content_hash(sorted(parts))


class SpecError(ValueError):
    """A robot spec is missing, malformed or self-inconsistent."""


@dataclass
class JointSpec:
    name: str
    lower: float
    upper: float
    vel_limit: float = math.inf
    effort_limit: float = math.inf

    def __post_init__(self) -> None:
        self.lower = float(self.lower)
        self.upper = float(self.upper)
        if self.lower >= self.upper:
            raise SpecError(
                f"joint {self.name!r}: lower ({self.lower}) must be < upper ({self.upper}). "
                "Limits are in RADIANS -- degrees are a common cause of this."
            )
        if self.vel_limit <= 0 or self.effort_limit <= 0:
            raise SpecError(f"joint {self.name!r}: vel/effort limits must be positive")


@dataclass
class GripperSpec:
    """Maps native gripper units onto the canonical [0, 1] range.

    This is the only place the conversion lives, which is why a Feetech tick
    range and a Franka finger width in metres look identical to a policy.
    """

    joint: str
    open_pos: float
    closed_pos: float
    #: Actuator that drives the jaw, when it is not named after the joint. A
    #: coupled two-finger gripper typically has one actuator on one finger.
    actuator: str | None = None
    #: Fraction of travel, at each end, within which the reading saturates to
    #: exactly 0.0 or 1.0. A jaw only lands precisely on an extreme at a hard
    #: stop; when "closed" means the pads pressing on each other, friction and
    #: contact leave it a fraction of a millimetre short. Real calibration does
    #: the same thing. Zero means no saturation band.
    deadband: float = 0.0

    def __post_init__(self) -> None:
        self.open_pos = float(self.open_pos)
        self.closed_pos = float(self.closed_pos)
        self.deadband = float(self.deadband)
        if math.isclose(self.open_pos, self.closed_pos):
            raise SpecError(
                f"gripper {self.joint!r}: open_pos and closed_pos must differ"
            )
        if not 0.0 <= self.deadband < 0.5:
            raise SpecError(
                f"gripper {self.joint!r}: deadband must be in [0, 0.5), got {self.deadband}"
            )

    @property
    def actuator_name(self) -> str:
        return self.actuator or self.joint

    def normalize(self, raw: float) -> float:
        """Native units -> [0, 1], 0 = open, 1 = closed."""
        span = self.closed_pos - self.open_pos
        value = float(np.clip((float(raw) - self.open_pos) / span, 0.0, 1.0))
        if value <= self.deadband:
            return 0.0
        if value >= 1.0 - self.deadband:
            return 1.0
        return value

    def denormalize(self, value: float) -> float:
        """[0, 1] -> native units."""
        span = self.closed_pos - self.open_pos
        return float(self.open_pos + float(np.clip(value, 0.0, 1.0)) * span)


@dataclass
class CameraSpec:
    name: str
    mount: str = "world"
    pose: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    fovy_deg: float = 58.0
    resolution: tuple[int, int] = (640, 480)

    def __post_init__(self) -> None:
        self.pose = tuple(float(v) for v in self.pose)
        if len(self.pose) != 7:
            raise SpecError(
                f"camera {self.name!r}: pose must be 7 values (xyz + wxyz quat), "
                f"got {len(self.pose)}"
            )
        self.fovy_deg = float(self.fovy_deg)
        if not 0.0 < self.fovy_deg < 180.0:
            raise SpecError(
                f"camera {self.name!r}: fovy_deg must be in (0, 180), got {self.fovy_deg}"
            )
        self.resolution = (int(self.resolution[0]), int(self.resolution[1]))
        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise SpecError(
                f"camera {self.name!r}: resolution must be positive, got {self.resolution}"
            )

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    def intrinsics(self) -> np.ndarray:
        """Pinhole intrinsics implied by fovy and resolution (square pixels)."""
        fy = (self.height / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)
        fx = fy
        return np.array(
            [[fx, 0.0, self.width / 2.0], [0.0, fy, self.height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class WorkspaceBounds:
    """Axis-aligned safety envelope for the end effector.

    Present from Stage 0 because autonomous operation (Stages 6 and 11) removes
    the human who would otherwise notice the arm leaving the table.
    """

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self) -> None:
        self.lower = tuple(float(v) for v in self.lower)
        self.upper = tuple(float(v) for v in self.upper)
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise SpecError("workspace bounds must have 3 components each")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise SpecError(
                f"workspace lower {self.lower} must be strictly below upper {self.upper}"
            )

    def contains(self, xyz: np.ndarray) -> bool:
        xyz = np.asarray(xyz, dtype=np.float64)[:3]
        return bool(
            np.all(xyz >= np.asarray(self.lower)) and np.all(xyz <= np.asarray(self.upper))
        )


@dataclass
class SensingSpec:
    """What this arm can actually measure.

    Declared rather than assumed: v2 gave the safety monitor a force limit
    while Observation carried nothing to read, so the limit could never fire.
    The startup check (control/compat.py) now rejects that combination.
    """

    joint_current: bool = False
    joint_torque: bool = False
    ee_wrench: bool = False
    estop: bool = False

    def provides(self, key: str) -> bool:
        return bool(getattr(self, key, False))


@dataclass
class RobotSpec:
    name: str
    joints: list[JointSpec]
    ee_link: str
    gripper: GripperSpec
    cameras: list[CameraSpec] = field(default_factory=list)
    #: Point that ee_pose reports, if not the ee_link body origin -- usually
    #: the grasp point between the fingertips, which is what the workspace
    #: envelope and any reaching task actually care about.
    ee_site: str | None = None
    #: Joint configuration to reset to. Defaults to the middle of every range,
    #: which is fine for symmetric limits and nonsense for a joint like YAM's
    #: joint2, whose range is [0, 3.67]: its midpoint folds the arm over.
    home: list[float] | None = None
    workspace: WorkspaceBounds | None = None
    version: str = "1"
    supported_modes: list[ControlMode] = field(default_factory=list)
    sensing: SensingSpec = field(default_factory=SensingSpec)
    mjcf_path: str | None = field(default=None, metadata={"hash": False})
    urdf_path: str | None = field(default=None, metadata={"hash": False})
    default_control_mode: ControlMode = ControlMode.JOINT_POSITION
    control_hz: float = 30.0
    source_path: Path | None = field(default=None, metadata={"hash": False})

    def __post_init__(self) -> None:
        if not self.joints:
            raise SpecError(f"robot {self.name!r}: must declare at least one joint")
        names = [j.name for j in self.joints]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise SpecError(
                f"robot {self.name!r}: duplicate joint names {sorted(duplicates)}"
            )
        cam_names = [c.name for c in self.cameras]
        dup_cams = {n for n in cam_names if cam_names.count(n) > 1}
        if dup_cams:
            raise SpecError(
                f"robot {self.name!r}: duplicate camera names {sorted(dup_cams)}"
            )
        self.default_control_mode = ControlMode(self.default_control_mode)
        self.supported_modes = [ControlMode(m) for m in self.supported_modes]
        if not self.supported_modes:
            self.supported_modes = [self.default_control_mode]
        if self.default_control_mode not in self.supported_modes:
            raise SpecError(
                f"robot {self.name!r}: default_control_mode "
                f"{self.default_control_mode.value!r} is not in supported_modes "
                f"{[m.value for m in self.supported_modes]}"
            )
        self.control_hz = float(self.control_hz)
        if self.control_hz <= 0:
            raise SpecError(
                f"robot {self.name!r}: control_hz must be positive, got {self.control_hz}"
            )
        if self.home is not None:
            self.home = [float(v) for v in self.home]
            if len(self.home) != len(self.joints):
                raise SpecError(
                    f"robot {self.name!r}: home has {len(self.home)} values for "
                    f"{len(self.joints)} joints"
                )
            outside = [
                j.name for j, v in zip(self.joints, self.home, strict=True)
                if not j.lower <= v <= j.upper
            ]
            if outside:
                raise SpecError(f"robot {self.name!r}: home is outside the limits of {outside}")

    @property
    def dof(self) -> int:
        return len(self.joints)

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def camera_names(self) -> list[str]:
        return [c.name for c in self.cameras]

    @property
    def lower_limits(self) -> np.ndarray:
        return np.array([j.lower for j in self.joints], dtype=np.float64)

    @property
    def upper_limits(self) -> np.ndarray:
        return np.array([j.upper for j in self.joints], dtype=np.float64)

    def camera(self, name: str) -> CameraSpec:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise SpecError(
            f"robot {self.name!r}: no camera named {name!r}; have {self.camera_names}"
        )

    def clip_joints(self, q: np.ndarray) -> np.ndarray:
        """Clamp a joint vector into the declared limits."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.dof,):
            raise SpecError(
                f"robot {self.name!r}: expected {self.dof} joint values, got shape {q.shape}"
            )
        return np.clip(q, self.lower_limits, self.upper_limits)

    def neutral_joints(self) -> np.ndarray:
        """Midpoint of every joint range."""
        return (self.lower_limits + self.upper_limits) / 2.0

    def home_joints(self) -> np.ndarray:
        """The declared home pose, or the range midpoints if none is declared."""
        if self.home is None:
            return self.neutral_joints()
        return np.asarray(self.home, dtype=np.float64)

    def supports(self, mode: ControlMode) -> bool:
        return ControlMode(mode) in self.supported_modes

    def content_hash(self) -> str:
        """Stable hash of this spec's content, for dataset provenance (§4.5).

        Covers the spec's fields plus the *content* of the model files it
        references. Absolute paths are excluded deliberately: they are
        machine-local, and a hash that changes when a checkout moves cannot
        tell you whether two datasets were recorded against the same robot.
        A changed MJCF does change the hash, because that is a different robot.
        """
        return content_hash({"spec": self, "assets": self.asset_digests()})

    def asset_digests(self) -> dict[str, str | None]:
        """Content digests of the model files, following MJCF includes."""
        return {
            "mjcf": _file_tree_digest(self.mjcf_path),
            "urdf": _file_tree_digest(self.urdf_path),
        }

    # ------------------------------------------------------------------ load

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: Path | None = None) -> RobotSpec:
        data = dict(data)
        try:
            joints = [JointSpec(**j) for j in data.pop("joints")]
            gripper = GripperSpec(**data.pop("gripper"))
        except KeyError as exc:
            raise SpecError(f"robot spec missing required section: {exc}") from exc
        except TypeError as exc:
            raise SpecError(f"robot spec has a malformed joint/gripper entry: {exc}") from exc

        cameras = [CameraSpec(**c) for c in data.pop("cameras", [])]
        try:
            sensing = SensingSpec(**data.pop("sensing", {}))
        except TypeError as exc:
            raise SpecError(f"robot spec has a malformed sensing section: {exc}") from exc
        workspace_data = data.pop("workspace", None)
        workspace = WorkspaceBounds(**workspace_data) if workspace_data else None

        # Asset paths in a spec are written relative to the spec file itself, so a
        # spec stays portable no matter where the process was launched from.
        if source_path is not None:
            for key in ("mjcf_path", "urdf_path"):
                if data.get(key):
                    data[key] = str((source_path.parent / data[key]).resolve())

        try:
            return cls(
                joints=joints,
                gripper=gripper,
                cameras=cameras,
                workspace=workspace,
                sensing=sensing,
                source_path=source_path,
                **data,
            )
        except TypeError as exc:
            raise SpecError(f"unknown or missing field in robot spec: {exc}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotSpec:
        path = Path(path)
        if not path.is_file():
            raise SpecError(f"robot spec not found: {path}")
        with path.open() as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise SpecError(f"robot spec {path} must contain a YAML mapping")
        try:
            return cls.from_dict(data, source_path=path.resolve())
        except SpecError as exc:
            raise SpecError(f"{path}: {exc}") from exc
