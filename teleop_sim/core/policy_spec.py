"""What a checkpoint needs in order to run (§4.3).

RobotSpec made hardware a data contract. This is its counterpart for policies.
Without it a checkpoint's requirements -- camera *names*, resolutions, control
mode, rate -- are implicit, and a mismatch surfaces as a KeyError at best or as
quiet degradation at worst. That is the central risk in "swap the model".

Written into every checkpoint at train time, read at load time, and checked
once before step 0 by control/compat.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from teleop_sim.core.hashing import content_hash
from teleop_sim.core.types import ControlMode, Observation


class PolicySpecError(ValueError):
    """A policy spec is malformed or names something unknown."""


@dataclass
class CameraRequirement:
    name: str
    height: int
    width: int

    def __post_init__(self) -> None:
        self.height = int(self.height)
        self.width = int(self.width)
        if self.height <= 0 or self.width <= 0:
            raise PolicySpecError(
                f"camera requirement {self.name!r}: resolution must be positive"
            )

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class PolicySpec:
    policy_id: str
    control_mode: ControlMode = ControlMode.JOINT_POSITION
    checkpoint_hash: str | None = None
    cameras: list[CameraRequirement] = field(default_factory=list)
    obs_keys: list[str] = field(default_factory=lambda: ["joint_pos", "gripper"])
    obs_history: int = 1
    action_horizon: int = 1
    expected_control_hz: float = 30.0
    language_conditioned: bool = False
    trained_against_robot_spec: str | None = None

    def __post_init__(self) -> None:
        self.control_mode = ControlMode(self.control_mode)
        self.obs_history = int(self.obs_history)
        self.action_horizon = int(self.action_horizon)
        self.expected_control_hz = float(self.expected_control_hz)

        if self.obs_history < 1:
            raise PolicySpecError(f"obs_history must be >= 1, got {self.obs_history}")
        if self.action_horizon < 1:
            raise PolicySpecError(f"action_horizon must be >= 1, got {self.action_horizon}")
        if self.expected_control_hz <= 0:
            raise PolicySpecError(
                f"expected_control_hz must be positive, got {self.expected_control_hz}"
            )

        unknown = [key for key in self.obs_keys if key not in Observation.OBS_KEYS]
        if unknown:
            raise PolicySpecError(
                f"policy {self.policy_id!r} names unknown observation key(s) {unknown}; "
                f"known keys: {list(Observation.OBS_KEYS)}"
            )

        names = [camera.name for camera in self.cameras]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise PolicySpecError(
                f"policy {self.policy_id!r}: duplicate camera names {sorted(duplicates)}"
            )

    @property
    def camera_names(self) -> list[str]:
        return [camera.name for camera in self.cameras]

    def camera(self, name: str) -> CameraRequirement:
        for camera in self.cameras:
            if camera.name == name:
                return camera
        raise PolicySpecError(
            f"policy {self.policy_id!r} has no camera {name!r}; have {self.camera_names}"
        )

    def content_hash(self) -> str:
        return content_hash(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySpec:
        data = dict(data)
        cameras = [CameraRequirement(**c) for c in data.pop("cameras", [])]
        try:
            return cls(cameras=cameras, **data)
        except TypeError as exc:
            raise PolicySpecError(f"unknown or missing field in policy spec: {exc}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicySpec:
        path = Path(path)
        if not path.is_file():
            raise PolicySpecError(f"policy spec not found: {path}")
        with path.open() as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise PolicySpecError(f"policy spec {path} must contain a YAML mapping")
        try:
            return cls.from_dict(data)
        except PolicySpecError as exc:
            raise PolicySpecError(f"{path}: {exc}") from exc
