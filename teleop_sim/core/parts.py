"""Composable robot descriptions: arm + end effector (+ scene).

A robot used to be one flat description, which baked the gripper into the arm
and the table into the robot. That made "same arm, different gripper" a copy
of the whole description, and meant the real robot's description mentioned
glasses. Each physical thing now has its own file:

    arm            joints, limits, home, the flange an end effector bolts to
    end effector   gripper calibration, grasp point, the cameras it carries
    scene          the table, objects, fixed cameras and workspace -- the cell

``compose`` resolves them into the same ``RobotSpec`` everything downstream
already consumes, so the loop, policies, recorder and conformance suite do not
change. The real robot uses arm + end effector with no scene; the simulator
adds one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from teleop_sim.core.spec import (
    CameraSpec,
    GripperSpec,
    JointSpec,
    RobotSpec,
    SensingSpec,
    SpecError,
    WorkspaceBounds,
)
from teleop_sim.core.types import ControlMode

Pose = tuple[float, float, float, float, float, float, float]


def _load_yaml(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise SpecError(f"{kind} description not found: {path}")
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SpecError(f"{kind} description {path} must contain a YAML mapping")
    return data


def _resolve(path: Path, value: str | None) -> str | None:
    """Asset paths are written relative to the description that names them."""
    return None if not value else str((path.parent / value).resolve())


def _build(cls, kind: str, path: Path, **kwargs: Any):
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise SpecError(f"{path}: unknown or missing field in {kind} description: {exc}") from exc
    except SpecError as exc:
        raise SpecError(f"{path}: {exc}") from exc


def _vec(values: Any, size: int, name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if len(out) != size:
        raise SpecError(f"{name} must have {size} values, got {len(out)}")
    return out


@dataclass
class ArmSpec:
    name: str
    joints: list[JointSpec]
    #: Body the end effector attaches to.
    flange: str
    #: Body at the root of the arm's kinematic tree, placed into a scene.
    root_body: str
    mjcf_path: str | None = field(default=None, metadata={"hash": False})
    version: str = "1"
    home: list[float] | None = None
    default_control_mode: ControlMode = ControlMode.JOINT_POSITION
    supported_modes: list[ControlMode] = field(default_factory=list)
    control_hz: float = 30.0
    sensing: SensingSpec = field(default_factory=SensingSpec)
    source_path: Path | None = field(default=None, metadata={"hash": False})

    @classmethod
    def from_yaml(cls, path: str | Path) -> ArmSpec:
        path = Path(path).resolve()
        data = _load_yaml(path, "arm")
        try:
            joints = [JointSpec(**j) for j in data.pop("joints")]
        except KeyError as exc:
            raise SpecError(f"{path}: arm description needs a 'joints' section") from exc
        sensing = SensingSpec(**data.pop("sensing", {}))
        data["mjcf_path"] = _resolve(path, data.get("mjcf_path"))
        return _build(cls, "arm", path, joints=joints, sensing=sensing, source_path=path, **data)


@dataclass
class EndEffectorSpec:
    name: str
    gripper: GripperSpec
    #: Body in the end effector's MJCF that is attached to the arm's flange.
    root_body: str
    mjcf_path: str | None = field(default=None, metadata={"hash": False})
    version: str = "1"
    #: Pose of root_body relative to the flange: xyz + wxyz quaternion.
    mount: Pose = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    #: Grasp point, reported as ee_pose.
    ee_site: str | None = None
    #: Cameras carried by the end effector; mounts are its own bodies.
    cameras: list[CameraSpec] = field(default_factory=list)
    source_path: Path | None = field(default=None, metadata={"hash": False})

    def __post_init__(self) -> None:
        self.mount = _vec(self.mount, 7, f"end effector {self.name!r} mount")

    @classmethod
    def from_yaml(cls, path: str | Path) -> EndEffectorSpec:
        path = Path(path).resolve()
        data = _load_yaml(path, "end effector")
        try:
            gripper = GripperSpec(**data.pop("gripper"))
        except KeyError as exc:
            raise SpecError(f"{path}: end effector description needs a 'gripper'") from exc
        cameras = [CameraSpec(**c) for c in data.pop("cameras", [])]
        data["mjcf_path"] = _resolve(path, data.get("mjcf_path"))
        return _build(
            cls, "end effector", path, gripper=gripper, cameras=cameras, source_path=path, **data
        )


@dataclass
class SceneSpec:
    """The cell: everything around the robot.

    In sim that includes objects and physics options. On hardware the same
    description still carries the fixed cameras and the workspace envelope,
    which belong to the table the arm is bolted to, not to the arm.
    """

    name: str
    mjcf_path: str | None = field(default=None, metadata={"hash": False})
    version: str = "1"
    #: Where the arm's root body sits in the scene: xyz + wxyz quaternion.
    arm_mount: Pose = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    cameras: list[CameraSpec] = field(default_factory=list)
    workspace: WorkspaceBounds | None = None
    source_path: Path | None = field(default=None, metadata={"hash": False})

    def __post_init__(self) -> None:
        self.arm_mount = _vec(self.arm_mount, 7, f"scene {self.name!r} arm_mount")
        for camera in self.cameras:
            if camera.mount != "world":
                raise SpecError(
                    f"scene {self.name!r}: camera {camera.name!r} is mounted on "
                    f"{camera.mount!r}; a scene's cameras are fixed, so mount must be "
                    "'world'. Cameras on the robot belong to the end effector."
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> SceneSpec:
        path = Path(path).resolve()
        data = _load_yaml(path, "scene")
        cameras = [CameraSpec(**c) for c in data.pop("cameras", [])]
        workspace = data.pop("workspace", None)
        data["mjcf_path"] = _resolve(path, data.get("mjcf_path"))
        return _build(
            cls,
            "scene",
            path,
            cameras=cameras,
            workspace=WorkspaceBounds(**workspace) if workspace else None,
            source_path=path,
            **data,
        )


@dataclass
class Assembly:
    """Which parts a composed RobotSpec was built from, for the model builder
    and for provenance: the content hash covers every part."""

    arm: ArmSpec
    end_effector: EndEffectorSpec
    scene: SceneSpec | None = None


def compose(
    arm: ArmSpec,
    end_effector: EndEffectorSpec,
    scene: SceneSpec | None = None,
    name: str | None = None,
    source_path: Path | None = None,
) -> RobotSpec:
    """Resolve parts into the one RobotSpec the rest of the system consumes."""
    cameras = [*end_effector.cameras, *(scene.cameras if scene else [])]
    names = [c.name for c in cameras]
    clash = sorted({n for n in names if names.count(n) > 1})
    if clash:
        raise SpecError(
            f"camera name(s) {clash} are declared by both the end effector "
            f"{end_effector.name!r} and the scene {scene.name if scene else None!r}"
        )
    return RobotSpec(
        name=name or f"{arm.name}+{end_effector.name}",
        version=f"{arm.version}/{end_effector.version}",
        joints=list(arm.joints),
        home=None if arm.home is None else list(arm.home),
        ee_link=end_effector.root_body,
        ee_site=end_effector.ee_site,
        gripper=end_effector.gripper,
        cameras=cameras,
        workspace=scene.workspace if scene else None,
        default_control_mode=arm.default_control_mode,
        supported_modes=list(arm.supported_modes),
        control_hz=arm.control_hz,
        sensing=arm.sensing,
        assembly=Assembly(arm=arm, end_effector=end_effector, scene=scene),
        source_path=source_path,
    )


def load_composition(path: Path, data: dict[str, Any]) -> RobotSpec:
    """A robot description that names its parts:

        name: yam
        arm: ../arms/yam.yaml
        end_effector: ../end_effectors/yam_linear.yaml
        scene: ../../scenes/glasses_table.yaml     # optional; sim only
    """
    unknown = set(data) - {"name", "arm", "end_effector", "scene"}
    if unknown:
        raise SpecError(f"{path}: unknown field(s) in robot composition: {sorted(unknown)}")
    for required in ("arm", "end_effector"):
        if required not in data:
            raise SpecError(f"{path}: a robot composition needs '{required}'")
    arm = ArmSpec.from_yaml(path.parent / data["arm"])
    end_effector = EndEffectorSpec.from_yaml(path.parent / data["end_effector"])
    scene = SceneSpec.from_yaml(path.parent / data["scene"]) if data.get("scene") else None
    return compose(arm, end_effector, scene, name=data.get("name"), source_path=path)


def with_scene(spec: RobotSpec, scene: SceneSpec | None) -> RobotSpec:
    """The same robot placed in a different cell (or in none)."""
    if spec.assembly is None:
        raise SpecError(
            f"robot {spec.name!r} is a monolithic description; only a composed robot "
            "(arm + end_effector) can be placed into a scene"
        )
    return compose(
        spec.assembly.arm,
        spec.assembly.end_effector,
        scene,
        name=spec.name,
        source_path=spec.source_path,
    )
