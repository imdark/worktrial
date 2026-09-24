"""Config -> wired system.

A registry lookup and a YAML file. Switching from teleoperated to autonomous is
``source.type``; switching from simulation to hardware is ``robot.type``, and
-- because success detection and reset split along the same line (§3.3) --
``success.type`` and ``reset.type`` change with it. None of that touches code.
The startup check then validates the combination before anything moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import teleop_sim.builtins  # noqa: F401  -- registration must not depend on import order
from teleop_sim.control.loop import ControlLoop
from teleop_sim.control.sources import AsyncPolicySource, PolicySource, TeleopSource
from teleop_sim.core.clock import Clock, WallClock
from teleop_sim.core.protocols import (
    ActionSource,
    ResetStrategy,
    Robot,
    SafetyMonitor,
    SuccessDetector,
)
from teleop_sim.core.registry import (
    POLICIES,
    RESETS,
    RETARGETERS,
    ROBOTS,
    SAFETY,
    SOURCES,
    SUCCESS,
    TELEOPS,
    build,
    lookup,
)
from teleop_sim.core.spec import RobotSpec
from teleop_sim.envs.task import Task

DEFAULT_SAFETY = {"type": "sim_watchdog"}


class ConfigError(ValueError):
    """A run config is missing a section or names something unknown."""


@dataclass
class RunConfig:
    robot: dict[str, Any]
    source: dict[str, Any]
    success: dict[str, Any]
    reset: dict[str, Any]
    safety: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SAFETY))
    task: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    path: Path | None = None

    SECTIONS = ("robot", "source", "success", "reset", "safety", "task", "record", "seed")
    REQUIRED = ("robot", "source", "success", "reset")

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"run config not found: {path}")
        with path.open() as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ConfigError(f"run config {path} must contain a YAML mapping")

        unknown = set(data) - set(cls.SECTIONS)
        if unknown:
            raise ConfigError(
                f"{path}: unknown top-level section(s) {sorted(unknown)}; "
                f"expected any of {sorted(cls.SECTIONS)}"
            )
        for required in cls.REQUIRED:
            if required not in data:
                raise ConfigError(f"{path}: missing required section {required!r}")

        return cls(path=path.resolve(), **data)

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a path written relative to this config file."""
        candidate = Path(relative)
        if candidate.is_absolute():
            return candidate
        base = self.path.parent if self.path is not None else Path.cwd()
        return (base / candidate).resolve()


@dataclass
class System:
    spec: RobotSpec
    robot: Robot
    source: ActionSource
    success: SuccessDetector
    reset: ResetStrategy
    safety: SafetyMonitor
    task: Task
    loop: ControlLoop
    clock: Clock


def build_source(config: dict[str, Any], spec: RobotSpec, clock: Clock) -> ActionSource:
    config = dict(config)
    kind = config.pop("type", None)
    if kind is None:
        raise ConfigError("source config must contain a 'type' key")
    lookup(SOURCES, kind)  # validates the name and reports the alternatives

    if kind == "teleop":
        if "device" not in config:
            raise ConfigError("teleop source requires a 'device' section")
        device_cfg = dict(config.pop("device"))
        retarget_cfg = dict(config.pop("retarget", {"type": "identity"}))
        teleop = build(TELEOPS, device_cfg, spec=spec, clock=clock)
        retargeter = build(RETARGETERS, retarget_cfg, spec=spec)
        return TeleopSource(teleop, retargeter, **config)

    if kind in ("policy", "async_policy"):
        if "policy" not in config:
            raise ConfigError(f"{kind} source requires a 'policy' section")
        policy_cfg = dict(config.pop("policy"))
        policy = build(POLICIES, policy_cfg, spec=spec, clock=clock)
        cls = PolicySource if kind == "policy" else AsyncPolicySource
        return cls(policy, clock=clock, **config)

    raise ConfigError(f"source type {kind!r} is registered but has no builder")


def build_system(config: RunConfig, clock: Clock | None = None) -> System:
    clock = clock or WallClock()

    robot_cfg = dict(config.robot)
    spec_path = robot_cfg.pop("spec", None)
    if spec_path is None:
        raise ConfigError("robot config must name a 'spec' file")
    spec = RobotSpec.from_yaml(config.resolve(spec_path))

    # The cell the robot is placed in. A property of the run, not of the robot:
    # the same robot description drives the real arm with no scene at all.
    scene_path = robot_cfg.pop("scene", None)
    if scene_path is not None:
        from teleop_sim.core.parts import SceneSpec

        spec = spec.with_scene(SceneSpec.from_yaml(config.resolve(scene_path)))

    robot = build(ROBOTS, robot_cfg, spec=spec, clock=clock)
    source = build_source(config.source, spec=spec, clock=clock)
    success = build(SUCCESS, dict(config.success))
    reset = build(RESETS, dict(config.reset))

    safety_cfg = dict(config.safety or DEFAULT_SAFETY)
    safety = build(SAFETY, safety_cfg, spec=spec)
    safety_params = {k: v for k, v in safety_cfg.items() if k != "type"}

    task_cfg = dict(config.task)
    task = Task(
        name=task_cfg.pop("name", "unnamed"),
        success=success,
        reset=reset,
        metadata=task_cfg,
    )

    loop = ControlLoop(
        robot,
        source,
        success,
        reset,
        safety,
        clock=clock,
        safety_config=safety_params,
    )
    return System(
        spec=spec,
        robot=robot,
        source=source,
        success=success,
        reset=reset,
        safety=safety,
        task=task,
        loop=loop,
        clock=clock,
    )
