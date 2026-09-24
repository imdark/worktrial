"""Simulation-side safety: nothing to damage, everything to bound.

In sim a trip costs a wasted episode, so this monitor only ends episodes. The
hardware counterpart additionally commands a physical safe stop.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from teleop_sim.core.protocols import SafetyMonitor, SafetyVerdict
from teleop_sim.core.registry import SAFETY, register
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Observation, Outcome


@dataclass
class SimWatchdogConfig:
    max_steps: int = 600
    max_seconds: float | None = None

    # Stall detection is opt-in because a human pausing mid-demonstration looks
    # exactly like a stalled policy. Autonomous configs enable it; teleop
    # configs leave it off.
    stall_steps: int | None = None
    stall_eps: float = 1e-3

    enforce_workspace: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.stall_steps is not None and self.stall_steps <= 0:
            raise ValueError(f"stall_steps must be positive, got {self.stall_steps}")


@register(SAFETY, "sim_watchdog")
class SimWatchdog(SafetyMonitor):
    def __init__(self, spec: RobotSpec | None = None, **kwargs: object) -> None:
        self.spec = spec
        try:
            self.config = SimWatchdogConfig(**kwargs)  # type: ignore[arg-type]
        except TypeError as exc:
            accepted = [f.name for f in fields(SimWatchdogConfig)]
            raise ValueError(
                f"sim_watchdog: {exc}. Accepted keys: {accepted}. Force, current and "
                "e-stop limits belong to safety type 'hardware_safety' (Stage 11)."
            ) from exc
        self._start_time: float | None = None
        self._prev_joints: np.ndarray | None = None
        self._still_for = 0

    def reset(self) -> None:
        self._start_time = None
        self._prev_joints = None
        self._still_for = 0

    def check(self, obs: Observation, step: int) -> SafetyVerdict | None:
        if self._start_time is None:
            self._start_time = obs.timestamp

        if step >= self.config.max_steps:
            return SafetyVerdict(
                Outcome.TIMEOUT, "max_steps", f"reached step limit {self.config.max_steps}"
            )

        if self.config.max_seconds is not None:
            elapsed = obs.timestamp - self._start_time
            if elapsed >= self.config.max_seconds:
                return SafetyVerdict(
                    Outcome.TIMEOUT,
                    "max_seconds",
                    f"elapsed {elapsed:.2f}s >= {self.config.max_seconds}s",
                )

        return self._check_stall(obs) or self._check_workspace(obs)

    def _check_stall(self, obs: Observation) -> SafetyVerdict | None:
        if self.config.stall_steps is None:
            return None

        if self._prev_joints is not None:
            motion = float(np.max(np.abs(obs.joint_pos - self._prev_joints)))
            self._still_for = self._still_for + 1 if motion < self.config.stall_eps else 0
        self._prev_joints = obs.joint_pos.copy()

        if self._still_for >= self.config.stall_steps:
            return SafetyVerdict(
                Outcome.WATCHDOG,
                "stalled",
                f"joints moved < {self.config.stall_eps} rad for "
                f"{self._still_for} consecutive steps",
            )
        return None

    def _check_workspace(self, obs: Observation) -> SafetyVerdict | None:
        if not self.config.enforce_workspace:
            return None
        if self.spec is None or self.spec.workspace is None or obs.ee_pose is None:
            return None
        if not self.spec.workspace.contains(obs.ee_pose[:3]):
            return SafetyVerdict(
                Outcome.WATCHDOG,
                "out_of_workspace",
                f"end effector at {obs.ee_pose[:3]} left the declared envelope",
            )
        return None
