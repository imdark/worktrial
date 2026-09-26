"""Straight-line sweeps with no target: a test bed for the safety monitors.

The gripper tip moves front and back along one line, then down and up along
another, at a fixed orientation and a set tip speed, for a number of cycles.
Nothing is looked for or picked. An operator puts a hand or an object in the
way, and the contact and hazard monitors' logs show how soon the arm stopped
and how soon it went on (scripts/run_sweep.py runs it and reports that).

    tip ---- front/back:  center - [travel_x, 0, 0]  <->  center + [travel_x, 0, 0]
         \\-- down/up:     center - [0, 0, travel_z]  <->  center + [0, 0, travel_z]

Each line is solved by IK every ``step_m`` along its length and played back at
``tip_speed`` (m/s, average) with a minimum-jerk start and stop, so the tip really travels in a
straight line (a joint-space move between the ends would bow). Phases are
named forward / back / down / up, none of them a phase where the contact
monitor expects contact, so its strict limits apply throughout.

Reuses PickLiftPlacePolicy's motion machinery (arrival waits, sag
compensation, the phase attribute the monitors read) and needs no vision.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from typing import Any

import numpy as np

from teleop_sim.core.clock import Clock
from teleop_sim.core.registry import POLICIES, register
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import ActionChunk, Observation
from teleop_sim.robots.kinematics import rot_y, rot_z
from teleop_sim.robots.trajectory import min_jerk
from teleop_sim.tasks.pick_lift_place import PickLiftPlacePolicy, TaskFailed


@register(POLICIES, "linear_sweep")
class LinearSweepPolicy(PickLiftPlacePolicy):
    def __init__(
        self,
        spec: RobotSpec,
        clock: Clock,
        center: list[float] = (0.45, -0.05, 0.20),
        travel_x: float = 0.08,
        travel_z: float = 0.05,
        pitch_deg: float = 10.0,
        tip_speed: float = 0.08,
        cycles: int = 4,
        pause_s: float = 0.5,
        step_m: float = 0.01,
        min_z: float = 0.10,
        line_arrive_timeout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("instruction", "sweep the gripper along two lines")
        kwargs.pop("vision", None)
        super().__init__(spec, clock, vision={"type": "none"}, **kwargs)
        self.center = np.asarray(center, dtype=float)
        self.travel_x, self.travel_z = float(travel_x), float(travel_z)
        self.pitch_deg = float(pitch_deg)
        self.tip_speed = float(tip_speed)
        self.cycles = int(cycles)
        self.pause_s = float(pause_s)
        self.step_m = float(step_m)
        # The pick waits after every move until the arm is within arrive_tol of
        # its target (it must be, before grasping). A sweep has nothing to reach:
        # by default the next line starts as soon as this one's commands end.
        self.line_arrive_timeout = float(line_arrive_timeout)
        if self.tip_speed <= 0 or self.tip_speed > 0.25:
            raise ValueError(f"linear_sweep: tip_speed must be in (0, 0.25] m/s, got {tip_speed}")
        if self.center[2] - self.travel_z < float(min_z):
            raise ValueError(
                f"linear_sweep: the down stroke would reach z={self.center[2] - self.travel_z:.3f}"
                f" m, below min_z={min_z} (keep the gripper clear of the table)"
            )
        self.vision = "none"  # never built: nothing is looked for
        self.lines = {
            "x": (self.center - [self.travel_x, 0, 0], self.center + [self.travel_x, 0, 0]),
            "z": (self.center - [0, 0, self.travel_z], self.center + [0, 0, self.travel_z]),
        }

    def _build_vision(self, config: dict | None = None):
        return "none"  # nothing is looked for; the monitors read the cameras themselves

    # ------------------------------------------------------------ script

    def _run(self) -> Generator[ActionChunk | None, Observation, None]:
        obs = yield None
        rot = self._rotation()
        x0, x1 = self.lines["x"]
        z0, z1 = self.lines["z"]
        self.log.event(
            "sweep_plan",
            center=self.center.round(3),
            travel_x=self.travel_x,
            travel_z=self.travel_z,
            tip_speed=self.tip_speed,
            cycles=self.cycles,
        )
        obs = yield from self._move("go_to_start", self._ik(x0, rot), 0.0, obs)
        for cycle in range(1, self.cycles + 1):
            self.log.event("cycle", n=cycle)
            obs = yield from self._line("forward", x0, x1, rot, obs)
            obs = yield from self._line("back", x1, x0, rot, obs)
            obs = yield from self._move("to_z_line", self._ik(z1, rot), 0.0, obs)
            obs = yield from self._line("down", z1, z0, rot, obs)
            obs = yield from self._line("up", z0, z1, rot, obs)
            obs = yield from self._move("to_x_line", self._ik(x0, rot), 0.0, obs)
        if self.return_to_start:
            obs = yield from self._move("return", self.robot_spec.home_joints(), 0.0, obs)

    def _trajectory(
        self, q: np.ndarray, gripper: float, obs: Observation, speed: float, settle: float
    ) -> ActionChunk:
        """The moves between lines, with the same minimum-jerk profile as the lines.

        A minimum-jerk move peaks at 1.875x its average speed (a cosine ease at
        1.57x), so the duration is set from the peak, keeping joints at or under
        ``speed``."""
        q0, g0 = self._q_cmd, self._g_cmd
        q1 = self.robot_spec.clip_joints(q)
        duration = max(1.875 * float(np.max(np.abs(q1 - q0))) / speed, 0.3)
        n = max(1, math.ceil(duration * self.hz))
        now = self.clock.now()
        actions = [
            self._action(q0 + min_jerk(k / n) * (q1 - q0), g0 + (k / n) * (gripper - g0), now, obs)
            for k in range(1, n + 1)
        ]
        actions += [self._action(q1, gripper, now, obs) for _ in range(math.ceil(settle * self.hz))]
        self._q_cmd, self._g_cmd = q1, float(gripper)
        return ActionChunk(actions=actions, obs_timestamp=obs.timestamp, horizon_hz=self.hz)

    def _rotation(self) -> np.ndarray:
        yaw = math.atan2(self.center[1], self.center[0])
        home_rot = self.kin.frame(self.robot_spec.home_joints(), "site", self.robot_spec.ee_site)[1]
        return rot_z(yaw) @ rot_y(math.radians(self.pitch_deg)) @ home_rot

    def _ik(self, p: np.ndarray, rot: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray:
        seeds = [s for s in (seed, self._q_cmd, self.robot_spec.home_joints()) if s is not None]
        result = self.kin.ik_multi_seed(
            "site",
            self.robot_spec.ee_site,
            p,
            rot,
            seeds + list(self._GRASP_SEEDS),
            pos_tol=2e-3,
            rot_tol=math.radians(1.0),
        )
        if not result.ok:
            raise TaskFailed("unreachable", f"{p.round(3)} off by {result.pos_err * 1000:.0f} mm")
        return result.q

    def _line(
        self, label: str, p0: np.ndarray, p1: np.ndarray, rot: np.ndarray, obs: Observation
    ) -> Generator[ActionChunk, Observation, Observation]:
        """Move the tip in a straight line from p0 to p1 at tip_speed."""
        length = float(np.linalg.norm(p1 - p0))
        n_pts = max(2, math.ceil(length / self.step_m) + 1)
        grid = np.linspace(0.0, 1.0, n_pts)
        qs, seed = [], self._q_cmd
        for s in grid:
            q = self._ik(p0 + s * (p1 - p0), rot, seed)
            qs.append(q)
            seed = q
        qs = np.array(qs)
        duration = max(length / self.tip_speed, 0.2)
        n = max(1, math.ceil(duration * self.hz))
        now = self.clock.now()
        actions = []
        for k in range(1, n + 1):
            s = min_jerk(k / n)  # zero velocity and acceleration at both ends
            q = np.array([np.interp(s, grid, qs[:, j]) for j in range(qs.shape[1])])
            actions.append(self._action(q, 0.0, now, obs))
        actions += [
            self._action(qs[-1], 0.0, now, obs) for _ in range(math.ceil(self.pause_s * self.hz))
        ]
        self._q_cmd, self._g_cmd = qs[-1], 0.0
        self.phase = label
        self.log.event(
            "line", phase=label, start=p0.round(3), end=p1.round(3), seconds=round(duration, 2)
        )
        obs = yield ActionChunk(actions=actions, obs_timestamp=obs.timestamp, horizon_hz=self.hz)
        waited = 0.0
        while self._tracking_error(obs) > self.arrive_tol and waited < self.line_arrive_timeout:
            if self.sag_compensation:
                shortfall = self._q_cmd - obs.joint_pos
                self._bias = np.clip(self._bias + 0.5 * shortfall, -self.max_bias, self.max_bias)
            obs = yield self._hold(obs, seconds=0.2)
            waited += 0.2
        return obs
