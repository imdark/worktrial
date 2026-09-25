"""Pick up a glass, lift it, put it back where it was -- found by a vision model.

The MVP of the VLM-guided track. It is a Policy like any other, so the same
control loop drives it in MuJoCo and on the real arm, and the same config
switch (``robot.type``) moves it from one to the other.

The sequence, one attempt:

    survey view  -> plan (smart model: which glass?) -> locate (fast model: pixels)
    refine view  -> locate again, closer
    grasp        -> above, standoff, in, close, lift
    verify       -> jaws + fast model: is it held?  (no -> put down, open, retry)
    place        -> hold, lower to where it was, open, back out, return to start

The vision model answers in pixels only. Pixels become a table position by
intersecting the camera ray with the table plane (perception/camera.py) --
the glass is transparent, and depth on it cannot be trusted -- and that
position becomes joint targets through the kinematic model of the same
RobotSpec the robot was built from. The model never commands the arm.

Written as a generator: ``_run`` reads top to bottom as the task, and every
``yield`` hands one motion segment to the loop as an ActionChunk and resumes
with the observation taken when that segment finished. A slow vision call
blocks the loop while the arm is standing still between segments; on the real
arm, robots_realtime holds position when commands pause, so slow is safe.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from teleop_sim.core.clock import Clock
from teleop_sim.core.policy_spec import CameraRequirement, PolicySpec
from teleop_sim.core.protocols import Policy
from teleop_sim.core.registry import POLICIES, register
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Action, ActionChunk, ActionOrigin, Observation, TaskEvent
from teleop_sim.perception.camera import CameraModel
from teleop_sim.perception.vision import Detection, Views, Vision, VisionError
from teleop_sim.robots.kinematics import Kinematics, rot_y, rot_z
from teleop_sim.runlog import RunLog


class TaskFailed(Exception):
    def __init__(self, tag: str, detail: str = "") -> None:
        super().__init__(f"{tag}: {detail}" if detail else tag)
        self.tag = tag


@dataclass
class ViewParams:
    """A wrist-camera viewpoint: behind the point of interest, looking down at it."""

    height: float  # camera height above the table, m
    tilt_deg: float  # optical axis tilted forward from straight down
    back: float  # camera set back from the point, towards the arm base, m


@dataclass
class GraspParams:
    """Side grasp relative to the glass, found by sweep in sim (envs/scripted_grasp.py)."""

    past_centre: float = 0.02  # grasp point beyond the glass centre, along the approach
    height: float = 0.09  # grasp-site height above the table
    pitch_deg: float = 45.0  # tool pitched down from the home orientation
    standoff: float = 0.12  # back off along the approach before going in
    clearance: float = 0.03  # extra height for the first waypoint
    lift: float = 0.12
    # Calibration: how many degrees further nose-down the real tool sits than
    # the model predicts for the same joints (Kronos mount angle is inferred,
    # not measured). Commanded pitch is reduced by this much.
    pitch_trim_deg: float = 0.0


def _params(cls, value: dict[str, Any] | None, default):
    if value is None:
        return default
    accepted = {f.name for f in fields(cls)}
    unknown = set(value) - accepted
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown keys {sorted(unknown)}; accepted {sorted(accepted)}"
        )
    return cls(**{**default.__dict__, **value})


Gate = Callable[[str, str], bool]
MoveGen = Generator[ActionChunk, Observation, Observation]


@register(POLICIES, "pick_lift_place")
class PickLiftPlacePolicy(Policy):
    def __init__(
        self,
        spec: RobotSpec,
        clock: Clock,
        instruction: str = "pick up the glass",
        vision: dict[str, Any] | None = None,
        wrist_camera: str = "wrist",
        overview_camera: str | None = None,
        table_z: float = 0.0,
        glass_height: float = 0.10,
        survey_center: list[float] = (0.40, 0.0),
        survey: dict[str, float] | None = None,
        refine: dict[str, float] | None = None,
        grasp: dict[str, float] | None = None,
        workspace_xy: dict[str, list[float]] | None = None,
        max_joint_speed: float = 0.35,
        approach_speed: float = 0.15,
        gripper_seconds: float = 1.2,
        settle_seconds: float = 0.5,
        hold_seconds: float = 2.0,
        max_attempts: int = 2,
        verify_grasp: bool = True,
        plan_target: bool = True,
        return_to_start: bool = True,
        min_finger_clearance: float = 0.04,
        empty_jaw_threshold: float = 0.9,
        reject_confidence: float = 0.8,
        look_only: bool = False,
        arrive_tol_deg: float = 1.5,
        arrive_timeout: float = 3.0,
        approach_only: bool = False,
        sag_compensation: bool = False,
        max_bias_deg: float = 8.6,
    ) -> None:
        self.robot_spec = spec
        self.clock = clock
        self.instruction = instruction
        self.vision_config = dict(vision or {"type": "claude"})
        self.wrist_camera = wrist_camera
        self.overview_camera = overview_camera
        self.table_z = float(table_z)
        self.glass_height = float(glass_height)
        self.survey_center = np.asarray(survey_center, dtype=float)
        self.survey = _params(ViewParams, survey, ViewParams(height=0.42, tilt_deg=30.0, back=0.20))
        self.refine = (
            None
            if refine is False
            else _params(ViewParams, refine, ViewParams(height=0.30, tilt_deg=35.0, back=0.15))
        )
        self.grasp = _params(GraspParams, grasp, GraspParams())
        if workspace_xy is not None:
            self.ws_lower = np.asarray(workspace_xy["lower"], float)
            self.ws_upper = np.asarray(workspace_xy["upper"], float)
        elif spec.workspace is not None:
            self.ws_lower = np.asarray(spec.workspace.lower[:2], float)
            self.ws_upper = np.asarray(spec.workspace.upper[:2], float)
        else:
            self.ws_lower, self.ws_upper = np.array([0.15, -0.45]), np.array([0.65, 0.45])
        self.max_joint_speed = float(max_joint_speed)
        self.approach_speed = float(approach_speed)
        self.gripper_seconds = float(gripper_seconds)
        self.settle_seconds = float(settle_seconds)
        self.hold_seconds = float(hold_seconds)
        self.max_attempts = int(max_attempts)
        self.verify_grasp = bool(verify_grasp)
        self.plan_target = bool(plan_target)
        self.return_to_start = bool(return_to_start)
        self.min_finger_clearance = float(min_finger_clearance)
        # Normalised jaw reading at or above which a close caught nothing.
        self.empty_jaw_threshold = float(empty_jaw_threshold)
        # A vision 'not held' overrides blocked jaws only when at least this sure.
        self.reject_confidence = float(reject_confidence)
        # Survey and refine views only, then home: checks perception on a new
        # rig without the arm going near the object.
        self.look_only = bool(look_only)
        # After each move, wait (up to arrive_timeout) for tracking error below this.
        self.arrive_tol = math.radians(float(arrive_tol_deg))
        self.arrive_timeout = float(arrive_timeout)
        # Stop at the standoff pose, photograph the alignment, back out.
        self.approach_only = bool(approach_only)
        # Offset commands by the measured gravity sag (real arms; sim needs none).
        self.sag_compensation = bool(sag_compensation)
        self.max_bias = math.radians(float(max_bias_deg))
        self.hz = spec.control_hz

        self.kin = Kinematics(spec)
        cameras = [wrist_camera] + ([overview_camera] if overview_camera else [])
        self.spec = PolicySpec(
            policy_id="pick_lift_place",
            control_mode=spec.default_control_mode,
            cameras=[self._camera_requirement(name) for name in cameras],
            obs_keys=["images", "joint_pos", "gripper"],
            expected_control_hz=spec.control_hz,
            language_conditioned=True,
            trained_against_robot_spec=spec.content_hash(),
        )

        self.vision: Vision | None = None
        self.log = RunLog(None)
        self.gate: Gate | None = None
        self._robot = None
        self.reset()

    def _camera_requirement(self, name: str) -> CameraRequirement:
        try:
            width, height = self.robot_spec.camera(name).resolution
        except Exception:
            width, height = 640, 480  # a camera the scene or rig provides, VGA by default
        return CameraRequirement(name=name, height=height, width=width)

    # ------------------------------------------------------------ wiring

    def bind(self, robot=None, run_log: RunLog | None = None, gate: Gate | None = None) -> None:
        """Runtime context a config file cannot carry: the robot (for the sim
        oracle), where to log, and an operator gate called before each motion."""
        self._robot = robot
        if run_log is not None:
            self.log = run_log
        self.gate = gate
        self.vision = None

    def _build_vision(self) -> Vision:
        cfg = dict(self.vision_config)
        kind = cfg.pop("type", "claude")
        if kind == "claude":
            from teleop_sim.perception.claude_vision import ClaudeVision

            return ClaudeVision(run_log=self.log, **cfg)
        if kind == "oracle":
            from teleop_sim.perception.oracle import OracleVision

            if self._robot is None or not hasattr(self._robot, "model"):
                raise ValueError("vision type 'oracle' needs a simulated robot bound with bind()")
            return OracleVision(self._robot, height=self.glass_height, **cfg)
        raise ValueError(f"unknown vision type {kind!r}; expected 'claude' or 'oracle'")

    # ------------------------------------------------------------ Policy API

    def reset(self) -> None:
        self._script: Generator | None = None
        self._q_cmd: np.ndarray | None = None
        self._g_cmd = 0.0
        self._bias = np.zeros(self.robot_spec.dof)
        self._events: set[str] = set()
        self.outcome: str | None = None
        self.failure_tag: str | None = None

    def poll_events(self) -> set[str]:
        events, self._events = self._events, set()
        return events

    def predict(self, obs: Observation) -> ActionChunk:
        if self.vision is None:
            self.vision = self._build_vision()
        if self.outcome is not None:
            return self._hold(obs)
        if self._script is None:
            self._q_cmd = obs.joint_pos.copy()
            self._g_cmd = float(obs.gripper)
            self._script = self._run()
            next(self._script)
        try:
            return self._script.send(obs)
        except StopIteration:
            self._finish("success")
        except TaskFailed as exc:
            self._finish("failure", exc.tag, str(exc))
        except VisionError as exc:
            self._finish("failure", "vision_error", str(exc))
        return self._hold(obs)

    def _finish(self, outcome: str, tag: str | None = None, detail: str = "") -> None:
        self.outcome, self.failure_tag = outcome, tag
        self.log.event("finished", outcome=outcome, tag=tag, detail=detail)
        self._events.add(
            TaskEvent.SUCCESS if outcome == "success" else TaskEvent.failure(tag or "")
        )

    # ------------------------------------------------------------ the task

    def _run(self) -> Generator[ActionChunk | None, Observation, None]:
        obs = yield None
        start_q = obs.joint_pos.copy()
        self.log.event("start", instruction=self.instruction, joints=start_q.round(3))
        target_desc: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            self.log.event("attempt", n=attempt)
            q_survey = self._view_pose(self.survey_center, self.survey)
            obs = yield from self._move("survey", q_survey, gripper=0.0, obs=obs)

            if target_desc is None:
                target_desc = self._plan(obs)
            p = self._locate(obs, target_desc, "survey")
            if p is None:
                continue

            if self.refine is not None:
                obs = yield from self._move("refine", self._view_pose(p[:2], self.refine), 0.0, obs)
                refined = self._locate(obs, target_desc, "refine")
                if refined is not None:
                    self.log.event(
                        "refined", moved_mm=round(1000 * float(np.linalg.norm(refined - p)), 1)
                    )
                    p = refined

            if self.look_only:
                self.log.event("look_only", table_xy=p[:2].round(4))
                obs = yield from self._move("survey_again", q_survey, 0.0, obs)
                obs = yield from self._move("return", start_q, 0.0, obs)
                return

            above, standoff, at, lifted = self._grasp_waypoints(p)
            obs = yield from self._move("above", above, 0.0, obs)
            obs = yield from self._move("standoff", standoff, 0.0, obs)
            if self.approach_only:
                self.log.image("standoff_alignment", obs.images[self.wrist_camera])
                self.log.event("approach_only", glass_xy=p[:2].round(4))
                obs = yield from self._move("clear", above, 0.0, obs)
                obs = yield from self._move("survey_again", q_survey, 0.0, obs)
                obs = yield from self._move("return", start_q, 0.0, obs)
                return
            obs = yield from self._move("reach_in", at, 0.0, obs, speed=self.approach_speed)
            obs = yield from self._move("close", at, 1.0, obs, settle=2 * self.settle_seconds)
            obs = yield from self._move("lift", lifted, 1.0, obs, speed=self.approach_speed)

            if self.verify_grasp and not self._grasp_held(obs, target_desc):
                # Never let go in the air: put whatever may be in the jaws back
                # on the table first, then back out and try again.
                obs = yield from self._move("put_down", at, 1.0, obs, speed=self.approach_speed)
                obs = yield from self._move("release", at, 0.0, obs, settle=self.settle_seconds)
                obs = yield from self._move("back_off", standoff, 0.0, obs)
                obs = yield from self._move("clear_failed", above, 0.0, obs)
                continue

            obs = yield from self._move("hold", lifted, 1.0, obs, settle=self.hold_seconds)
            obs = yield from self._move("lower", at, 1.0, obs, speed=self.approach_speed)
            obs = yield from self._move("open", at, 0.0, obs, settle=2 * self.settle_seconds)
            obs = yield from self._move("back_out", standoff, 0.0, obs, speed=self.approach_speed)
            obs = yield from self._move("clear", above, 0.0, obs)
            if self.return_to_start:
                obs = yield from self._move("survey_again", q_survey, 0.0, obs)
                obs = yield from self._move("return", start_q, 0.0, obs)
            return
        raise TaskFailed(
            "attempts_exhausted", f"{self.max_attempts} attempt(s) without a held glass"
        )

    # ------------------------------------------------------------ perception

    def _grasp_held(self, obs: Observation, target: str = "") -> bool:
        """Is the glass held? The jaws answer first, the vision model second.

        Jaws that closed all the way hold nothing -- no model call needed. Jaws
        stopped part-way hold *something*; the model is asked whether it is the
        glass and lifted clear, and is told what the jaws report. Measured in
        sim: 0.12 closing on a glass, 1.00 closing on air. A camera-only check
        failed there (Sonnet saw the other glass on the table and said "not
        held" about a glass it was looking straight down into).
        """
        jaw = float(obs.gripper)
        if jaw >= self.empty_jaw_threshold:
            self.log.event("verify", ok=False, jaw=round(jaw, 3), reason="jaws closed on nothing")
            return False
        verdict = self.vision.verify(
            self._views(obs),
            "The robot just closed its gripper on a drinking glass and lifted. The gripper's "
            f"jaws stopped at {jaw:.0%} of their closing travel, so something is between "
            "them. In the wrist camera a held glass looks large and centred between the two "
            "fingers, seen from above, often looking down into its opening; other glasses "
            "standing on the table are not the one in question. Is a glass held between the "
            "fingers and lifted clear of the table?"
            + (f" The object being picked is: {target} Any other object still on the table "
               "is a different one." if target else ""),
        )
        held = verdict.ok or verdict.confidence < self.reject_confidence
        self.log.event("verify", ok=held, jaw=round(jaw, 3), vision_ok=verdict.ok,
                       confidence=verdict.confidence, reason=verdict.reason, model=verdict.model)
        return held

    def _views(self, obs: Observation) -> Views:
        names = [self.wrist_camera] + ([self.overview_camera] if self.overview_camera else [])
        return Views(images={n: obs.images[n] for n in names if n in obs.images})

    def _plan(self, obs: Observation) -> str:
        if not self.plan_target:
            return self.instruction
        plan = self.vision.plan(self._views(obs), self.instruction)
        self.log.event(
            "plan", description=plan.description, feasible=plan.feasible, reason=plan.reason
        )
        if not plan.feasible:
            raise TaskFailed("infeasible", plan.reason)
        return plan.description

    def _camera(self, obs: Observation) -> CameraModel:
        image = obs.images[self.wrist_camera]
        height, width = image.shape[:2]
        # The pose the image was taken from: measured joints, sag included.
        pos, rot = self.kin.frame(obs.joint_pos, "camera", self.wrist_camera)
        intr = obs.extra.get("intrinsics", {}).get(self.wrist_camera)
        if intr:
            return CameraModel.from_intrinsics_dict(intr, pos, rot, width, height)
        K = self.robot_spec.camera(self.wrist_camera).intrinsics()
        sx, sy = (
            width / self.robot_spec.camera(self.wrist_camera).width,
            height / self.robot_spec.camera(self.wrist_camera).height,
        )
        K = np.diag([sx, sy, 1.0]) @ K
        return CameraModel(K=K, position=pos, rotation=rot, width=width, height=height)

    def _locate(self, obs: Observation, target: str, label: str) -> np.ndarray | None:
        camera = self._camera(obs)
        det = self.vision.locate(obs.images[self.wrist_camera], camera, target)
        point = self._to_table(det, camera)
        self._annotate(obs, det, label)
        self.log.event(
            "locate",
            view=label,
            found=det.found,
            confidence=round(det.confidence, 2),
            model=det.model,
            note=det.note,
            base_px=None if det.base is None else (round(det.base.u), round(det.base.v)),
            table_xy=None if point is None else point[:2].round(4),
        )
        if point is None:
            return None
        if np.any(point[:2] < self.ws_lower) or np.any(point[:2] > self.ws_upper):
            self.log.event("rejected", why="outside workspace", xy=point[:2].round(3))
            return None
        return point

    def _to_table(self, det: Detection, camera: CameraModel) -> np.ndarray | None:
        if not det.found or det.base is None:
            return None
        base = camera.intersect_z(det.base.u, det.base.v, self.table_z)
        if base is None:
            return None
        if det.rim is not None and self.glass_height > 0:
            # The rim pixel on the plane at the glass's height is a second,
            # independent estimate of the same axis. Use it only if it agrees.
            rim = camera.intersect_z(det.rim.u, det.rim.v, self.table_z + self.glass_height)
            if rim is not None and np.linalg.norm(rim[:2] - base[:2]) < 0.03:
                base = np.array([(base[0] + rim[0]) / 2, (base[1] + rim[1]) / 2, self.table_z])
        return base

    def _annotate(self, obs: Observation, det: Detection, label: str) -> None:
        if self.log.root is None:
            return
        img = obs.images[self.wrist_camera].copy()
        for px, colour in ((det.base, (255, 0, 0)), (det.rim, (0, 255, 0))):
            if px is not None:
                u, v = int(px.u), int(px.v)
                img[max(v - 6, 0) : v + 7, max(u - 1, 0) : u + 2] = colour
                img[max(v - 1, 0) : v + 2, max(u - 6, 0) : u + 7] = colour
        self.log.image(f"{label}_located", img)

    # ------------------------------------------------------------ geometry

    def _view_pose(self, center_xy: np.ndarray, view: ViewParams) -> np.ndarray:
        cx, cy = float(center_xy[0]), float(center_xy[1])
        yaw = math.atan2(cy, cx)
        radial = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        position = np.array([cx, cy, self.table_z + view.height]) - view.back * radial
        # Looking straight down with image-up pointing away from the base, then
        # tilted forward about the camera's own x axis.
        down = np.column_stack([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        t = math.radians(view.tilt_deg)
        tilt = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
        rotation = rot_z(yaw) @ down @ tilt
        seeds = [
            s
            for s in (
                self._q_cmd,
                self.robot_spec.home_joints(),
                np.array([0.0, 1.2, 1.2, 0.0, -0.8, 0.0]),
            )
            if s is not None
        ]
        result = self.kin.ik_multi_seed(
            "camera",
            self.wrist_camera,
            position,
            rotation,
            seeds,
            pos_tol=3e-3,
            rot_tol=math.radians(2.0),
        )
        if not result.ok:
            raise TaskFailed(
                "view_unreachable",
                f"camera view over {center_xy.round(3)} off by "
                f"{result.pos_err * 1000:.0f} mm / {math.degrees(result.rot_err):.1f} deg",
            )
        fingertips_z = self.kin.frame(result.q, "site", self.robot_spec.ee_site)[0][2]
        if fingertips_z < self.table_z + self.glass_height + self.min_finger_clearance:
            raise TaskFailed(
                "view_too_low", f"fingertips at {fingertips_z:.3f} m would reach the glass"
            )
        return result.q

    #: Extra IK starting points for low, flat grasps, where the home seed alone
    #: converges to a folded wrist that cannot reach.
    _GRASP_SEEDS = (
        np.array([0.0, 1.4, 1.0, -0.4, 0.0, 0.0]),
        np.array([0.0, 1.8, 1.6, -0.2, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.6, -0.4, 0.0, 0.0]),
    )

    def _grasp_waypoints(self, p: np.ndarray) -> list[np.ndarray]:
        g = self.grasp
        yaw = math.atan2(p[1], p[0])
        radial = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        home_rot = self.kin.frame(self.robot_spec.home_joints(), "site", self.robot_spec.ee_site)[1]
        rot = rot_z(yaw) @ rot_y(math.radians(g.pitch_deg - g.pitch_trim_deg)) @ home_rot
        at = np.array([p[0], p[1], self.table_z + g.height]) + g.past_centre * radial
        standoff = at - g.standoff * radial
        targets = {
            "above": standoff + [0, 0, g.clearance],
            "standoff": standoff,
            "at": at,
            "lifted": at + [0, 0, g.lift],
        }
        waypoints, seed = [], self.robot_spec.home_joints()
        for name, target in targets.items():
            result = self.kin.ik_multi_seed(
                "site",
                self.robot_spec.ee_site,
                target,
                rot,
                [seed, self._q_cmd, *self._GRASP_SEEDS],
                pos_tol=2e-3,
                rot_tol=math.radians(1.0),
            )
            if not result.ok:
                raise TaskFailed(
                    "unreachable",
                    f"{name} waypoint {target.round(3)} off by {result.pos_err * 1000:.0f} mm",
                )
            waypoints.append(result.q)
            seed = result.q
        self.log.event("grasp_plan", glass_xy=p[:2].round(4), yaw_deg=round(math.degrees(yaw), 1))
        return waypoints

    # ------------------------------------------------------------ motion

    def _move(
        self,
        label: str,
        q: np.ndarray,
        gripper: float,
        obs: Observation,
        speed: float | None = None,
        settle: float | None = None,
    ) -> MoveGen:
        if self.gate is not None and not self.gate(label, self._describe(q, gripper)):
            raise TaskFailed("operator_declined", label)
        chunk = self._trajectory(
            q,
            gripper,
            obs,
            speed or self.max_joint_speed,
            self.settle_seconds if settle is None else settle,
        )
        self.log.event("move", phase=label, seconds=round(len(chunk) / self.hz, 1), gripper=gripper)
        obs = yield chunk
        # Wait for the arm to arrive before looking or grasping from here. On
        # gem13 a raise lagged its command by >0.25 rad (fleet gains, no Kronos
        # mass in gravity comp); a camera pose taken mid-lag is a wrong pose.
        # With sag compensation, the command is walked off by the measured
        # shortfall (integral action, capped) until the arm is where the plan
        # says; the offset carries into the next move, whose sag is similar.
        waited = 0.0
        while self._tracking_error(obs) > self.arrive_tol and waited < self.arrive_timeout:
            if self.sag_compensation:
                shortfall = self._q_cmd - obs.joint_pos
                self._bias = np.clip(self._bias + 0.5 * shortfall, -self.max_bias, self.max_bias)
            obs = yield self._hold(obs, seconds=0.2)
            waited += 0.2
        if waited:
            planned = self.kin.frame(self._q_cmd, "site", self.robot_spec.ee_site)[0]
            actual = self.kin.frame(obs.joint_pos, "site", self.robot_spec.ee_site)[0]
            self.log.event(
                "arrived", phase=label, waited_s=round(waited, 1),
                error_deg=np.degrees(obs.joint_pos - self._q_cmd).round(1),
                bias_deg=np.degrees(self._bias).round(1),
                tip_error_mm=(1000 * (actual - planned)).round(0),
            )
        return obs

    def _tracking_error(self, obs: Observation) -> float:
        return float(np.max(np.abs(obs.joint_pos - self._q_cmd)))

    def _describe(self, q: np.ndarray, gripper: float) -> str:
        dq = np.abs(q - self._q_cmd)
        return (
            f"largest joint move {math.degrees(dq.max()):.0f} deg (joint{int(dq.argmax()) + 1}), "
            f"gripper {'close' if gripper > 0.5 else 'open'}"
        )

    def _trajectory(
        self, q: np.ndarray, gripper: float, obs: Observation, speed: float, settle: float
    ) -> ActionChunk:
        q0, g0 = self._q_cmd, self._g_cmd
        q1 = self.robot_spec.clip_joints(q)
        duration = max(
            float(np.max(np.abs(q1 - q0))) / speed, abs(gripper - g0) * self.gripper_seconds, 0.2
        )
        n = max(1, math.ceil(duration * self.hz))
        now = self.clock.now()
        actions = []
        for k in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * k / n)  # ease in and out
            actions.append(
                self._action(q0 + s * (q1 - q0), g0 + (k / n) * (gripper - g0), now, obs)
            )
        actions += [self._action(q1, gripper, now, obs) for _ in range(math.ceil(settle * self.hz))]
        self._q_cmd, self._g_cmd = q1, float(gripper)
        return ActionChunk(actions=actions, obs_timestamp=obs.timestamp, horizon_hz=self.hz)

    def _action(self, q: np.ndarray, gripper: float, now: float, obs: Observation) -> Action:
        return Action(
            mode=self.robot_spec.default_control_mode,
            values=q + self._bias,
            gripper=float(np.clip(gripper, 0.0, 1.0)),
            timestamp=now,
            obs_timestamp=obs.timestamp,
            source=ActionOrigin.SCRIPTED,
        )

    def _hold(self, obs: Observation, seconds: float = 0.0) -> ActionChunk:
        q = obs.joint_pos if self._q_cmd is None else self._q_cmd
        n = max(1, math.ceil(seconds * self.hz))
        return ActionChunk(
            actions=[self._action(q, self._g_cmd, self.clock.now(), obs) for _ in range(n)],
            obs_timestamp=obs.timestamp,
            horizon_hz=self.hz,
        )
