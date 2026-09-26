"""Contact detection from the arm's own motors: expected torque vs measured.

gem13's joint torques swing 0-11 Nm with pose on J2/J3 (gravity), so a fixed
torque limit is useless. Instead:

1. ``TorqueModel`` predicts each joint's torque from the robot description:
   MuJoCo inverse dynamics at the measured joint angles, velocities and
   accelerations (gravity + inertia, calibrated gem13 model with the Kronos),
   then a per-joint linear correction fitted on free-motion runs of the real
   rig (sign, scale, friction, what the rig's gravity compensation leaves).
2. ``ContactDetector`` watches the residual, measured minus predicted, per
   joint, normalised by its spread on free motion, and trips on
   - a sustained residual (a press or a push): ``|z| > sustained_z`` for
     ``consecutive`` steps;
   - a sudden jump (a hit): the residual minus its recent baseline, same test;
   - the arm lagging its command by more than it ever did on free motion
     (only when commands are given, i.e. offline replays).
   Phases where contact is expected (closing on the cup, putting it down)
   use looser limits.
3. ``ContactMonitor`` runs it in the control loop. Opt-in, and log-only by
   default (``mode: log``): it records what it would have done
   (contact_checks.jsonl) and never stops the arm until ``mode: enforce``.

Fit a model: ``scripts/fit_torque_model.py``; replay recorded runs through the
detector: ``scripts/eval_contact_detector.py``.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from teleop_sim.control.safety.review import ClaudeReview, Reviewer
from teleop_sim.core.protocols import Robot, SafetyMonitor, SafetyVerdict
from teleop_sim.core.registry import SAFETY, build, register
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Observation, Outcome

#: Phases of PickLiftPlacePolicy where the gripper or the cup is meant to touch
#: something: closing on the cup, the cup meeting the table, letting go.
CONTACT_PHASES = frozenset({"close", "put_down", "lower", "release", "open"})

#: Plain names for the YAM's joints, for Claude's review.
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist pitch", "wrist yaw", "wrist roll")

#: Features per joint: gravity, inertia, velocity, Coulomb friction, offset.
FEATURES = ("gravity", "inertia", "velocity", "friction", "offset")
FRICTION_VEL = 0.02  # rad/s: tanh width of the friction sign
GAP_S = 0.3  # s between observations that means the loop was paused


class TorqueModel:
    """Expected joint torque: MuJoCo inverse dynamics plus a fitted correction."""

    def __init__(self, spec: RobotSpec, coef: np.ndarray | None = None) -> None:
        import mujoco

        from teleop_sim.robots.kinematics import Kinematics

        self._mj = mujoco
        self.kin = Kinematics(spec)
        # The sim model enables MuJoCo gravity compensation to mimic the rig's
        # controller, which hides the arm's weight from inverse dynamics. The
        # motors report the torque they actually apply, weight included, so
        # this private copy of the model sees full gravity.
        self.kin.model.body_gravcomp[:] = 0.0
        self.dof = len(self.kin._dadr)
        self.coef = None if coef is None else np.asarray(coef, dtype=float)

    def dynamics(self, q, qd, qdd) -> tuple[np.ndarray, np.ndarray]:
        """(gravity, inertial) joint torques from the model, Nm."""
        m, d, qa, da = self.kin.model, self.kin.data, self.kin._qadr, self.kin._dadr
        d.qpos[:] = m.qpos0
        d.qpos[qa] = q
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        self._mj.mj_inverse(m, d)
        grav = d.qfrc_inverse[da].copy()
        d.qvel[da] = qd
        d.qacc[da] = qdd
        self._mj.mj_inverse(m, d)
        return grav, d.qfrc_inverse[da].copy() - grav

    def features(self, q, qd, qdd) -> np.ndarray:
        """(dof, len(FEATURES)) feature matrix for one step."""
        grav, inert = self.dynamics(q, qd, qdd)
        qd = np.asarray(qd, dtype=float)
        return np.stack([grav, inert, qd, np.tanh(qd / FRICTION_VEL), np.ones(self.dof)], axis=1)

    def predict(self, q, qd, qdd) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("TorqueModel has no fitted coefficients; see fit_torque_model.py")
        return np.einsum("jf,jf->j", self.features(q, qd, qdd), self.coef)

    def fit(self, feats: np.ndarray, eff: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
        """Least squares per joint. feats (N, dof, F), eff (N, dof) -> residuals (N, dof)."""
        n, dof, nf = feats.shape
        coef = np.zeros((dof, nf))
        for j in range(dof):
            a, b = feats[:, j, :], eff[:, j]
            coef[j] = np.linalg.solve(a.T @ a + ridge * np.eye(nf), a.T @ b)
        self.coef = coef
        return eff - np.einsum("njf,jf->nj", feats, coef)


def accelerations(qd: np.ndarray, t: np.ndarray, window: int = 5) -> np.ndarray:
    """Smoothed finite-difference joint accelerations for a recorded run."""
    qdd = np.gradient(qd, t, axis=0)
    k = np.ones(window) / window
    return np.stack([np.convolve(qdd[:, j], k, mode="same") for j in range(qd.shape[1])], 1)


@dataclass
class DetectorParams:
    """Everything the detector needs, fitted from free-motion runs (JSON-able)."""

    coef: list[list[float]]  # (dof, len(FEATURES))
    sigma: list[float]  # residual spread per joint on free motion (robust)
    sigma_jump: list[float]  # spread of residual minus its recent baseline
    track_max_deg: list[float] | None = None  # worst |command - measured| on free motion
    sustained_z: float = 6.0
    jump_z: float = 6.0
    contact_phase_scale: float = 2.5  # looser limits where contact is expected
    consecutive: int = 3  # ~100 ms at 28 Hz
    baseline_steps: int = 28  # ~1 s of residual history for the jump test
    track_margin_deg: float = 3.0
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> DetectorParams:
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class ContactTrip:
    rule: str  # "sustained" | "jump" | "tracking"
    joint: int  # 0-based
    value: float  # z-score, or degrees for tracking
    detail: str


class ContactDetector:
    """Step-by-step contact test on residual torques. No I/O; replay it offline."""

    def __init__(self, params: DetectorParams) -> None:
        self.p = params
        self.sigma = np.maximum(np.asarray(params.sigma), 1e-3)
        self.sigma_jump = np.maximum(np.asarray(params.sigma_jump), 1e-3)
        self.track_max = None if params.track_max_deg is None else np.asarray(params.track_max_deg)
        self.reset()

    def reset(self) -> None:
        self._hist: deque[np.ndarray] = deque(maxlen=self.p.baseline_steps)
        self._streak: dict[tuple[str, int], int] = {}
        self.last_z: np.ndarray | None = None
        self.last_jump_z: np.ndarray | None = None

    def update(
        self,
        residual: np.ndarray,
        phase: str = "",
        cmd_q: np.ndarray | None = None,
        q: np.ndarray | None = None,
    ) -> ContactTrip | None:
        scale = self.p.contact_phase_scale if phase in CONTACT_PHASES else 1.0
        z = residual / self.sigma
        if len(self._hist) >= self.p.baseline_steps // 2:
            base = np.median(np.array(self._hist)[: -3 or None], axis=0)
            jz = (residual - base) / self.sigma_jump
        else:
            jz = np.zeros_like(residual)
        self._hist.append(np.asarray(residual, dtype=float))
        self.last_z, self.last_jump_z = z, jz
        checks = [
            ("sustained", np.abs(z) > self.p.sustained_z * scale, z),
            ("jump", np.abs(jz) > self.p.jump_z * scale, jz),
        ]
        if self.track_max is not None and cmd_q is not None and q is not None:
            lag = np.degrees(np.abs(np.asarray(cmd_q) - np.asarray(q)))
            over = lag > (self.track_max + self.p.track_margin_deg) * scale
            checks.append(("tracking", over, lag))
        trip = None
        for rule, over, val in checks:
            for j in range(len(over)):
                key = (rule, j)
                self._streak[key] = self._streak.get(key, 0) + 1 if over[j] else 0
                if trip is None and self._streak[key] >= self.p.consecutive:
                    unit = "deg" if rule == "tracking" else "sigma"
                    trip = ContactTrip(
                        rule,
                        j,
                        float(val[j]),
                        f"{rule} on J{j + 1}: {val[j]:+.1f} {unit} for "
                        f"{self._streak[key]} steps" + (f" (phase {phase})" if phase else ""),
                    )
        return trip


def gripper_state(obs: Observation) -> str:
    """What the gripper's own sensors say it holds, in words, for Claude's review.

    From the camera, closed jaws pressing on a cup can look like jaws holding it
    (2026-09-25: Claude called exactly that collision a false alarm). The jaw
    position and the servo current settle it: on gem13's Kronos a held cup
    reads ~300 with the jaws part-way, empty jaws read ~0 fully closed
    (docs/experiments/2026-09-25-gripper-current).
    """
    motors = (obs.extra or {}).get("motors") or {}
    jaw = float(obs.gripper)  # 0 open .. 1 closed
    cur = motors.get("gripper_eff")
    cur = None if cur is None else float(np.asarray(cur).ravel()[0])
    reading = f"jaw {jaw:.2f} closed" + (f", gripper current {cur:.0f}" if cur is not None else "")
    if jaw >= 0.9:
        return (
            f"The gripper's sensors ({reading}) say its jaws are shut with NOTHING between "
            "them: it is not holding anything, whatever the wrist image seems to show "
            "between the fingers."
        )
    if jaw > 0.05 and (cur is None or cur >= 150):
        return f"The gripper's sensors ({reading}) say it is holding an object."
    if jaw <= 0.05:
        return f"The gripper's sensors ({reading}) say its jaws are open."
    return f"The gripper's sensors read {reading}."


DEFAULT_PARAMS = "data/torque_models/gem13.json"
#: The opt-in settings --contact-monitor applies (run_pick.py, collect_real_picks.py).
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "safety" / "contact_monitor.yaml"


def with_contact_monitor(
    safety: dict[str, Any] | None,
    mode: str = "log",
    config: str | Path = DEFAULT_CONFIG,
    params: str | None = None,
) -> dict[str, Any]:
    """A safety config that wraps ``safety`` (the run's own monitor) in contact_monitor."""
    import yaml

    wrapped = dict(yaml.safe_load(Path(config).read_text()))
    wrapped["mode"] = mode
    if params is not None:
        wrapped["params"] = params
    wrapped["inner"] = dict(safety or {"type": "sim_watchdog"})
    return wrapped


@register(SAFETY, "contact_monitor")
class ContactMonitor(SafetyMonitor):
    """Motor-torque contact detection in the control loop. Log-only by default.

    In ``enforce`` mode a contact stops the arm, backs it off to the command sent
    ``retreat_s`` ago (the robot must offer ``command_before``, as the
    robots_realtime bridge does; otherwise it just holds), and hands the decision
    to Claude (``review``): resume if it was not a collision, wait, or abort.
    Without a ``review`` section a contact aborts at once.

    Needs ``obs.extra["motors"]`` with ``joint_eff`` and ``joint_vel`` (the
    robots_realtime bridge provides them) and fitted parameters from
    scripts/fit_torque_model.py.
    """

    def __init__(
        self,
        spec: RobotSpec | None = None,
        params: str = DEFAULT_PARAMS,
        mode: str = "log",
        inner: dict[str, Any] | None = None,
        log_dir: str | None = None,
        retreat_s: float = 1.0,
        retreat_m: float = 0.03,
        glide_speed: float = 0.3,
        cameras: tuple[str, ...] = ("wrist", "overview"),
        review: dict[str, Any] | None = None,
        reviewer: Reviewer | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if mode not in ("log", "enforce"):
            raise ValueError(f"contact_monitor: mode must be 'log' or 'enforce', got {mode!r}")
        if spec is None:
            raise ValueError("contact_monitor needs the robot spec for its torque model")
        self.inner = build(SAFETY, dict(inner), spec=spec) if inner is not None else None
        path = Path(params)
        if not path.exists():
            raise FileNotFoundError(
                f"contact_monitor: no fitted parameters at {path}. Fit them from free-motion "
                "runs: python scripts/fit_torque_model.py runs/<batch>/ep01 ..."
            )
        self.params = DetectorParams.load(path)
        self.model = TorqueModel(spec, self.params.coef)
        self.ee_site = spec.ee_site
        self.detector = ContactDetector(self.params)
        self.mode = mode
        self.retreat_s = float(retreat_s)  # fallback when the robot has no command list
        self.retreat_m = float(retreat_m)  # back off this far along the path just travelled
        self.glide_speed = float(glide_speed)  # rad/s peak, gliding back before resuming
        self.cameras = list(cameras)
        self.review = ClaudeReview(review, reviewer, clock=clock, sleep=sleep, name="contact")
        self.sleep = sleep
        self.robot: Robot | None = None
        self.log_dir = Path(log_dir) if log_dir else None
        self.phase_of: Callable[[], str] = lambda: ""
        self._prev: tuple[float, np.ndarray] | None = None
        self.trips: list[ContactTrip] = []

    def bind(
        self,
        run_log=None,
        phase_of: Callable[[], str] | None = None,
        robot: Robot | None = None,
        context: str | None = None,
        **kw: Any,
    ) -> None:
        if run_log is not None:
            self.log_dir = run_log.root
        if phase_of is not None:
            self.phase_of = phase_of
        if robot is not None:
            self.robot = robot
        self.review.bind(run_log=run_log, robot=robot, context=context)
        if self.inner is not None and hasattr(self.inner, "bind"):
            self.inner.bind(run_log=run_log, phase_of=phase_of, robot=robot, context=context, **kw)

    def reset(self) -> None:
        if self.inner is not None:
            self.inner.reset()
        self.detector.reset()
        self._prev, self.trips = None, []
        if self.mode == "enforce":
            self.review.reset()

    def check(self, obs: Observation, step: int) -> SafetyVerdict | None:
        if self.inner is not None:
            verdict = self.inner.check(obs, step)
            if verdict is not None:
                return verdict
        if self.mode == "enforce":
            self.review.note_baseline(obs, self.cameras)
        motors = (obs.extra or {}).get("motors") or {}
        if "joint_eff" not in motors or "joint_vel" not in motors:
            return None  # sim, or a rig that does not publish torques: nothing to judge
        q = np.asarray(obs.joint_pos, dtype=float)
        qd = np.asarray(motors["joint_vel"], dtype=float)[: len(q)]
        eff = np.asarray(motors["joint_eff"], dtype=float)[: len(q)]
        qdd = np.zeros_like(qd)
        if self._prev is not None and obs.timestamp - self._prev[0] > GAP_S:
            self.detector.reset()  # after a pause: fresh baseline, no stale acceleration
        elif self._prev is not None and obs.timestamp > self._prev[0]:
            qdd = (qd - self._prev[1]) / (obs.timestamp - self._prev[0])
        self._prev = (obs.timestamp, qd)
        residual = eff - self.model.predict(q, qd, qdd)
        phase = str(self.phase_of())
        trip = self.detector.update(residual, phase)
        if trip is None:
            return None
        self.trips.append(trip)
        self._log(
            {
                "event": "contact",
                "mode": self.mode,
                "phase": phase,
                **asdict(trip),
                "residual": np.round(residual, 3).tolist(),
            }
        )
        print(f"[safety] contact ({self.mode}): {trip.detail}", flush=True)
        self.detector.reset()  # count the next one independently
        if self.mode == "log":
            return None
        verdict = SafetyVerdict(
            Outcome.WATCHDOG, f"contact:{trip.rule}", trip.detail, safe_stop=True
        )
        stopped_at = self._latest_command()
        backed_off = self._retreat()
        if not self.review.enabled:
            return verdict
        result = self._review(obs, trip, phase, backed_off, verdict)
        if result is None and backed_off and stopped_at is not None:
            self._glide_to(stopped_at)  # the sweep's next command continues from there
        return result

    def _latest_command(self):
        cmds = getattr(self.robot, "recent_commands", None)
        if cmds is not None:
            recent = cmds()
            return recent[-1][1] if recent else None
        back = getattr(self.robot, "command_before", None)
        return back(0.0) if back is not None else None

    def _backoff_target(self):
        """The most recent command whose tip is ``retreat_m`` behind the latest one,
        on the path just travelled (so clear of whatever the arm met)."""
        cmds = getattr(self.robot, "recent_commands", None)
        if cmds is None:
            back = getattr(self.robot, "command_before", None)
            return back(self.retreat_s) if back is not None else None
        recent = [a for _, a in cmds()]
        if not recent:
            return None
        site = self.model.kin
        ee = self.ee_site
        p_last = site.frame(np.asarray(recent[-1].values), "site", ee)[0]
        for a in reversed(recent):
            if (
                np.linalg.norm(site.frame(np.asarray(a.values), "site", ee)[0] - p_last)
                >= self.retreat_m
            ):
                return a
        return recent[0]

    def _retreat(self) -> bool:
        """Back off along the path just travelled, then hold there. False if the
        robot keeps no command history (it then holds where it stopped)."""
        target = self._backoff_target() if self.robot is not None else None
        if target is None:
            return False
        print(f"[safety] backing off ~{self.retreat_m * 100:.0f} cm along the path", flush=True)
        self._log(
            {
                "event": "retreat",
                "metres": self.retreat_m,
                "target": np.round(target.values, 4).tolist(),
            }
        )
        trace = []
        for _ in range(8):  # ~0.8 s of the same target; then silence holds it
            self.robot.send_action(target)
            self.sleep(0.1)
            try:  # the loop records nothing while paused: keep our own trace
                o = self.robot.get_observation()
                trace.append({"t": time.time(), "q": np.round(o.joint_pos, 4).tolist()})
            except Exception:  # a trace is nice to have, never a reason to fail
                pass
        self._log({"event": "retreat_trace", "trace": trace})
        return True

    def _review(
        self,
        obs: Observation,
        trip: ContactTrip,
        phase: str,
        backed_off: bool,
        verdict: SafetyVerdict,
    ) -> SafetyVerdict | None:
        self.review.run_log.event(
            "contact_pause",
            rule=trip.rule,
            joint=trip.joint + 1,
            value=round(trip.value, 1),
            phase=phase,
        )
        at_contact = {
            f"{c} (at the moment of contact)": obs.images[c]
            for c in self.cameras
            if c in obs.images
        }
        now = self.robot.get_observation() if self.robot is not None else obs
        grip = gripper_state(obs)
        unit = (
            "degrees behind its command"
            if trip.rule == "tracking"
            else ("times its normal variation")
        )

        def describe(held_s: float) -> str:
            return (
                "The alarm came from the arm's own motors, not a camera. Joint "
                f"J{trip.joint + 1} ({JOINT_NAMES[trip.joint]}) "
                f"read {abs(trip.value):.0f} {unit} ({trip.rule}) during the '{phase}' "
                "phase, as if the arm hit or pressed on something. "
                + (
                    "The arm stopped and backed off about "
                    f"{self.retreat_m * 100:.0f} cm along its path. "
                    if backed_off
                    else "The arm stopped. "
                )
                + (f"{grip} " if grip else "")
                + f"It has been holding for {held_s:.0f}s. Decide whether the arm touched "
                "something it should not have (a person, an object in its way, the cup "
                "knocked over or pushed) or whether this was a false alarm, such as the "
                "expected contact of the gripper with the cup it is picking up."
            )

        why = self.review.decide(at_contact, now, self.cameras, describe, log=self._log)
        if why is not None:
            return SafetyVerdict(verdict.outcome, verdict.tag, f"{why}; {trip.detail}", True)
        self.detector.reset()
        self._prev = None
        self._log({"event": "resume"})
        return None

    def _glide_to(self, target) -> None:
        """Move smoothly (minimum jerk, peak ``glide_speed``) from where the arm is
        back to where it stopped, so resuming does not snap it back at the
        bridge's per-command step limit."""
        import dataclasses

        from teleop_sim.robots.trajectory import min_jerk

        q0 = np.asarray(self.robot.get_observation().joint_pos, dtype=float)
        q1 = np.asarray(target.values, dtype=float)
        seconds = max(1.875 * float(np.max(np.abs(q1 - q0))) / self.glide_speed, 0.5)
        n = max(1, int(seconds * 30))
        self._log({"event": "glide_back", "seconds": round(seconds, 2)})
        print(f"[safety] gliding back to where it stopped ({seconds:.1f}s)", flush=True)
        for k in range(1, n + 1):
            q = q0 + min_jerk(k / n) * (q1 - q0)
            self.robot.send_action(dataclasses.replace(target, values=q, timestamp=time.time()))
            self.sleep(1 / 30)
        self._prev = None
        self.detector.reset()

    def on_trip(self, robot: Robot, verdict: SafetyVerdict) -> None:
        if verdict.safe_stop:
            robot.safe_stop()
        print(f"[safety] STOP {verdict.tag}: {verdict.detail}", flush=True)
        self._log({"event": "stop", "tag": verdict.tag, "detail": verdict.detail})

    def _log(self, line: dict[str, Any]) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / "contact_checks.jsonl", "a") as fh:
            fh.write(json.dumps({"t": time.time(), **line}) + "\n")


@dataclass
class RunArrays:
    """One recorded episode's telemetry, as arrays (steps with motor readings only)."""

    name: str
    t: np.ndarray  # (N,) wall seconds from the start
    phase: list[str]
    q: np.ndarray  # (N, dof) measured
    cmd_q: np.ndarray  # (N, dof) commanded
    qd: np.ndarray  # (N, dof) from the motors
    eff: np.ndarray  # (N, dof) joint torque from the motors

    @classmethod
    def load(cls, episode: str | Path) -> RunArrays:
        rows = [json.loads(line) for line in (Path(episode) / "telemetry.jsonl").open()]
        rows = [r for r in rows if "joint_eff" in r.get("motors", {})]
        if not rows:
            raise ValueError(
                f"{episode}: no joint torques in telemetry (recorded before 2026-09-25?)"
            )
        t = np.array([r["wall"] for r in rows])
        return cls(
            name=str(episode),
            t=t - t[0],
            phase=[r["phase"] for r in rows],
            q=np.array([r["joint_pos"] for r in rows]),
            cmd_q=np.array([r["cmd_joint_pos"] for r in rows]),
            qd=np.array([r["motors"]["joint_vel"] for r in rows]),
            eff=np.array([r["motors"]["joint_eff"] for r in rows]),
        )

    def free(self, after_gap_s: float = 1.5) -> np.ndarray:
        """Mask of steps outside the phases where contact is expected, and not just
        after a gap in the log (a safety pause freezes the loop; the first steps
        after it carry wrong accelerations and a stale baseline)."""
        mask = np.array([p not in CONTACT_PHASES for p in self.phase])
        gaps = np.flatnonzero(np.diff(self.t) > GAP_S) + 1
        for g in gaps:
            mask[g : g + int(np.searchsorted(self.t[g:], self.t[g] + after_gap_s))] = False
        return mask


def run_features(model: TorqueModel, run: RunArrays) -> np.ndarray:
    qdd = accelerations(run.qd, run.t)
    return np.stack([model.features(run.q[i], run.qd[i], qdd[i]) for i in range(len(run.t))])


def jump_signal(residual: np.ndarray, baseline_steps: int) -> np.ndarray:
    """Residual minus the median of its recent past (as ContactDetector computes it)."""
    out = np.zeros_like(residual)
    for i in range(len(residual)):
        lo = max(0, i - baseline_steps)
        past = residual[lo : max(lo, i - 3)]
        if len(past) >= baseline_steps // 2 - 3:
            out[i] = residual[i] - np.median(past, axis=0)
    return out


def sustained_max(x: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Per column: the largest value |x| held for k consecutive masked steps."""
    a = np.abs(x)
    best = np.zeros(a.shape[1])
    for i in range(len(a) - k + 1):
        if mask[i : i + k].all():
            best = np.maximum(best, a[i : i + k].min(axis=0))
    return best


def robust_sigma(x: np.ndarray) -> np.ndarray:
    """Per-column spread: 1.4826 x median absolute deviation (a std that ignores outliers)."""
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)


def fit_params(
    runs: list[RunArrays],
    spec: RobotSpec,
    baseline_steps: int = 28,
    margin: float = 1.25,
    floor_nm: float = 0.02,
):
    """Fit the torque model and the detector's spreads on free-motion steps of ``runs``.

    Returns (model, params, residuals per run).
    """
    model = TorqueModel(spec)
    feats = [run_features(model, r) for r in runs]
    free = [r.free() for r in runs]
    model.fit(
        np.concatenate([f[m] for f, m in zip(feats, free, strict=True)]),
        np.concatenate([r.eff[m] for r, m in zip(runs, free, strict=True)]),
    )
    res = [r.eff - np.einsum("njf,jf->nj", f, model.coef) for r, f in zip(runs, feats, strict=True)]
    free_res = np.concatenate([x[m] for x, m in zip(res, free, strict=True)])
    jump_runs = [jump_signal(x, baseline_steps) for x in res]
    probe = DetectorParams(coef=[], sigma=[], sigma_jump=[])
    # Each limit sits above the largest 3-step-sustained value free motion
    # produced (with a margin), and never below the robust noise estimate or a
    # floor: a nearly idle joint's noise is tiny, and its normal start-up
    # torque would otherwise look like a hit.
    sus = np.max(
        [sustained_max(x, m, probe.consecutive) for x, m in zip(res, free, strict=True)], 0
    )
    jmp = np.max(
        [sustained_max(x, m, probe.consecutive) for x, m in zip(jump_runs, free, strict=True)], 0
    )
    sigma = np.maximum.reduce(
        [robust_sigma(free_res), sus * margin / probe.sustained_z, np.full_like(sus, floor_nm)]
    )
    jumps = np.concatenate([x[m] for x, m in zip(jump_runs, free, strict=True)])
    sigma_jump = np.maximum.reduce(
        [robust_sigma(jumps), jmp * margin / probe.jump_z, np.full_like(jmp, floor_nm)]
    )
    lag = np.concatenate(
        [np.degrees(np.abs(r.cmd_q - r.q))[m] for r, m in zip(runs, free, strict=True)]
    )
    params = DetectorParams(
        coef=model.coef.round(6).tolist(),
        sigma=sigma.round(4).tolist(),
        sigma_jump=sigma_jump.round(4).tolist(),
        track_max_deg=lag.max(axis=0).round(2).tolist(),
        baseline_steps=baseline_steps,
        meta={
            "features": list(FEATURES),
            "runs": [r.name for r in runs],
            "free_steps": int(free_res.shape[0]),
            "raw_torque_std": np.concatenate([r.eff[m] for r, m in zip(runs, free, strict=True)])
            .std(axis=0)
            .round(3)
            .tolist(),
            "residual_std": free_res.std(axis=0).round(3).tolist(),
        },
    )
    return model, params, res
