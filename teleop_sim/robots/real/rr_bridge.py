"""One arm of a running robots_realtime session, as a Robot.

This does not drive CAN. It talks to an ``rr-session`` that the operator has
launched on the rig (on gem13: ``scripts/launch_gem13.sh``), which already
owns the motors, gravity compensation, the Kronos gripper logic and its own
safety layers (SafeMotorChainRobot: joint limits, velocity clamp, the
max-command-error guard, the handoff ramp). Commands go through all of that.

Protocol facts this relies on, read from robots_realtime on gem13:

* Joint commands: publish ``{"joint_pos": [j1..j6, grip]}`` on the arm's
  RobotNode ``cmd_topic``. On gem13_inference.yaml that is
  ``hud/teleop/<arm>/joint_pos`` -- the topic the HUD's WebRTC teleop panel
  also writes, but only while a browser operator is actively driving. Do not
  drive from the HUD panel and from here at once.
* ``grip`` is normalised with 1.0 = OPEN (i2rt maps [0, 1] onto
  ``gripper_limits: [closed, open]``). teleop_sim uses 0 = open, so it is
  inverted here and nowhere else.
* A RobotNode stops applying a command once its ts is 0.5 s old and holds the
  last pose; when commands resume it ramps in from the measured pose. Silence
  is therefore safe, and this driver relies on it: when it cannot see fresh
  state it sends nothing.
* State arrives on ``<arm>/joint_state`` as the driver's get_observations():
  ``joint_pos`` (6, radians), ``gripper_pos`` ([1], normalised, 1 = open).
* Camera frames arrive on ``<cam>/rgb`` as ``{"images": {"rgb": HxWx3},
  "timestamp", "intrinsics": {fx, fy, cx, cy, ...}, "depth_data": HxW
  uint16}`` when the CameraNode has no ``publish_resize``.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np

from teleop_sim.core.clock import Clock
from teleop_sim.core.protocols import Robot, UnsupportedControlMode
from teleop_sim.core.registry import ROBOTS, register
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Action, Observation
from teleop_sim.robots.real.rr_wire import (
    DEFAULT_PUB_PORT,
    DEFAULT_SUB_PORT,
    BusPublisher,
    BusSubscriber,
)


class RobotLinkLost(RuntimeError):
    """No fresh state from the rig. The rig holds position on its own."""


@register(ROBOTS, "robots_realtime")
class RobotsRealtimeRobot(Robot):
    def __init__(
        self,
        spec: RobotSpec,
        clock: Clock,
        host: str = "pantheon-gem13.local",
        pub_port: int = DEFAULT_PUB_PORT,
        sub_port: int = DEFAULT_SUB_PORT,
        arm: str = "yam_right",
        cmd_topic: str | None = None,
        state_topic: str | None = None,
        cameras: dict[str, str] | None = None,
        camera_resolution: list[int] = (640, 480),
        joint_limits: list[list[float]] | None = None,
        max_step_rad: float = 0.25,
        state_timeout_s: float = 0.3,
        link_grace_s: float = 3.0,
        connect_timeout_s: float = 15.0,
        rgb_order: str = "rgb",
        camera_port: int | None = None,
    ) -> None:
        self.spec = spec
        self.clock = clock
        self.host = host
        self.pub_port, self.sub_port = int(pub_port), int(sub_port)
        self.arm = arm
        self.cmd_topic = cmd_topic or f"hud/teleop/{arm}/joint_pos"
        self.state_topic = state_topic or f"{arm}/joint_state"
        # spec camera name -> bus topic, e.g. {"wrist": "right_wrist_cam/rgb"}
        self.camera_topics = dict(cameras or {})
        self.camera_resolution = (int(camera_resolution[0]), int(camera_resolution[1]))
        self.joint_limits = None if joint_limits is None else np.asarray(joint_limits, float)[:6]
        self.max_step_rad = float(max_step_rad)
        self.state_timeout_s = float(state_timeout_s)
        self.link_grace_s = float(link_grace_s)
        self.connect_timeout_s = float(connect_timeout_s)
        if rgb_order not in ("rgb", "bgr"):
            raise ValueError("rgb_order must be 'rgb' or 'bgr'")
        self.rgb_order = rgb_order
        # Cameras from a separate port (camera_relay.py on the rig): over a slow
        # link the full-rate camera streams delay everything sharing their socket.
        self.camera_port = None if camera_port is None else int(camera_port)
        self._cam_sub: BusSubscriber | None = None

        self._pub: BusPublisher | None = None
        self._sub: BusSubscriber | None = None
        self._skew = 0.0  # local time.time() minus rig time.time()
        self._halted = False
        self._clipped_steps = 0
        self._sent: deque[tuple[float, Action]] = deque(maxlen=300)  # ~10 s of commands
        self._kin = None

    # ------------------------------------------------------------ link

    @property
    def is_connected(self) -> bool:
        return self._sub is not None

    def connect(self) -> None:
        if self._sub is not None:
            return
        cameras = list(self.camera_topics.values())
        if self.camera_port is None:
            self._sub = BusSubscriber(self.host, [self.state_topic, *cameras], port=self.sub_port)
        else:
            self._sub = BusSubscriber(self.host, [self.state_topic], port=self.sub_port)
            self._cam_sub = BusSubscriber(self.host, cameras, port=self.camera_port)
        self._pub = BusPublisher(self.host, port=self.pub_port)
        self._halted = False

    def disconnect(self) -> None:
        for sock in (self._pub, self._sub, self._cam_sub):
            if sock is not None:
                sock.close()
        self._pub = self._sub = self._cam_sub = None

    def _rec(self, topic: str):
        if self._cam_sub is not None and topic in self.camera_topics.values():
            return self._cam_sub.get(topic)
        return self._sub.get(topic)

    def _wait_for(self, topics: list[str], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            missing = [
                t
                for t in topics
                if self._rec(t) is None or self._rec(t).age() > max(self.state_timeout_s, 1.0)
            ]
            if not missing:
                return
            time.sleep(0.05)
        seen = self._sub.topics() + (self._cam_sub.topics() if self._cam_sub else [])
        raise RobotLinkLost(
            f"no fresh data from {self.host} on {missing} within {timeout:.0f}s "
            f"(topics seen: {seen or 'none'}). Is rr-session running on the rig, and is "
            f"tcp {self.pub_port}/{self.sub_port} reachable?"
        )

    def _measure_skew(self) -> None:
        samples = []
        for _ in range(10):
            rec = self._sub.get(self.state_topic)
            samples.append(rec.local_time - float(rec.envelope["ts"]))
            time.sleep(0.02)
        self._skew = float(np.median(samples))
        if abs(self._skew) > 0.25:
            print(
                f"[rr] clock skew to {self.host} is {self._skew:+.3f}s; commands are "
                "stamped in the rig's clock to compensate"
            )

    def describe_link(self) -> str:
        rec = self._sub.get(self.state_topic) if self._sub else None
        lines = [
            f"host {self.host}  pub:{self.pub_port} sub:{self.sub_port}",
            f"command topic {self.cmd_topic}",
            f"state topic {self.state_topic}",
            f"clock skew {self._skew * 1000:+.0f} ms (local - rig)",
            f"topics seen: {self._sub.topics() if self._sub else []}",
        ]
        if rec is not None:
            lines.append(f"joint_state keys: {sorted(rec.data)}  age {rec.age() * 1000:.0f} ms")
        return "\n".join(lines)

    # ------------------------------------------------------------ Robot API

    def reset(self, seed: int | None = None) -> Observation:
        """Connect and read. Never moves the arm: the task moves it, slowly."""
        self.connect()
        self._wait_for([self.state_topic, *self.camera_topics.values()], self.connect_timeout_s)
        self._measure_skew()
        return self.get_observation()

    def camera_config(self) -> dict[str, tuple[int, int]]:
        return {name: self.camera_resolution for name in self.camera_topics}

    def _state(self) -> dict[str, Any]:
        rec = self._sub.get(self.state_topic) if self._sub else None
        if rec is None or rec.age() > self.state_timeout_s:
            started = time.monotonic()
            if self.link_grace_s > 3.0:
                print(
                    f"[rr] link to {self.host} stalled; holding (rig holds the arm), "
                    f"waiting up to {self.link_grace_s:.0f}s"
                )
            try:
                self._wait_for([self.state_topic], self.link_grace_s)
            except RobotLinkLost as exc:
                raise RobotLinkLost(f"arm state from {self.host} went stale: {exc}") from None
            if self.link_grace_s > 3.0:
                print(f"[rr] link back after {time.monotonic() - started:.1f}s; resuming")
            rec = self._sub.get(self.state_topic)
        return rec.data

    def get_observation(self) -> Observation:
        state = self._state()
        q = np.asarray(state["joint_pos"], dtype=np.float64).ravel()[: self.spec.dof]
        grip_rr = float(np.asarray(state.get("gripper_pos", [1.0])).ravel()[0])
        images, intrinsics, depth = {}, {}, {}
        for name, topic in self.camera_topics.items():
            rec = self._rec(topic)
            if rec is None or rec.age() > self.link_grace_s:
                raise RobotLinkLost(f"camera {name} ({topic}) went stale")
            msg = rec.data
            if "rgb_jpeg" in msg["images"]:  # from camera_relay.py: JPEG of RGB
                import cv2

                bgr = cv2.imdecode(
                    np.frombuffer(msg["images"]["rgb_jpeg"], np.uint8), cv2.IMREAD_COLOR
                )
                rgb = bgr[:, :, ::-1]
            else:
                rgb = np.asarray(msg["images"]["rgb"], dtype=np.uint8)
            images[name] = np.ascontiguousarray(rgb[:, :, ::-1] if self.rgb_order == "bgr" else rgb)
            if msg.get("intrinsics"):
                intrinsics[name] = dict(msg["intrinsics"])
            if msg.get("depth_data") is not None:
                depth[name] = msg["depth_data"]
        return Observation(
            images=images,
            joint_pos=self.spec.clip_joints(q) if q.shape[0] == self.spec.dof else q,
            gripper=float(np.clip(1.0 - grip_rr, 0.0, 1.0)),
            ee_pose=self._ee_pose(q),
            timestamp=self.clock.now(),
            extra={
                "intrinsics": intrinsics,
                "depth": depth,
                "gripper_rr": grip_rr,
                "motors": _motor_fields(state),
            },
        )

    def _ee_pose(self, q: np.ndarray) -> np.ndarray | None:
        if self.spec.ee_site is None:
            return None
        if self._kin is None:
            from teleop_sim.robots.kinematics import Kinematics

            self._kin = Kinematics(self.spec)
        pos, rot = self._kin.frame(q, "site", self.spec.ee_site)
        import mujoco

        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rot.ravel())
        return np.concatenate([pos, quat])

    def send_action(self, action: Action) -> Action:
        if self._pub is None:
            raise RuntimeError("RobotsRealtimeRobot.send_action called before connect()")
        if not self.spec.supports(action.mode):
            raise UnsupportedControlMode(
                f"robot {self.spec.name!r} does not support {action.mode.value!r}"
            )
        measured = np.asarray(self._state()["joint_pos"], dtype=np.float64).ravel()[: self.spec.dof]
        target = self.spec.clip_joints(action.values)
        if self.joint_limits is not None:
            target = np.clip(target, self.joint_limits[:, 0], self.joint_limits[:, 1])
        # Never ask for a pose far from where the arm is: robots_realtime latches a
        # safe-stop on a command more than max_command_error_rad from measured.
        step = np.clip(target - measured, -self.max_step_rad, self.max_step_rad)
        if not np.allclose(step, target - measured):
            self._clipped_steps += 1
            if self._clipped_steps in (1, 10, 100):
                print(
                    f"[rr] command limited to {self.max_step_rad} rad from measured "
                    f"({self._clipped_steps} times)"
                )
        target = measured + step
        if not self._halted:
            joint_pos = np.append(target, 1.0 - action.gripper)  # rr: 1.0 = open
            self._pub.publish(self.cmd_topic, {"joint_pos": joint_pos}, ts=time.time() - self._skew)
        sent = action.replace_values(target)
        self._sent.append((time.monotonic(), sent))
        return sent

    def recent_commands(self) -> list[tuple[float, Action]]:
        """The last ~10 s of commands sent, oldest first, as (monotonic time, action)."""
        return list(self._sent)

    def command_before(self, seconds: float) -> Action | None:
        """The command sent ``seconds`` ago (or the oldest kept, ~10 s): somewhere the
        arm was just heading through, so free of whatever it met since. Safety
        monitors back off to it after a contact."""
        if not self._sent:
            return None
        cutoff = time.monotonic() - seconds
        older = [a for t, a in self._sent if t <= cutoff]
        return older[-1] if older else self._sent[0][1]

    def safe_stop(self) -> None:
        """Stop commanding. The RobotNode holds the last pose after 0.5 s of silence."""
        self._halted = True


#: Raw motor readings the driver publishes beside positions, passed through
#: untouched for telemetry. On gem13's Kronos gripper, ``gripper_eff`` is the sum
#: of |present current| of its two XC330 servos (raw units, ~mA) and
#: ``gripper_vel`` is servo velocity; ``joint_eff`` / ``joint_vel`` are the
#: Damiao arm motors' torque and velocity. Not declared as spec sensing:
#: they are unverified against a reference and nothing controls on them.
MOTOR_FIELDS = ("gripper_eff", "gripper_vel", "joint_eff", "joint_vel")


def _motor_fields(state: dict) -> dict[str, list[float]]:
    out = {}
    for key in MOTOR_FIELDS:
        if state.get(key) is not None:
            out[key] = [float(v) for v in np.asarray(state[key], dtype=np.float64).ravel()]
    return out
