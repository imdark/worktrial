"""A robots_realtime rig made of MuJoCo: the real driver's path, tested without the arm.

It speaks the protocol RobotsRealtimeRobot speaks (robots/real/rr_wire.py) --
the same broker ports, topics, envelope, gripper convention and 0.5 s command
timeout as gem13_inference.yaml -- with a simulated YAM + Kronos behind it.
Pointing the real config at it (``--host 127.0.0.1``) runs everything that
will run against gem13 except CAN and the physical cameras: the bridge, the
clock-skew handling, the step limiting, the camera plumbing, the VLM calls.

    python -m teleop_sim.robots.real.sim_rig          # then, in another shell:
    python scripts/run_pick.py teleop_sim/configs/yam_kronos_pick_real.yaml --host 127.0.0.1
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from teleop_sim.core.types import Action
from teleop_sim.robots.real.rr_wire import (
    DEFAULT_PUB_PORT,
    DEFAULT_SUB_PORT,
    BusPublisher,
    BusSubscriber,
)

PACKAGE = Path(__file__).resolve().parents[2]


class SimRig:
    def __init__(
        self,
        pub_port: int = DEFAULT_PUB_PORT,
        sub_port: int = DEFAULT_SUB_PORT,
        arm: str = "yam_right",
        cmd_topic: str | None = None,
        cameras: dict[str, str] | None = None,
        camera_hz: float = 10.0,
        cmd_timeout_s: float = 0.5,
        start_joints: list[float] | None = None,
    ) -> None:
        from teleop_sim.core.parts import SceneSpec
        from teleop_sim.core.spec import RobotSpec

        spec = RobotSpec.from_yaml(PACKAGE / "robots/specs/yam_kronos.yaml").with_scene(
            SceneSpec.from_yaml(PACKAGE / "scenes/glasses_table.yaml")
        )
        # rr node name -> sim camera it renders
        self.cameras = dict(cameras or {"right_wrist_cam": "wrist", "exo_cam": "top"})
        self.robot = None  # built on the rig thread: a GL context belongs to its thread
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.spec = spec
        self.arm = arm
        self.cmd_topic = cmd_topic or f"hud/teleop/{arm}/joint_pos"
        self.pub_port, self.sub_port = pub_port, sub_port
        self.camera_period = 1.0 / camera_hz
        self.cmd_timeout_s = cmd_timeout_s
        self.commands_applied = 0
        self.last_command: np.ndarray | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        home = spec.home_joints() if start_joints is None else np.asarray(start_joints, float)
        self._target = Action(
            mode=spec.default_control_mode, values=home, gripper=0.0, timestamp=0.0
        )
        self._lock = threading.Lock()

    def start(self) -> SimRig:
        import zmq

        ctx = zmq.Context.instance()
        self._xsub = ctx.socket(zmq.XSUB)
        self._xsub.bind(f"tcp://127.0.0.1:{self.pub_port}")
        self._xpub = ctx.socket(zmq.XPUB)
        self._xpub.bind(f"tcp://127.0.0.1:{self.sub_port}")
        broker = threading.Thread(target=self._broker, daemon=True, name="sim-rig-broker")
        broker.start()
        self._pub = BusPublisher("127.0.0.1", self.pub_port, src=self.arm)
        self._sub = BusSubscriber("127.0.0.1", [self.cmd_topic], port=self.sub_port)
        loop = threading.Thread(target=self._run_safely, daemon=True, name="sim-rig")
        loop.start()
        self._threads = [broker, loop]
        if not self._ready.wait(30.0) or self._error is not None:
            raise RuntimeError(f"sim rig failed to start: {self._error}")
        return self

    def _run_safely(self) -> None:
        try:
            self._run()
        except BaseException as exc:  # surface to start() instead of dying silently
            self._error = exc
            self._ready.set()
            raise

    def _broker(self) -> None:
        import zmq

        try:
            zmq.proxy(self._xsub, self._xpub)
        except zmq.ContextTerminated:
            pass
        except zmq.ZMQError:
            pass

    def _run(self) -> None:
        from teleop_sim.core.clock import WallClock
        from teleop_sim.robots.sim.mujoco_robot import MujocoRobot

        self.robot = MujocoRobot(self.spec, WallClock(), render_cameras=list(self.cameras.values()))
        period = 1.0 / self.spec.control_hz
        next_cam = 0.0
        self.robot.reset()
        self._ready.set()
        while not self._stop.is_set():
            started = time.monotonic()
            rec = self._sub.get(self.cmd_topic)
            if rec is not None and time.time() - float(rec.envelope["ts"]) <= self.cmd_timeout_s:
                jp = np.asarray(rec.data["joint_pos"], dtype=float)
                with self._lock:
                    self.last_command = jp.copy()
                    self.commands_applied += 1
                    self._target = Action(
                        mode=self.spec.default_control_mode,
                        values=jp[:6],
                        gripper=float(np.clip(1.0 - jp[6], 0.0, 1.0)),
                        timestamp=0.0,
                    )
            with self._lock:
                self.robot.send_action(self._target)  # stale command: hold the last target
            q = self.robot.data.qpos[self.robot._qpos_adr].copy()
            grip = self.spec.gripper.normalize(
                float(self.robot.data.qpos[self.robot._gripper_qadr])
            )
            self._pub.publish(
                f"{self.arm}/joint_state",
                {
                    "joint_pos": q,
                    "joint_vel": np.zeros(6),
                    "joint_eff": np.zeros(6),
                    "gripper_pos": np.array([1.0 - grip]),
                },
            )
            now = time.monotonic()
            if now >= next_cam:
                next_cam = now + self.camera_period
                for node, cam in self.cameras.items():
                    camera = next(c for c in self.robot.cameras if c.name == cam)
                    K = camera.intrinsics
                    w, h = camera.resolution
                    self._pub.publish(
                        f"{node}/rgb",
                        {
                            "images": {"rgb": camera.read()},
                            "timestamp": time.time(),
                            "intrinsics": {
                                "fx": K[0, 0],
                                "fy": K[1, 1],
                                "cx": K[0, 2],
                                "cy": K[1, 2],
                                "width": w,
                                "height": h,
                            },
                        },
                    )
            time.sleep(max(0.0, period - (time.monotonic() - started)))

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads[1:]:
            t.join(timeout=2.0)
        self._pub.close()
        self._sub.close()
        self._xsub.close(linger=0)
        self._xpub.close(linger=0)


if __name__ == "__main__":
    rig = SimRig().start()
    print(
        f"sim rig up: publish to tcp://127.0.0.1:{rig.pub_port}, "
        f"subscribe on {rig.sub_port}. Ctrl-C to stop."
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        rig.stop()
