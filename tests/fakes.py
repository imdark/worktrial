"""Implementations of every seam with no physics, no hardware, no rendering.

Not throwaway scaffolding. If the control loop runs against a robot with no
dynamics, a camera returning noise and an action source with no human, then the
seams are real seams. These stay in the suite permanently as the interfaces'
conscience, and they are what lets CI run anywhere with only the `dev` extra.

FakeRobot is also the first implementation put through tests/conformance/.
"""

from __future__ import annotations

import numpy as np

from teleop_sim.core.clock import Clock
from teleop_sim.core.policy_spec import PolicySpec
from teleop_sim.core.protocols import (
    Camera,
    Recorder,
    ResetStrategy,
    Robot,
    SuccessDetector,
    Teleoperator,
    UnsupportedControlMode,
)
from teleop_sim.core.registry import RESETS, ROBOTS, SUCCESS, TELEOPS, register
from teleop_sim.core.spec import CameraSpec, RobotSpec
from teleop_sim.core.types import (
    Action,
    Button,
    EpisodeResult,
    Observation,
    TeleopCommand,
)


class FakeCamera(Camera):
    """Deterministic noise at a deliberately tiny resolution.

    Overrides ``resolution`` rather than honouring ``spec.resolution``: tests do
    not need 640x480 of random bytes, and the override keeps the reported
    resolution truthful about what ``read`` actually returns -- which is what
    the startup compatibility check reads.
    """

    def __init__(self, spec: CameraSpec, size: tuple[int, int], seed: int = 0) -> None:
        self.spec = spec
        self._size = (int(size[0]), int(size[1]))
        self._rng = np.random.default_rng(seed)

    @property
    def resolution(self) -> tuple[int, int]:
        return self._size

    def read(self) -> np.ndarray:
        width, height = self._size
        return self._rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@register(ROBOTS, "fake")
class FakeRobot(Robot):
    """Perfect first-order tracking, or none at all when ``stall`` is set.

    Honours its spec in both directions: it rejects control modes the spec does
    not declare, and it populates exactly the sensing fields the spec declares.
    """

    def __init__(
        self,
        spec: RobotSpec,
        clock: Clock,
        image_size: tuple[int, int] = (64, 48),
        stall: bool = False,
        tracking: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.clock = clock
        self.stall = bool(stall)
        self.tracking = float(tracking)
        self.stopped = False
        self._connected = False
        self._q = spec.neutral_joints()
        self._gripper = 0.0
        self.cameras = [
            FakeCamera(cam, image_size, seed + i) for i, cam in enumerate(spec.cameras)
        ]

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def camera_config(self) -> dict[str, tuple[int, int]]:
        return {camera.name: camera.resolution for camera in self.cameras}

    def safe_stop(self) -> None:
        self.stopped = True

    def _ee_pose(self) -> np.ndarray:
        # Not kinematics -- a deterministic, bounded function of the joints so
        # the workspace check has something real to read.
        reach = 0.20 + 0.05 * float(self._q[1])
        return np.array(
            [reach * np.cos(self._q[0]), reach * np.sin(self._q[0]), 0.20, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    def get_observation(self) -> Observation:
        sensing = self.spec.sensing
        zeros = np.zeros_like(self._q)
        return Observation(
            images={cam.name: cam.read() for cam in self.cameras},
            joint_pos=self._q.copy(),
            joint_vel=zeros.copy(),
            joint_current=zeros.copy() if sensing.joint_current else None,
            joint_torque=zeros.copy() if sensing.joint_torque else None,
            ee_wrench=np.zeros(6) if sensing.ee_wrench else None,
            ee_pose=self._ee_pose(),
            gripper=self._gripper,
            timestamp=self.clock.now(),
        )

    def send_action(self, action: Action) -> Action:
        if not self._connected:
            raise RuntimeError("FakeRobot.send_action called before connect()")
        if not self.spec.supports(action.mode):
            raise UnsupportedControlMode(
                f"robot {self.spec.name!r} does not support {action.mode.value!r}; "
                f"supported: {[m.value for m in self.spec.supported_modes]}"
            )
        target = self.spec.clip_joints(action.values)
        if not self.stall:
            self._q = self._q + self.tracking * (target - self._q)
            self._gripper = action.gripper
        # What was actually commanded, not what was requested.
        return action.replace_values(target)

    def reset(self, seed: int | None = None) -> Observation:
        if not self._connected:
            self.connect()
        self._q = self.spec.home_joints()
        self._gripper = 0.0
        self.stopped = False
        return self.get_observation()


@register(TELEOPS, "fake")
class FakeTeleop(Teleoperator):
    """Ramps every joint away from neutral; optionally presses a button."""

    def __init__(
        self,
        spec: RobotSpec,
        clock: Clock,
        step: float = 0.01,
        button: str = Button.ESTOP,
        button_at: int | None = None,
    ) -> None:
        self.spec = spec
        self.clock = clock
        self.step = float(step)
        self.button = button
        self.button_at = button_at
        self._connected = False
        self._calls = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_command(self) -> TeleopCommand:
        self._calls += 1
        target = self.spec.neutral_joints() + self.step * self._calls
        buttons: dict[str, bool] = {}
        if self.button_at is not None and self._calls >= self.button_at:
            buttons[self.button] = True
        return TeleopCommand(
            kind="joint",
            values=self.spec.clip_joints(target),
            gripper=0.0,
            timestamp=self.clock.now(),
            buttons=buttons,
        )


@register(SUCCESS, "fake")
class FakeSuccessDetector(SuccessDetector):
    """Succeeds after a fixed number of checks, or never."""

    def __init__(self, success_after: int | None = None) -> None:
        self.success_after = success_after
        self.checks = 0

    def reset(self) -> None:
        self.checks = 0

    def is_success(self, obs: Observation) -> bool:
        self.checks += 1
        return self.success_after is not None and self.checks >= self.success_after

    def failure_tag(self, obs: Observation) -> str | None:
        return "fake_no_success"


@register(RESETS, "fake")
class CountingReset(ResetStrategy):
    """No-op reset that records that it was called."""

    def __init__(self) -> None:
        self.calls = 0

    def reset(self, rng: np.random.Generator) -> None:
        self.calls += 1


class SlowPolicy:
    """A policy whose chunk covers several ticks, for buffering tests."""

    def __init__(self, spec: RobotSpec, clock: Clock, horizon: int = 5) -> None:
        from teleop_sim.core.types import ActionChunk, ActionOrigin

        self._ActionChunk = ActionChunk
        self._origin = ActionOrigin.POLICY
        self.robot_spec = spec
        self.clock = clock
        self.horizon = horizon
        self.predict_calls = 0
        self.spec = PolicySpec(
            policy_id="slow",
            control_mode=spec.default_control_mode,
            action_horizon=horizon,
            expected_control_hz=spec.control_hz,
        )

    def reset(self) -> None:
        self.predict_calls = 0

    def predict(self, obs: Observation):
        self.predict_calls += 1
        now = self.clock.now()
        actions = [
            Action(
                mode=self.robot_spec.default_control_mode,
                values=obs.joint_pos.copy(),
                gripper=0.0,
                timestamp=now,
                obs_timestamp=obs.timestamp,
                source=self._origin,
            )
            for _ in range(self.horizon)
        ]
        return self._ActionChunk(
            actions=actions, obs_timestamp=obs.timestamp, horizon_hz=self.robot_spec.control_hz
        )


class CollectingRecorder(Recorder):
    """In-memory recorder, so tests can assert on what the loop logged."""

    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.actions: list[Action] = []
        self.results: list[EpisodeResult] = []
        self.episodes_started = 0
        self.episodes_discarded = 0

    def start_episode(self, seed: int | None = None) -> None:
        self.episodes_started += 1

    def record(self, obs: Observation, action: Action) -> None:
        self.observations.append(obs)
        self.actions.append(action)

    def end_episode(self, result: EpisodeResult) -> None:
        self.results.append(result)

    def discard_episode(self) -> None:
        self.episodes_discarded += 1
