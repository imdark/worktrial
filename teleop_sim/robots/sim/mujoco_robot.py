"""MuJoCo implementation of the Robot seam.

Everything the loop, the recorder and any policy see comes through the same
interface FakeRobot implements, and this class is held to the same contract by
tests/conformance/.

Joint indexing is by name through the model, never by position: the scene
carries a free joint for the cube, so qpos is not the arm's joint vector and
assuming otherwise is a silent, catastrophic off-by-seven.
"""

from __future__ import annotations

import numpy as np

from teleop_sim.core.clock import Clock
from teleop_sim.core.protocols import Robot, UnsupportedControlMode
from teleop_sim.core.registry import ROBOTS, register
from teleop_sim.core.spec import RobotSpec, SpecError
from teleop_sim.core.types import Action, Observation


class ModelMismatch(SpecError):
    """The MJCF and the RobotSpec disagree about what this robot is."""


@register(ROBOTS, "mujoco")
class MujocoRobot(Robot):
    def __init__(
        self,
        spec: RobotSpec,
        clock: Clock,
        image_size: tuple[int, int] | None = None,
        render_cameras: list[str] | None = None,
        substeps: int | None = None,
    ) -> None:
        import mujoco

        from teleop_sim.cameras.sim_cam import MujocoCamera, RendererPool

        self._mujoco = mujoco
        self.spec = spec
        self.clock = clock

        if not spec.mjcf_path:
            raise ModelMismatch(
                f"robot {spec.name!r}: mjcf_path is not set; MujocoRobot needs a model"
            )
        self.model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
        self.data = mujoco.MjData(self.model)

        self._check_sensing()
        self._bind_joints()
        self._bind_gripper()
        self._bind_ee()

        # Each send_action advances one control period of simulated time.
        period = 1.0 / spec.control_hz
        exact = period / self.model.opt.timestep
        self.substeps = int(substeps) if substeps else max(1, round(exact))
        if substeps is None:
            self._check_control_period(period, exact)

        selected = render_cameras if render_cameras is not None else spec.camera_names
        self._pool = RendererPool(self.model)
        self.cameras = [
            MujocoCamera(spec.camera(name), self.data, self._pool, size=image_size)
            for name in selected
        ]
        for camera in self.cameras:
            self._require(mujoco.mjtObj.mjOBJ_CAMERA, camera.name, "camera")

        self._connected = False
        self.reset()

    def _check_control_period(self, period: float, exact: float) -> None:
        """A control period that is not a whole number of physics substeps runs
        long or short on every single step. It is invisible in sim and shows up
        later as an unexplained sim2real gap, so refuse it here."""
        drift = abs(self.substeps - exact) / exact
        if drift > 0.005:
            timestep = self.model.opt.timestep
            raise ModelMismatch(
                f"control_hz {self.spec.control_hz:g} over timestep {timestep:g} needs "
                f"{exact:.4f} substeps, which rounds to {self.substeps} and makes every "
                f"control step run {drift:.1%} long. Set the MJCF timestep to "
                f"{period / self.substeps!r} (or pick a control rate that divides it)."
            )

    # ------------------------------------------------------------ binding

    def _require(self, objtype, name: str, label: str) -> int:
        ident = self._mujoco.mj_name2id(self.model, objtype, name)
        if ident < 0:
            raise ModelMismatch(
                f"{label} {name!r} is declared in spec {self.spec.name!r} but is not "
                f"in {self.spec.mjcf_path}"
            )
        return ident

    def _bind_joints(self) -> None:
        mujoco = self._mujoco
        ids = [
            self._require(mujoco.mjtObj.mjOBJ_JOINT, name, "joint")
            for name in self.spec.joint_names
        ]
        self._qpos_adr = np.array([self.model.jnt_qposadr[i] for i in ids], dtype=int)
        self._dof_adr = np.array([self.model.jnt_dofadr[i] for i in ids], dtype=int)
        self._act_ids = np.array(
            [
                self._require(mujoco.mjtObj.mjOBJ_ACTUATOR, name, "actuator for joint")
                for name in self.spec.joint_names
            ],
            dtype=int,
        )

    def _bind_gripper(self) -> None:
        mujoco = self._mujoco
        joint = self.spec.gripper.joint
        jid = self._require(mujoco.mjtObj.mjOBJ_JOINT, joint, "gripper joint")
        self._gripper_jid = jid
        self._gripper_qadr = int(self.model.jnt_qposadr[jid])
        self._gripper_act = self._require(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.spec.gripper.actuator_name, "gripper actuator"
        )

    def _bind_ee(self) -> None:
        mujoco = self._mujoco
        self._ee_body = self._require(
            mujoco.mjtObj.mjOBJ_BODY, self.spec.ee_link, "end-effector body"
        )
        self._ee_site = (
            self._require(mujoco.mjtObj.mjOBJ_SITE, self.spec.ee_site, "end-effector site")
            if self.spec.ee_site
            else None
        )

    def _ee_pose(self) -> np.ndarray:
        if self._ee_site is None:
            return np.concatenate([self.data.xpos[self._ee_body], self.data.xquat[self._ee_body]])
        quat = np.empty(4)
        self._mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._ee_site])
        return np.concatenate([self.data.site_xpos[self._ee_site], quat])

    def _sync_coupled_joints(self) -> None:
        """Satisfy linear joint-equality constraints before the first step.

        A two-finger gripper usually slaves one finger to the other through an
        equality constraint. Writing only the driven finger's qpos at reset
        leaves the constraint violated, and the solver snaps the other finger
        into place on step one -- an impulse at t=0 in every episode.
        """
        mujoco = self._mujoco
        for eq in range(self.model.neq):
            if self.model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            j1, j2 = int(self.model.eq_obj1id[eq]), int(self.model.eq_obj2id[eq])
            c0, c1, *higher = self.model.eq_data[eq][:5]
            if j2 < 0 or abs(c1) < 1e-12 or any(abs(c) > 1e-12 for c in higher):
                continue  # only linear two-joint couplings are handled
            q1 = self.data.qpos[self.model.jnt_qposadr[j1]]
            self.data.qpos[self.model.jnt_qposadr[j2]] = (q1 - c0) / c1

    def _check_sensing(self) -> None:
        """Refuse to pretend to sense things this simulation cannot.

        joint_torque is real (qfrc_actuator). Motor current has no MuJoCo
        analogue, and an end-effector wrench needs force/torque sensors in the
        MJCF -- claiming either would put fabricated numbers into a dataset
        that a real arm would have to reproduce.
        """
        for field, reason in (
            ("joint_current", "MuJoCo does not model motor current"),
            ("ee_wrench", "this model declares no force/torque sensor at the end effector"),
        ):
            if self.spec.sensing.provides(field):
                raise ModelMismatch(
                    f"spec {self.spec.name!r} declares sensing.{field}: true, but "
                    f"{reason}. Set it false, or add the sensor to the MJCF."
                )

    # ------------------------------------------------------------ Robot API

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        self._pool.close()

    def camera_config(self) -> dict[str, tuple[int, int]]:
        return {camera.name: camera.resolution for camera in self.cameras}

    def safe_stop(self) -> None:
        """Hold position and stop the jaws."""
        self.data.ctrl[self._act_ids] = self.data.qpos[self._qpos_adr]
        self.data.ctrl[self._gripper_act] = self.data.qpos[self._gripper_qadr]
        self.data.qvel[:] = 0.0

    def get_observation(self) -> Observation:
        sensing = self.spec.sensing
        return Observation(
            images={camera.name: camera.read() for camera in self.cameras},
            joint_pos=self.data.qpos[self._qpos_adr].copy(),
            joint_vel=self.data.qvel[self._dof_adr].copy(),
            joint_torque=(
                self.data.qfrc_actuator[self._dof_adr].copy()
                if sensing.joint_torque
                else None
            ),
            gripper=self.spec.gripper.normalize(float(self.data.qpos[self._gripper_qadr])),
            ee_pose=self._ee_pose(),
            # Loop time, not sim time (principle 7: the loop owns the clock).
            # Sim time rides along so a dataset can be replayed deterministically.
            timestamp=self.clock.now(),
            extra={"sim_time": float(self.data.time)},
        )

    def send_action(self, action: Action) -> Action:
        if not self._connected:
            raise RuntimeError("MujocoRobot.send_action called before connect()")
        if not self.spec.supports(action.mode):
            raise UnsupportedControlMode(
                f"robot {self.spec.name!r} does not support {action.mode.value!r}; "
                f"supported: {[m.value for m in self.spec.supported_modes]}"
            )

        target = self.spec.clip_joints(action.values)
        self.data.ctrl[self._act_ids] = target
        self.data.ctrl[self._gripper_act] = self.spec.gripper.denormalize(action.gripper)
        for _ in range(self.substeps):
            self._mujoco.mj_step(self.model, self.data)

        # What was actually commanded, not what was requested.
        return action.replace_values(target)

    def reset(self, seed: int | None = None) -> Observation:
        if not self._connected:
            self.connect()
        self._mujoco.mj_resetData(self.model, self.data)

        home = self.spec.home_joints()
        self.data.qpos[self._qpos_adr] = home
        self.data.qpos[self._gripper_qadr] = self.spec.gripper.open_pos
        self._sync_coupled_joints()
        self.data.ctrl[self._act_ids] = home
        self.data.ctrl[self._gripper_act] = self.spec.gripper.open_pos
        self._mujoco.mj_forward(self.model, self.data)
        return self.get_observation()
