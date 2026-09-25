"""Forward and inverse kinematics on a robot description, off the physics state.

One MuJoCo model is built from the same RobotSpec the simulator uses, with its
own MjData. Planning therefore never touches the state a simulation is
stepping, and the real arm is planned against exactly the geometry the sim
has -- which is the point: a plan that works in sim is the plan sent to
hardware.

Known gap, measured: this is the Menagerie-derived YAM, and i2rt's own yam.xml
(the model the real controller uses for gravity compensation) differs from it
by ~6 mm mean / ~12 mm worst at the flange over the joint range. Joint signs
and zeros agree. Targets located by the wrist camera and reached with the same
model largely cancel that error; targets from a fixed camera would not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teleop_sim.core.spec import RobotSpec


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    pos_err: float  # metres
    rot_err: float  # radians
    ok: bool


class Kinematics:
    FRAME_KINDS = ("site", "camera", "body")

    def __init__(self, spec: RobotSpec) -> None:
        import mujoco

        from teleop_sim.robots.sim.assembly import build_model

        self._mj = mujoco
        self.spec = spec
        self.model = build_model(spec)
        self.data = mujoco.MjData(self.model)
        ids = []
        for name in spec.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name!r} is not in the model of {spec.name!r}")
            ids.append(jid)
        self._qadr = np.array([self.model.jnt_qposadr[j] for j in ids], dtype=int)
        self._dadr = np.array([self.model.jnt_dofadr[j] for j in ids], dtype=int)

    # ------------------------------------------------------------ frames

    def _id(self, kind: str, name: str) -> int:
        objtype = {
            "site": self._mj.mjtObj.mjOBJ_SITE,
            "camera": self._mj.mjtObj.mjOBJ_CAMERA,
            "body": self._mj.mjtObj.mjOBJ_BODY,
        }[kind]
        ident = self._mj.mj_name2id(self.model, objtype, name)
        if ident < 0:
            raise ValueError(f"{kind} {name!r} is not in the model of {self.spec.name!r}")
        return ident

    def _set(self, q: np.ndarray) -> None:
        self.data.qpos[self._qadr] = q
        self._mj.mj_kinematics(self.model, self.data)
        self._mj.mj_comPos(self.model, self.data)
        # Camera frames are not part of mj_kinematics; mj_camlight fills them.
        self._mj.mj_camlight(self.model, self.data)

    def _read(self, kind: str, ident: int) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        if kind == "site":
            return d.site_xpos[ident].copy(), d.site_xmat[ident].reshape(3, 3).copy()
        if kind == "camera":
            return d.cam_xpos[ident].copy(), d.cam_xmat[ident].reshape(3, 3).copy()
        return d.xpos[ident].copy(), d.xmat[ident].reshape(3, 3).copy()

    def frame(self, q: np.ndarray, kind: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation (columns = frame axes) of a named frame at ``q``.

        A camera frame follows MuJoCo's convention: it looks along -z, +y up.
        """
        ident = self._id(kind, name)
        self._set(np.asarray(q, dtype=np.float64))
        return self._read(kind, ident)

    # ------------------------------------------------------------ IK

    def ik(
        self,
        kind: str,
        name: str,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        seed: np.ndarray,
        iters: int = 800,
        pos_tol: float = 1e-3,
        rot_tol: float = np.radians(1.0),
    ) -> IKResult:
        """Damped least squares on a named frame, clipped to the spec's joint limits."""
        mj, m, d = self._mj, self.model, self.data
        ident = self._id(kind, name)
        body = (
            ident
            if kind == "body"
            else (m.site_bodyid[ident] if kind == "site" else m.cam_bodyid[ident])
        )
        target_pos = np.asarray(target_pos, dtype=np.float64)
        q = self.spec.clip_joints(np.asarray(seed, dtype=np.float64))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err = np.ones(6)
        for _ in range(iters):
            self._set(q)
            pos, rot = self._read(kind, ident)
            err = np.concatenate(
                [
                    target_pos - pos,
                    0.5 * sum(np.cross(rot[:, i], target_rot[:, i]) for i in range(3)),
                ]
            )
            if np.linalg.norm(err[:3]) < pos_tol * 0.1 and np.linalg.norm(err[3:]) < rot_tol * 0.1:
                break
            mj.mj_jac(m, d, jp, jr, pos, body)
            jac = np.vstack([jp, jr])[:, self._dadr]
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
            q = self.spec.clip_joints(q + 0.5 * step)
        self._set(q)
        pos, rot = self._read(kind, ident)
        pos_err = float(np.linalg.norm(target_pos - pos))
        rot_err = float(np.arccos(np.clip((np.trace(rot.T @ target_rot) - 1.0) / 2.0, -1.0, 1.0)))
        return IKResult(
            q=q, pos_err=pos_err, rot_err=rot_err, ok=pos_err <= pos_tol and rot_err <= rot_tol
        )

    def ik_multi_seed(
        self, kind: str, name: str, target_pos, target_rot, seeds: list[np.ndarray], **kwargs
    ) -> IKResult:
        """Best of several seeds; the first that converges wins."""
        best: IKResult | None = None
        for seed in seeds:
            result = self.ik(kind, name, target_pos, target_rot, seed, **kwargs)
            if result.ok:
                return result
            if (
                best is None
                or result.pos_err + 0.1 * result.rot_err < best.pos_err + 0.1 * best.rot_err
            ):
                best = result
        assert best is not None
        return best


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
