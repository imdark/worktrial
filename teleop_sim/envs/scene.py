"""Handles onto the objects in a MuJoCo scene.

Concrete and sim-only by design (§2.1): there is no real-hardware counterpart
to teleporting a cube, and pretending there is would be the exact
over-abstraction principle 8 warns about. Randomisation builds on this at
Stage 4.
"""

from __future__ import annotations

import numpy as np


class SceneError(KeyError):
    """The scene has no such body, or it is not free to be moved."""


class Scene:
    def __init__(self, model, data) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = data

        # Only bodies with a free joint can be placed; everything else is
        # welded into the kinematic tree and moving it means editing the model.
        # Both addresses are bound here: qpos and qvel are indexed by JOINT,
        # and reaching for them later via a body id is an easy, silent mistake.
        self._free: dict[str, tuple[int, int]] = {}
        for jid in range(model.njnt):
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[jid])
            self._free[body] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))

    @property
    def movable_bodies(self) -> list[str]:
        return sorted(self._free)

    def _addresses(self, body: str) -> tuple[int, int]:
        if body not in self._free:
            raise SceneError(
                f"body {body!r} has no free joint; movable bodies are {self.movable_bodies}"
            )
        return self._free[body]

    def body_pose(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """Position and wxyz orientation of a movable body."""
        adr, _ = self._addresses(body)
        qpos = self.data.qpos
        return qpos[adr : adr + 3].copy(), qpos[adr + 3 : adr + 7].copy()

    def set_body_pose(
        self, body: str, pos: np.ndarray, quat: np.ndarray | None = None
    ) -> None:
        adr, dof = self._addresses(body)
        self.data.qpos[adr : adr + 3] = np.asarray(pos, dtype=np.float64)[:3]
        if quat is not None:
            self.data.qpos[adr + 3 : adr + 7] = np.asarray(quat, dtype=np.float64)[:4]
        # A teleported body keeps its old velocity otherwise, and shoots off.
        self.data.qvel[dof : dof + 6] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def settle(self, steps: int = 100) -> None:
        """Let placed objects come to rest before an episode starts."""
        for _ in range(steps):
            self._mujoco.mj_step(self.model, self.data)
