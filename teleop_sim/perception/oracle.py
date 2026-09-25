"""Vision answered from privileged simulator state.

Two jobs. It runs the whole pick pipeline in sim with no API key and no
network, so the geometry, the motion and the loop can be tested on their own.
And it is the ground truth a VLM's answers are scored against later: it
answers ``locate`` by projecting the true glass position through the same
camera model the policy uses, so any difference is the model's error, not the
geometry's.
"""

from __future__ import annotations

import numpy as np

from teleop_sim.perception.camera import CameraModel
from teleop_sim.perception.vision import Detection, Pixel, TargetPlan, Verdict, Views, Vision


class OracleVision(Vision):
    def __init__(self, robot, target_body: str, height: float = 0.10, lifted_above: float = 0.03):
        import mujoco

        self._mj = mujoco
        self.robot = robot
        self.target_body = target_body
        self.height = float(height)
        self.lifted_above = float(lifted_above)
        self._bid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, target_body)
        if self._bid < 0:
            raise ValueError(f"oracle target {target_body!r} is not a body in the scene")
        self._rest_z = self._pos()[2]

    def _pos(self) -> np.ndarray:
        return self.robot.data.xpos[self._bid].copy()

    def plan(self, views: Views, instruction: str) -> TargetPlan:
        return TargetPlan(description=self.target_body)

    def locate(self, image: np.ndarray, camera: CameraModel, target: str) -> Detection:
        base = self._pos()
        base_px = camera.project(base)
        rim_px = camera.project(base + np.array([0.0, 0.0, self.height]))
        if base_px is None or not camera.in_image(*base_px):
            return Detection(found=False, note="target not in view", model="oracle")
        return Detection(
            found=True,
            base=Pixel(*base_px),
            rim=None if rim_px is None else Pixel(*rim_px),
            confidence=1.0,
            model="oracle",
        )

    def verify(self, views: Views, question: str) -> Verdict:
        lifted = self._pos()[2] - self._rest_z
        return Verdict(
            ok=bool(lifted > self.lifted_above),
            confidence=1.0,
            reason=f"target raised {lifted * 1000:.0f} mm",
            model="oracle",
        )
