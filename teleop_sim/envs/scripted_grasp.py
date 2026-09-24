"""A scripted side grasp for the YAM glasses scene.

Used by the scene tests and by scripts/record_demo.py. Not a policy and not the
Stage 4 scripted expert -- just enough IK to show the environment admits the
task with a given end effector: the glasses are reachable, fit between the
fingers, and can be lifted. Everything is driven through the Robot interface,
never by writing qpos, so it also exercises the driver.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from teleop_sim.core.types import Action
from teleop_sim.robots.sim.mujoco_robot import MujocoRobot


@dataclass(frozen=True)
class SideGrasp:
    """Found by sweep, not derived: 15 mm past the glass centre puts the pads on
    its widest chord (at the centre they land ~9 mm short and wedge the glass
    out), and a 40 degree downward pitch is the shallowest that reaches this
    height with the fingers horizontal."""

    past_centre: float = 0.015
    height: float = 0.075
    pitch_deg: float = 40.0
    standoff: float = 0.12
    lift: float = 0.12


DEFAULT_GRASP = SideGrasp()

#: Kronos grasps near its fingertips (the pads form a V), so it wants to sit
#: further past the glass centre and higher. Found by sweep: 3 of 16
#: configurations held upright; this one tilts the glass 4.8 deg.
KRONOS_GRASP = SideGrasp(past_centre=0.02, height=0.09, pitch_deg=45.0)


class GraspScript:
    def __init__(self, robot: MujocoRobot, on_step: Callable[[], None] | None = None) -> None:
        import mujoco

        self._mj = mujoco
        self.robot = robot
        #: Called after every control step -- how the demo recorder grabs frames.
        self.on_step = on_step
        self.spec = robot.spec
        self.model, self.data = robot.model, robot.data
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.spec.ee_site)

    def body_pos(self, name: str) -> np.ndarray:
        bid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def body_tilt_deg(self, name: str) -> float:
        bid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, name)
        up = self.data.xmat[bid].reshape(3, 3)[2, 2]
        return float(np.degrees(np.arccos(np.clip(up, -1.0, 1.0))))

    def tool_orientation(self, pitch_deg: float) -> np.ndarray:
        """The home tool orientation, pitched down about world y. Pitching about
        y keeps the fingers -- which travel along world y at home -- horizontal."""
        mj = self._mj
        save = self.data.qpos.copy()
        self.data.qpos[self.robot._qpos_adr] = self.spec.home_joints()
        mj.mj_kinematics(self.model, self.data)
        r_home = self.data.site_xmat[self.site].reshape(3, 3).copy()
        self.data.qpos[:] = save
        mj.mj_forward(self.model, self.data)
        a = np.radians(pitch_deg)
        pitch = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        return pitch @ r_home

    def ik(self, target: np.ndarray, rot: np.ndarray, seed: np.ndarray, iters: int = 600):
        """Damped least squares on the grasp site. Returns (q, pos_err, rot_err)."""
        mj, m, d = self._mj, self.model, self.data
        save_q, save_v = d.qpos.copy(), d.qvel.copy()
        q = seed.copy()
        err = np.ones(6)
        for _ in range(iters):
            d.qpos[self.robot._qpos_adr] = q
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            r = d.site_xmat[self.site].reshape(3, 3)
            err = np.concatenate(
                [target - d.site_xpos[self.site],
                 0.5 * sum(np.cross(r[:, i], rot[:, i]) for i in range(3))]
            )
            if np.linalg.norm(err) < 1e-6:
                break
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mj.mj_jacSite(m, d, jp, jr, self.site)
            jac = np.vstack([jp, jr])[:, self.robot._dof_adr]
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
            q = self.spec.clip_joints(q + 0.5 * step)
        d.qpos[:], d.qvel[:] = save_q, save_v
        mj.mj_forward(m, d)
        return q, float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))

    def move(self, q_from, q_to, gripper: float, steps: int, settle: int = 15) -> np.ndarray:
        """Joint-space interpolation. Jumping straight to a far target slams the
        fingertips into the table hard enough to wedge the fingers shut."""
        def send(q):
            self.robot.send_action(
                Action(mode=self.spec.default_control_mode, values=q, gripper=gripper,
                       timestamp=0.0, obs_timestamp=0.0)
            )
            if self.on_step is not None:
                self.on_step()
        for k in range(1, steps + 1):
            send(q_from + (q_to - q_from) * (k / steps))
        for _ in range(settle):
            send(q_to)
        return q_to

    def plan(self, glass: str, grasp: SideGrasp = DEFAULT_GRASP):
        """Joint waypoints: above-standoff, standoff, grasp, lifted. None if unreachable."""
        rot = self.tool_orientation(grasp.pitch_deg)
        centre = self.body_pos(glass)
        gx, gy, z = centre[0] + grasp.past_centre, centre[1], grasp.height
        targets = [
            (gx - grasp.standoff, gy, z + 0.03),
            (gx - grasp.standoff, gy, z),
            (gx, gy, z),
            (gx, gy, z + grasp.lift),
        ]
        waypoints, seed = [], self.spec.home_joints()
        for target in targets:
            q, pos_err, rot_err = self.ik(np.array(target), rot, seed)
            if pos_err > 1e-3 or rot_err > 1e-3:
                return None
            waypoints.append(q)
            seed = q
        return waypoints

    def pick(self, glass: str, grasp: SideGrasp = DEFAULT_GRASP) -> bool:
        """Approach, close, lift. Returns False if the plan was unreachable."""
        waypoints = self.plan(glass, grasp)
        if waypoints is None:
            return False
        above, standoff, at_glass, lifted = waypoints
        q = self.move(self.spec.home_joints(), above, 0.0, 60)
        q = self.move(q, standoff, 0.0, 30)
        q = self.move(q, at_glass, 0.0, 45)
        q = self.move(q, at_glass, 1.0, 30, settle=30)
        self.move(q, lifted, 1.0, 60, settle=60)
        return True
