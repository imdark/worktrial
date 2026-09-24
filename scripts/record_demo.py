"""Record one simulated pick as a multi-view video, plus stills.

    python scripts/record_demo.py                                   # Kronos, glasses table
    python scripts/record_demo.py --spec teleop_sim/robots/specs/yam.yaml --grasp stock

Every control step becomes one frame (30 fps, real time) tiling six views:

    3/4 view          side view        gripper close-up (tracking)
    top-down view     `top` camera     `wrist` camera

The bottom row's right two tiles are the cameras a policy actually receives;
the other four are for people. Writes to outputs/<name>/ (gitignored):
<name>.mp4, stills at the key moments, and a contact sheet of the stills.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from teleop_sim.core.clock import WallClock
from teleop_sim.core.parts import SceneSpec
from teleop_sim.core.spec import RobotSpec
from teleop_sim.envs.scripted_grasp import DEFAULT_GRASP, KRONOS_GRASP, GraspScript
from teleop_sim.robots.sim.mujoco_robot import MujocoRobot

ROOT = Path(__file__).resolve().parent.parent
TILE = (480, 360)  # width, height
HOLD = 60
# Frame counts per phase, as GraspScript.pick() moves (steps + settle each).
PHASES = [("approach", 1 + 75 + 45), ("reach in", 60), ("close", 60), ("lift", 120), ("hold", HOLD)]


class Recorder:
    def __init__(self, robot: MujocoRobot, ee_body: str) -> None:
        self.robot = robot
        self.renderer = mujoco.Renderer(robot.model, height=TILE[1], width=TILE[0])
        self.ee_body = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        self.glass = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "glass_right")
        self.rest_z = float(robot.data.xpos[self.glass][2])
        self.frames: list[np.ndarray] = []
        self.step = 0

    def _free(self, lookat, distance, azimuth, elevation) -> np.ndarray:
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance = lookat, distance
        cam.azimuth, cam.elevation = azimuth, elevation
        self.renderer.update_scene(self.robot.data, camera=cam)
        return self.renderer.render()

    def phase(self) -> str:
        n = self.step
        for name, steps in PHASES:
            if n < steps:
                return name
            n -= steps
        return PHASES[-1][0]

    def capture(self) -> None:
        d = self.robot.data
        obs = self.robot.get_observation()
        workspace = np.array([0.30, 0.0, 0.16])
        gripper = d.xpos[self.ee_body] + d.xmat[self.ee_body].reshape(3, 3)[:, 2] * 0.12
        tiles = [
            ("3/4 view", self._free(workspace, 1.05, 215, -22)),
            ("side view", self._free(workspace, 0.95, 90, -6)),
            ("gripper close-up", self._free(gripper, 0.36, 150, -28)),
            ("top-down", self._free(np.array([0.32, 0.0, 0.0]), 0.95, 180, -89)),
            ("policy camera: top", obs.images["top"]),
            ("policy camera: wrist", obs.images["wrist"]),
        ]
        labelled = []
        for label, img in tiles:
            img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
            cv2.rectangle(img, (0, 0), (TILE[0], 26), (20, 20, 20), -1)
            cv2.putText(
                img, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA
            )
            labelled.append(img)
        frame = np.vstack([np.hstack(labelled[:3]), np.hstack(labelled[3:])])
        bar = np.full((34, frame.shape[1], 3), 32, np.uint8)
        text = (
            f"{self.robot.spec.name}   t = {self.step / self.robot.spec.control_hz:5.2f} s   "
            f"phase: {self.phase():9}   gripper {obs.gripper:4.2f}   "
            f"glass lifted {(d.xpos[self.glass][2] - self.rest_z) * 1000:6.1f} mm"
        )
        cv2.putText(
            bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA
        )
        self.frames.append(np.vstack([bar, frame]))
        self.step += 1


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> str:
    h, w = frames[0].shape[:2]
    for codec in ("avc1", "mp4v"):  # H.264 plays in browsers; mp4v is the fallback
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
            return codec
    raise RuntimeError("no working MP4 encoder in this OpenCV build")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="teleop_sim/robots/specs/yam_kronos.yaml")
    parser.add_argument("--scene", default="teleop_sim/scenes/glasses_table.yaml")
    parser.add_argument("--grasp", choices=["kronos", "stock"], default="kronos")
    parser.add_argument("--hold", type=int, default=HOLD, help="control steps to hold at the end")
    args = parser.parse_args()

    spec = RobotSpec.from_yaml(ROOT / args.spec).with_scene(SceneSpec.from_yaml(ROOT / args.scene))
    robot = MujocoRobot(spec, WallClock(), image_size=TILE)
    robot.reset(seed=0)
    rec = Recorder(robot, spec.assembly.end_effector.root_body)
    rest_z = rec.rest_z
    script = GraspScript(robot, on_step=rec.capture)

    rec.capture()
    grasp = KRONOS_GRASP if args.grasp == "kronos" else DEFAULT_GRASP
    plan = script.plan("glass_right", grasp)
    if plan is None or not script.pick("glass_right", grasp):
        raise SystemExit("grasp plan unreachable")
    # Hold at the planned pose, not the measured one: the position servos sag
    # under the load, and commanding the sagged pose would sag it again.
    lifted = plan[-1]
    script.move(lifted, lifted, 1.0, steps=args.hold, settle=0)
    held = script.body_pos("glass_right")[2] - rest_z

    name = f"{spec.name}_glass_pick"
    out = ROOT / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    codec = write_video(out / f"{name}.mp4", rec.frames, spec.control_hz)

    # Stills: the first frame, then the last frame of each phase.
    marks, n = [0], 0
    for _, steps in PHASES:
        n += steps
        marks.append(min(n, len(rec.frames)) - 1)
    labels = ["start", "approached", "reached in", "closed", "lifted", "held"]
    stills = []
    for label, i in zip(labels, marks, strict=True):
        cv2.imwrite(str(out / f"{i:04d}_{label.replace(' ', '_')}.png"), rec.frames[i])
        stills.append(cv2.resize(rec.frames[i], None, fx=0.5, fy=0.5))
    cv2.imwrite(
        str(out / "contact_sheet.png"), np.vstack([np.hstack(stills[:3]), np.hstack(stills[3:])])
    )

    tilt = script.body_tilt_deg("glass_right")
    print(
        f"{len(rec.frames)} frames ({len(rec.frames) / spec.control_hz:.1f} s, {codec}) -> "
        f"{out.relative_to(ROOT)}/"
    )
    print(f"glass lifted {held * 1000:.1f} mm, tilted {tilt:.1f} deg after a {args.hold}-step hold")


if __name__ == "__main__":
    main()
