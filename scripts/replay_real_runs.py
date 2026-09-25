"""Replay the three real gem13 grasps in sim, set up the way the rig was.

    python scripts/replay_real_runs.py oracle                    # physics only, no API
    python scripts/replay_real_runs.py claude --runs-dir runs    # Claude vision, logged
    python scripts/replay_real_runs.py claude --video outputs/replays run3   # one run, filmed

The simulated robot is yam_kronos_gem13 (the rig as measured: 2 deg tool pitch,
jaw travel, D405 field of view) while the planner keeps the nominal yam_kronos,
the same mismatch the real runs had. The cup is placed where the rig found it,
and the task settings are those of yam_kronos_pick_real.yaml at the time of
each run. Compare each line's result with its `real` field: a sim that mimics
the rig should reproduce them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_sim.core.clock import ManualClock  # noqa: E402
from teleop_sim.core.config import RunConfig, build_system  # noqa: E402
from teleop_sim.core.protocols import Recorder  # noqa: E402
from teleop_sim.runlog import RunLog  # noqa: E402

TILE = (480, 360)  # width, height of each view in the video


class MultiViewVideo(Recorder):
    """Six views per control step, streamed to an mp4 (nothing held in memory).

    3/4, side (across the approach line), gripper close-up and top-down free
    cameras, plus the two images the policy itself sees.
    """

    def __init__(self, path: Path, robot, cup: int, title: str, phase_of, fps: float) -> None:
        import cv2

        self.cv2, self.path, self.robot, self.cup = cv2, Path(path), robot, cup
        self.title, self.phase_of, self.fps = title, phase_of, fps
        self.renderer = mujoco.Renderer(robot.model, height=TILE[1], width=TILE[0])
        self.ee = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "kronos")
        self.writer = None
        self.step = 0

    def _free(self, lookat, distance, azimuth, elevation):
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance = lookat, distance
        cam.azimuth, cam.elevation = azimuth, elevation
        self.renderer.update_scene(self.robot.data, camera=cam)
        return self.renderer.render()

    def start_episode(self, seed=None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, obs, action) -> None:
        cv2, d = self.cv2, self.robot.data
        cup = d.xpos[self.cup].copy()
        grip = d.xpos[self.ee] + d.xmat[self.ee].reshape(3, 3)[:, 2] * 0.12
        work = np.array([cup[0] * 0.8, cup[1] * 0.8, 0.14])
        yaw = float(np.degrees(np.arctan2(cup[1], cup[0])))
        tiles = [
            ("3/4 view", self._free(work, 1.05, 180 + yaw + 35, -22)),
            ("side view (across the approach)", self._free(work, 0.85, 90 + yaw, -5)),
            ("gripper close-up", self._free(grip, 0.36, 180 + yaw + 30, -25)),
            ("top-down", self._free(np.array([cup[0] * 0.7, cup[1] * 0.7, 0.0]), 0.95, 180, -89)),
            ("policy camera: overhead", obs.images.get("top")),
            ("policy camera: wrist (what Claude sees)", obs.images.get("wrist")),
        ]
        out = []
        for label, img in tiles:
            if img is None:
                img = np.zeros((TILE[1], TILE[0], 3), np.uint8)
            img = cv2.resize(np.ascontiguousarray(img), TILE, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.rectangle(img, (0, 0), (TILE[0], 24), (20, 20, 20), -1)
            cv2.putText(
                img, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA
            )
            out.append(img)
        frame = np.vstack([np.hstack(out[:3]), np.hstack(out[3:])])
        up = d.xmat[self.cup].reshape(3, 3)[2, 2]
        tilt = float(np.degrees(np.arccos(np.clip(up, -1, 1))))
        bar = np.full((34, frame.shape[1], 3), 32, np.uint8)
        text = (
            f"{self.title}   t = {self.step / self.fps:5.1f} s   phase: {self.phase_of():13}"
            f"   gripper {obs.gripper:4.2f}   cup {cup[2] * 1000:5.0f} mm up, tilt {tilt:4.1f} deg"
        )
        cv2.putText(
            bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        frame = np.vstack([bar, frame])
        if self.writer is None:
            h, w = frame.shape[:2]
            for codec in ("avc1", "mp4v"):  # H.264 plays in browsers
                self.writer = cv2.VideoWriter(
                    str(self.path), cv2.VideoWriter_fourcc(*codec), self.fps, (w, h)
                )
                if self.writer.isOpened():
                    break
        self.writer.write(frame)
        self.step += 1

    def end_episode(self, result) -> None:
        if self.writer is not None:
            self.writer.release()
        self.renderer.close()

    def discard_episode(self) -> None:
        self.end_episode(None)


NOMINAL = str(ROOT / "teleop_sim/robots/specs/yam_kronos.yaml")

#: Real grasp settings, survey centre and cup position, from the run logs.
RUNS = {
    "run1 17:28": dict(
        cup=(0.4707, 0.0826),
        survey=[0.44, 0.08],
        grasp=dict(past_centre=0.02, height=0.035, pitch_deg=45.0),
        real="toppled; jaws closed on nothing (0.995)",
    ),
    "run2 17:39": dict(
        cup=(0.5433, -0.0128),
        survey=[0.55, -0.03],
        grasp=dict(past_centre=0.0, height=0.05, pitch_deg=20.0, standoff=0.08),
        real="held, jaws 0.777 (12.9 deg before closed), placed upright",
    ),
    "run3 17:50": dict(
        cup=(0.5586, -0.0389),
        survey=[0.55, -0.03],
        grasp=dict(past_centre=0.0, height=0.08, pitch_deg=10.0, standoff=0.08, pitch_trim_deg=2.0),
        real="held, jaws 0.747 (14.7 deg before closed), placed upright",
    ),
}


def vision_config(kind: str) -> dict:
    if kind == "oracle":
        return {"type": "oracle", "target_body": "cup"}
    return {
        "type": "claude",
        "fast_model": "claude-sonnet-5",
        "smart_model": "claude-opus-5-5",
        "escalate_below": 0.5,
    }


def replay(
    name: str, run: dict, vision: str, runs_dir: str | None, video_dir: str | None = None
) -> dict:
    cfg = RunConfig.from_yaml(ROOT / "teleop_sim/configs/yam_kronos_pick_sim.yaml")
    cfg.robot["spec"] = "../robots/specs/yam_kronos_gem13.yaml"  # the rig as measured
    cfg.robot["scene"] = "../scenes/gem13_cell.yaml"  # wood table, curtains, black cup
    cfg.safety["enforce_workspace"] = False
    cfg.source["policy"].update(
        vision=vision_config(vision),
        planning_spec=NOMINAL,
        instruction="pick up the frosted translucent plastic cup in front of the right arm",
        glass_height=0.11,
        survey_center=run["survey"],
        grasp=run["grasp"],
        max_joint_speed=0.25,
        approach_speed=0.10,
        settle_seconds=0.8,
        sag_compensation=True,
        arrive_timeout=4.0,
        max_attempts=2,
        workspace_xy={"lower": [0.2, -0.3], "upper": [0.7, 0.3]},
    )
    system = build_system(cfg, clock=ManualClock())
    robot = system.robot
    model, data = robot.model, robot.data
    cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    cup_qpos = model.jnt_qposadr[model.body_jntadr[cup]]

    base_reset = robot.reset

    def reset(seed=None):
        base_reset(seed)
        data.qpos[cup_qpos : cup_qpos + 2] = run["cup"]
        mujoco.mj_forward(model, data)
        return robot.get_observation()

    robot.reset = reset

    def tilt() -> float:
        return float(np.degrees(np.arccos(np.clip(data.xmat[cup].reshape(3, 3)[2, 2], -1, 1))))

    policy = system.source.policy
    log = RunLog.timestamped(runs_dir, "replay_" + name.split()[0]) if runs_dir else None
    policy.bind(robot=robot, run_log=log)
    seen = {"jaw": [], "tilt": 0.0, "lift_mm": 0.0, "located": [], "phase": "start"}
    base_event = policy.log.event

    def event(kind, **fields):
        phase = fields.get("phase")
        if kind == "move":
            seen["phase"] = phase
        elif kind in ("plan", "locate", "verify"):
            seen["phase"] = f"{kind} (Claude)" if vision == "claude" else kind
        if kind == "move" and phase in ("hold", "lower", "put_down"):
            seen["jaw"].append(round(float(robot.get_observation().gripper), 3))
        if kind == "move" and phase in ("hold", "lower", "open"):
            seen["tilt"] = max(seen["tilt"], tilt())
            seen["lift_mm"] = max(seen["lift_mm"], float(data.xpos[cup][2]) * 1000)
        if kind == "locate" and fields.get("table_xy") is not None:
            seen["located"].append([float(v) for v in fields["table_xy"]])
        return base_event(kind, **fields)

    policy.log.event = event
    video = None
    if video_dir:
        video = Path(video_dir) / f"{name.split()[0]}_{vision}.mp4"
        system.loop.recorder = MultiViewVideo(
            video,
            robot,
            cup,
            f"{name} replay ({vision} vision)",
            lambda: seen["phase"],
            robot.spec.control_hz,
        )
    result = system.loop.run_episode(seed=0)

    span = float(np.degrees(robot.spec.gripper.closed_pos - robot.spec.gripper.open_pos))
    where = np.array(run["cup"])
    return dict(
        sim=f"{result.outcome.value} {result.failure_tag or ''}".strip(),
        jaw_at_grasp=seen["jaw"][:1],
        deg_before_closed=[round((1 - j) * span, 1) for j in seen["jaw"][:1]],
        cup_lifted_mm=round(seen["lift_mm"]),
        max_tilt_while_held=round(seen["tilt"], 1),
        final_tilt=round(tilt(), 1),
        final_moved_mm=round(float(np.linalg.norm(data.xpos[cup][:2] - where)) * 1000),
        locate_err_mm=[
            round(float(np.linalg.norm(np.array(p) - where)) * 1000) for p in seen["located"]
        ],
        sim_seconds=round(result.duration, 1),
        real=run["real"],
        video=str(video) if video else None,
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("vision", choices=["oracle", "claude"], nargs="?", default="oracle")
    ap.add_argument("runs", nargs="*", help="subset, e.g. run1 run3")
    ap.add_argument("--runs-dir", help="write each replay's run log here")
    ap.add_argument("--video", help="write a multi-view mp4 per replay into this directory")
    args = ap.parse_args()
    for name, run in RUNS.items():
        if args.runs and not any(o in name for o in args.runs):
            continue
        print(
            name, json.dumps(replay(name, run, args.vision, args.runs_dir, args.video)), flush=True
        )


if __name__ == "__main__":
    main()
