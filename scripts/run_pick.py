"""Run the VLM-guided pick / lift / place task, in sim or on a robots_realtime rig.

    # sim, vision from privileged state -- no API key, runs faster than real time:
    python scripts/run_pick.py teleop_sim/configs/yam_kronos_pick_sim.yaml --vision oracle --fast

    # sim, vision from Claude (needs ANTHROPIC_API_KEY or an `ant auth login` profile):
    python scripts/run_pick.py teleop_sim/configs/yam_kronos_pick_sim.yaml --video outputs/pick.mp4

    # real rig: read-only link check first (sends no commands), then the task
    # with a y/N confirmation before every motion:
    python scripts/run_pick.py teleop_sim/configs/yam_kronos_pick_real.yaml --probe
    python scripts/run_pick.py teleop_sim/configs/yam_kronos_pick_real.yaml

Every run writes runs/<timestamp>/: events.jsonl, calls.jsonl (each model call
with its image, answer, latency and tokens) and the annotated images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_sim.core.clock import ManualClock, WallClock  # noqa: E402
from teleop_sim.core.config import RunConfig, build_system  # noqa: E402
from teleop_sim.core.protocols import Recorder  # noqa: E402
from teleop_sim.runlog import RunLog  # noqa: E402


class VideoRecorder(Recorder):
    """Writes one camera stream of the episode to an mp4."""

    def __init__(self, path: Path, camera: str, fps: float) -> None:
        self.path, self.camera, self.fps = Path(path), camera, fps
        self._writer = None

    def start_episode(self, seed: int | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, obs, action) -> None:
        import cv2

        frame = obs.images.get(self.camera)
        if frame is None:
            return
        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
        self._writer.write(np.ascontiguousarray(frame[:, :, ::-1]))

    def end_episode(self, result) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"video: {self.path}")

    def discard_episode(self) -> None:
        self.end_episode(None)


def make_gate():
    state = {"all": False}

    def gate(label: str, detail: str) -> bool:
        if state["all"]:
            return True
        answer = input(f"\n[confirm] next motion: {label} ({detail}). Proceed? [y/N/a=all] ")
        answer = answer.strip().lower()
        if answer == "a":
            state["all"] = True
            return True
        return answer == "y"

    return gate


def probe(system, log: RunLog) -> int:
    """Read-only: connect, read state and cameras, report. Sends no command."""
    robot = system.robot
    robot.connect()
    obs = robot.reset()
    print(robot.describe_link() if hasattr(robot, "describe_link") else "")
    print(f"joint_pos (rad): {np.round(obs.joint_pos, 4)}")
    print(f"gripper (0=open, 1=closed): {obs.gripper:.3f}")
    for name, img in obs.images.items():
        path = log.image(f"probe_{name}", img)
        print(
            f"camera {name}: {img.shape} mean={img.mean():.1f} -> {log.root / path if path else ''}"
        )
    intr = obs.extra.get("intrinsics", {})
    for name, k in intr.items():
        print(
            f"intrinsics {name}: fx={k.get('fx'):.1f} fy={k.get('fy'):.1f} "
            f"cx={k.get('cx'):.1f} cy={k.get('cy'):.1f}"
        )
    robot.disconnect()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config")
    ap.add_argument(
        "--vision", choices=["claude", "oracle"], help="override the config's vision type"
    )
    ap.add_argument("--target-body", default="glass_right", help="oracle: the body to pick (sim)")
    ap.add_argument("--instruction", help="override the task instruction")
    ap.add_argument("--host", help="robots_realtime host (real rig)")
    ap.add_argument("--ports", type=int, nargs=2, metavar=("PUB", "SUB"),
                    help="bus ports, e.g. the local ends of an SSH tunnel")
    gate = ap.add_mutually_exclusive_group()
    gate.add_argument(
        "--confirm",
        dest="confirm",
        action="store_true",
        default=None,
        help="ask before every motion (default on a real robot)",
    )
    gate.add_argument("--no-confirm", dest="confirm", action="store_false")
    ap.add_argument(
        "--probe", action="store_true", help="real rig: read-only link check, no motion"
    )
    ap.add_argument(
        "--approach-only",
        action="store_true",
        help="go to the standoff pose in front of the target, photograph, back out",
    )
    ap.add_argument(
        "--look-only",
        action="store_true",
        help="move to the viewing poses and locate the target, never approach it",
    )
    ap.add_argument(
        "--fast", action="store_true", help="sim: run on a manual clock, no real-time wait"
    )
    ap.add_argument("--video", type=Path, help="sim: write the overview camera to this mp4")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args = ap.parse_args()

    config = RunConfig.from_yaml(args.config)
    policy_cfg = config.source.setdefault("policy", {})
    if args.vision == "oracle":
        policy_cfg["vision"] = {"type": "oracle", "target_body": args.target_body}
    elif args.vision == "claude":
        policy_cfg["vision"] = {
            k: v for k, v in policy_cfg.get("vision", {}).items() if k not in ("target_body",)
        } | {"type": "claude"}
    if args.look_only:
        policy_cfg["look_only"] = True
    if args.approach_only:
        policy_cfg["approach_only"] = True
    if args.instruction:
        policy_cfg["instruction"] = args.instruction
    if args.host:
        config.robot["host"] = args.host
    if args.ports:
        config.robot["pub_port"], config.robot["sub_port"] = args.ports

    is_sim = config.robot.get("type") == "mujoco"
    log = RunLog.timestamped(
        args.runs_dir, "probe" if args.probe else ("sim" if is_sim else "real")
    )
    clock = ManualClock() if (args.fast and is_sim) else WallClock()
    system = build_system(config, clock=clock)
    print(f"run log: {log.root}")

    if args.probe:
        return probe(system, log)

    confirm = (not is_sim) if args.confirm is None else args.confirm
    policy = system.source.policy
    policy.bind(robot=system.robot, run_log=log, gate=make_gate() if confirm else None)
    if args.video:
        camera = policy.overview_camera or policy.wrist_camera
        system.loop.recorder = VideoRecorder(args.video, camera, system.spec.control_hz)

    try:
        result = system.loop.run_episode(seed=config.seed)
    finally:
        system.robot.disconnect()
    log.event("result", **result.to_record())
    print(
        f"\n{result.outcome.value.upper()}"
        + (f" ({result.failure_tag})" if result.failure_tag else "")
        + f" in {result.duration:.1f}s, {result.steps} steps"
    )
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
