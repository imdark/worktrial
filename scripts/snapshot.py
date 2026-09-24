"""Render a contact sheet of every camera over a scripted sweep.

    python scripts/snapshot.py --out /tmp/sheet.png

Headless proof that the model moves and the cameras see it -- the check that
tests/conformance/ cannot make, because "is this framed sensibly" is a question
you have to look at. One row per camera, one column per sampled step.

Writes PNG with only the standard library, so there is no image dependency in
a project that does not otherwise need one.
"""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path

import numpy as np

from teleop_sim.core.clock import WallClock
from teleop_sim.core.spec import RobotSpec
from teleop_sim.core.types import Action
from teleop_sim.robots.sim.mujoco_robot import MujocoRobot

DEFAULT_SPEC = "teleop_sim/robots/specs/yam.yaml"


def write_png(path: str | Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=DEFAULT_SPEC)
    parser.add_argument("--out", default="snapshot.png")
    parser.add_argument("--steps", type=int, default=91)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--amplitude", type=float, default=0.12, help="fraction of each range")
    args = parser.parse_args()

    spec = RobotSpec.from_yaml(args.spec)
    robot = MujocoRobot(spec, WallClock(), image_size=(args.width, args.height))
    robot.connect()
    robot.reset(seed=0)

    home = spec.home_joints()
    span = (spec.upper_limits - spec.lower_limits) / 2 * args.amplitude
    every = max(1, args.steps // args.columns)

    frames: list[dict[str, np.ndarray]] = []
    for step in range(args.steps):
        phase = 2 * math.pi * 0.25 * (step / spec.control_hz)
        targets = spec.clip_joints(
            home + span * np.array([math.sin(phase + i * 0.7) for i in range(spec.dof)])
        )
        robot.send_action(
            Action(
                mode=spec.default_control_mode,
                values=targets,
                gripper=0.5 * (1 + math.sin(phase * 2)),
                timestamp=0.0,
                obs_timestamp=0.0,
            )
        )
        if step % every == 0 and len(frames) < args.columns:
            frames.append(robot.get_observation().images)

    names = [camera.name for camera in robot.cameras]
    sheet = np.vstack([np.hstack([f[name] for f in frames]) for name in names])
    write_png(args.out, sheet)
    robot.disconnect()

    print(f"{len(frames)} columns x {names} -> {args.out}  {sheet.shape}")


if __name__ == "__main__":
    main()
