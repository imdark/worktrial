"""Watch gem13's cameras through the vision safety loop's questions -- arm untouched.

    third_party/laya-vision/.venv/bin/python scripts/laya_server.py &
    python scripts/watch_hazards.py teleop_sim/configs/yam_kronos_pick_real.yaml \
        --host pantheon-gem13 --camera-port 5557                     # live P(yes)
    ... --label hazard --note "hand over cup" --seconds 20          # collect positives
    ... --label clear --seconds 20                                  # and negatives

Only subscribes: it builds the robot bridge, reads observations, and never
calls send_action, so the arm stays wherever the rig holds it. Prints P(yes)
for every check of vision_hazard (or --checks) about 3 times a second, with
TRIP when the monitor's debounce would have stopped the arm.

With --label every frame is saved to <out>/<label>/<camera>_<n>.jpg plus a
labels.jsonl line (label, note, P per check). Those folders feed
scripts/eval_hazard_monitor.py --hazard (detection rate) and, later,
fine-tuning a foreign-object check: put a box, a bottle, a hand in the arm's
path, record --label hazard; take it away, record --label clear.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import teleop_sim.builtins  # noqa: E402,F401
from teleop_sim.control.safety.vision_hazard import (  # noqa: E402
    DEFAULT_CHECKS,
    HazardCheck,
    HazardTracker,
    laya_asker,
)
from teleop_sim.core.config import RunConfig  # noqa: E402
from teleop_sim.core.registry import ROBOTS, build  # noqa: E402
from teleop_sim.core.spec import RobotSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config")
    ap.add_argument("--host")
    ap.add_argument("--camera-port", type=int)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8765")
    ap.add_argument("--checks", type=Path, help="JSON list of HazardCheck dicts")
    ap.add_argument("--label", choices=["hazard", "clear"])
    ap.add_argument("--note", default="")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    config = RunConfig.from_yaml(args.config)
    robot_cfg = dict(config.robot)
    if args.host:
        robot_cfg["host"] = args.host
    if args.camera_port:
        robot_cfg["camera_port"] = args.camera_port
    spec = RobotSpec.from_yaml(config.resolve(robot_cfg.pop("spec")))
    robot_cfg.pop("scene", None)
    robot = build(ROBOTS, robot_cfg, spec=spec)

    checks = (
        tuple(HazardCheck(**c) for c in json.loads(args.checks.read_text()))
        if args.checks
        else DEFAULT_CHECKS
    )
    tracker = HazardTracker(checks)
    ask = laya_asker(args.endpoint, timeout_s=5.0)
    out = None
    if args.label:
        out = args.out or ROOT / "runs" / f"hazards_{time.strftime('%Y%m%d-%H%M%S')}"
        (out / args.label).mkdir(parents=True, exist_ok=True)
        print(f"saving {args.label!r} frames to {out / args.label}")

    robot.connect()
    started, n = time.monotonic(), 0
    try:
        while args.seconds is None or time.monotonic() - started < args.seconds:
            obs = robot.get_observation()
            parts = []
            for camera in tracker.cameras():
                image = obs.images.get(camera)
                if image is None:
                    continue
                answers = ask(image, tracker.questions_for(camera))
                trip = tracker.update(camera, answers)
                parts.append(
                    f"{camera}: "
                    + " ".join(f"{k}={v:.2f}" for k, v in answers.items())
                    + (f"  TRIP({trip.check})" if trip else "")
                )
                if trip:
                    tracker.reset()
                if out is not None:
                    import cv2

                    n += 1
                    rel = f"{args.label}/{camera}_{n:05d}.jpg"
                    cv2.imwrite(str(out / rel), np.ascontiguousarray(image[:, :, ::-1]))
                    with open(out / "labels.jsonl", "a") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "image": rel,
                                    "camera": camera,
                                    "label": args.label,
                                    "note": args.note,
                                    "t": time.time(),
                                    "p": {k: round(v, 3) for k, v in answers.items()},
                                }
                            )
                            + "\n"
                        )
            print(" | ".join(parts), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
    if out is not None:
        print(f"saved {n} frames under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
