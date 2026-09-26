"""Run N pick / lift / place episodes back to back and keep everything, for training.

    python scripts/collect_real_picks.py teleop_sim/configs/yam_kronos_pick_real.yaml \
        --episodes 10 --host pantheon-gem13 --camera-port 5557

Each episode picks the cup where the previous one left it and puts it down at a
new random spot in --place-region, so positions and backgrounds vary and the cup
does not drift out of reach. Everything lands in one batch directory:

    <batch>/ep01/ ... ep10/     per episode: the run log (events, Claude calls with
                                their images) plus teleop_sim.telemetry output --
                                telemetry.jsonl, frames/, labels.jsonl, summary.json
    <batch>/episodes.jsonl      one line per episode: outcome, timing, where the cup
                                was found and put, the view gap, jaw reading, labels
    <batch>/labels.jsonl        every labelled wrist frame of the batch (paths are
                                relative to the batch directory), for laya-vision

The batch stops early after --max-failures failures in a row (a cup that has
fallen over or rolled away needs a person).
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

from teleop_sim.control.safety.review import task_context  # noqa: E402
from teleop_sim.control.safety.vision_hazard import with_hazard_monitor  # noqa: E402
from teleop_sim.core.config import RunConfig, build_system  # noqa: E402
from teleop_sim.runlog import RunLog  # noqa: E402
from teleop_sim.telemetry import TelemetryRecorder  # noqa: E402


def episode_facts(ep_dir: Path) -> dict:
    facts: dict = {}
    events = ep_dir / "events.jsonl"
    if not events.exists():
        return facts
    for line in events.read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "refined":
            facts.setdefault("view_gap_mm", []).append(e.get("moved_mm"))
            facts.setdefault("used", []).append(e.get("used"))
        elif e["event"] == "verify":
            facts.setdefault("verify", []).append(
                {k: e.get(k) for k in ("ok", "jaw", "vision_ok", "confidence", "model")}
            )
        elif e["event"] == "grasp_plan":
            facts.setdefault("grasp_xy", []).append(e.get("glass_xy"))
    return facts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--host")
    ap.add_argument("--camera-port", type=int)
    ap.add_argument(
        "--place-region",
        type=float,
        nargs=4,
        default=[0.50, 0.60, -0.08, 0.04],
        metavar=("X0", "X1", "Y0", "Y1"),
        help="where to put the cup down, m",
    )
    ap.add_argument("--start-xy", type=float, nargs=2, help="where the cup is now (survey centre)")
    ap.add_argument("--max-failures", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--hazard-monitor",
        action="store_true",
        help="opt in to the vision safety loop (configs/safety/vision_hazard.yaml)",
    )
    args = ap.parse_args()

    config = RunConfig.from_yaml(args.config)
    if args.host:
        config.robot["host"] = args.host
    if args.camera_port:
        config.robot["camera_port"] = args.camera_port
    if args.hazard_monitor:
        config.safety = with_hazard_monitor(config.safety)
    system = build_system(config)
    policy = system.source.policy
    batch = args.out or ROOT / "runs" / f"collect_{time.strftime('%Y%m%d-%H%M%S')}"
    batch.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    x0, x1, y0, y1 = args.place_region
    cup_xy = np.array(args.start_xy) if args.start_xy else np.asarray(policy.survey_center)
    failures = 0
    print(f"batch: {batch}")
    try:
        for n in range(1, args.episodes + 1):
            ep = batch / f"ep{n:02d}"
            place = [float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))]
            policy.survey_center = np.asarray(cup_xy, dtype=float)
            policy.place_xy = place
            log = RunLog(ep)
            policy.bind(robot=system.robot, run_log=log, gate=None)
            if hasattr(system.safety, "bind"):
                system.safety.bind(
                    run_log=log, robot=system.robot, context=task_context(policy.instruction)
                )
            system.loop.recorder = TelemetryRecorder(
                ep,
                phase_of=lambda: policy.phase,
                control_hz=system.spec.control_hz,
                meta={
                    "episode": n,
                    "survey_center": list(map(float, cup_xy)),
                    "place_xy": place,
                    "config": str(args.config),
                },
            )
            print(
                f"\n=== episode {n}/{args.episodes}: look near {np.round(cup_xy, 3)}, "
                f"put down at {np.round(place, 3)}",
                flush=True,
            )
            result = system.loop.run_episode(seed=args.seed + n)
            facts = episode_facts(ep)
            labels = json.loads((ep / "summary.json").read_text()).get("label_counts", {})
            row = {
                "episode": n,
                "outcome": result.outcome.value,
                "failure_tag": result.failure_tag,
                "seconds": round(result.duration, 1),
                "looked_near": list(map(float, cup_xy)),
                "placed_xy": None
                if policy.placed_xy is None
                else [round(float(v), 4) for v in policy.placed_xy],
                "labels": labels,
                **facts,
            }
            with open(batch / "episodes.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            with open(batch / "labels.jsonl", "a") as out:
                for line in (ep / "labels.jsonl").read_text().splitlines():
                    rec = json.loads(line)
                    rec["image"] = f"{ep.name}/{rec['image']}"
                    rec["episode"] = n
                    rec["episode_outcome"] = result.outcome.value
                    out.write(json.dumps(rec) + "\n")
            print(
                f"--- episode {n}: {result.outcome.value}"
                f"{' (' + result.failure_tag + ')' if result.failure_tag else ''}"
                f" in {result.duration:.0f}s, labels {labels}",
                flush=True,
            )
            if result.succeeded and policy.placed_xy is not None:
                cup_xy, failures = policy.placed_xy, 0
            else:
                failures += 1
                grasps = facts.get("grasp_xy")
                if grasps:  # the cup is probably near where it was last aimed at
                    cup_xy = np.array(grasps[-1][:2])
                if failures >= args.max_failures:
                    print(f"stopping: {failures} failures in a row")
                    break
    finally:
        system.robot.disconnect()
    rows = [json.loads(line) for line in (batch / "episodes.jsonl").read_text().splitlines()]
    total: dict[str, int] = {}
    for r in rows:
        for k, v in r["labels"].items():
            total[k] = total.get(k, 0) + v
    ok = sum(r["outcome"] == "success" for r in rows)
    print(f"\n{ok}/{len(rows)} succeeded; labelled wrist frames {total}; batch {batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
