"""Replay recorded frames through the vision safety loop and count its trips.

    third_party/laya-vision/.venv/bin/python scripts/laya_server.py &   # the model
    python scripts/eval_hazard_monitor.py runs/collect_20260925-133615
    python scripts/eval_hazard_monitor.py runs/collect_* --hazard runs/hazards_*/hazard

Clean batches (from scripts/collect_real_picks.py: telemetry.jsonl + frames/,
~2 Hz per camera) have no hazard in them, so every trip there is a false stop.
Folders given with --hazard (from scripts/watch_hazards.py --label hazard:
<cam>_<n>.jpg) contain one, so every frame of them should score high.

Frames go through the production path: JPEG over HTTP to laya_server.py, then
teleop_sim.control.safety.vision_hazard.HazardTracker with the same checks and
debounce as the monitor. Output: per check, P(yes) quantiles on clean frames,
false trips per episode and per minute of motion, and (with --hazard) the
fraction of hazard frames at or above the trip threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_sim.control.safety.vision_hazard import (  # noqa: E402
    DEFAULT_CHECKS,
    HazardCheck,
    HazardTracker,
    laya_asker,
)


def load_rgb(path: Path) -> np.ndarray:
    import cv2

    return np.ascontiguousarray(cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1])


def episodes(batch: Path):
    for ep in sorted(p for p in batch.glob("ep[0-9]*") if p.is_dir()):
        rows = [json.loads(line) for line in (ep / "telemetry.jsonl").read_text().splitlines()]
        framed = [r for r in rows if r.get("frames")]
        seconds = rows[-1]["wall"] - rows[0]["wall"] if rows else 0.0
        yield ep, framed, seconds


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("batches", nargs="*", type=Path, help="clean batches (no hazards)")
    ap.add_argument("--hazard", nargs="*", type=Path, default=[], help="folders of hazard frames")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8765")
    ap.add_argument(
        "--checks", type=Path, help="JSON list of HazardCheck dicts"
    )
    ap.add_argument("--out", type=Path, help="write the report as JSON here")
    args = ap.parse_args()

    checks = (
        tuple(HazardCheck(**c) for c in json.loads(args.checks.read_text()))
        if args.checks
        else DEFAULT_CHECKS
    )
    tracker = HazardTracker(checks)
    ask = laya_asker(args.endpoint, timeout_s=5.0)
    clean_p: dict[str, list[float]] = defaultdict(list)
    trips: dict[str, list[dict]] = defaultdict(list)
    n_eps, minutes = 0, 0.0

    for batch in args.batches:
        for ep, framed, seconds in episodes(batch):
            n_eps += 1
            minutes += seconds / 60
            tracker.reset()
            for r in framed:
                for camera in tracker.cameras():
                    rel = r["frames"].get(camera)
                    if rel is None:
                        continue
                    answers = ask(load_rgb(ep / rel), tracker.questions_for(camera))
                    for k, v in answers.items():
                        clean_p[k].append(v)
                    trip = tracker.update(camera, answers)
                    if trip is not None:
                        trips[trip.check].append(
                            {
                                "episode": f"{batch.name}/{ep.name}",
                                "step": r["step"],
                                "phase": r["phase"],
                                "history": [round(p, 2) for p in trip.history],
                            }
                        )
                        tracker.reset()  # count the next trip independently
            print(f"  {batch.name}/{ep.name}: {len(framed)} ticks with frames", flush=True)

    hazard_p: dict[str, list[float]] = defaultdict(list)
    for folder in args.hazard:
        for path in sorted(folder.glob("*.jpg")):
            camera = path.stem.split("_")[0]
            questions = tracker.questions_for(camera)
            if questions:
                for k, v in ask(load_rgb(path), questions).items():
                    hazard_p[k].append(v)

    report: dict = {"episodes": n_eps, "minutes": round(minutes, 1), "checks": {}}
    print(f"\n{n_eps} clean episodes, {minutes:.1f} min of motion")
    for c in checks:
        p = np.array(clean_p.get(c.name, []))
        row: dict = {
            "camera": c.camera,
            "question": c.question,
            "trip_above": c.trip_above,
            "consecutive": c.consecutive,
            "false_trips": len(trips[c.name]),
            "false_trip_detail": trips[c.name],
        }
        if p.size:
            row.update(
                clean_frames=int(p.size),
                clean_median=round(float(np.median(p)), 3),
                clean_p99=round(float(np.percentile(p, 99)), 3),
                clean_max=round(float(p.max()), 3),
                clean_frames_above=round(float(np.mean(p >= c.trip_above)), 4),
            )
        h = np.array(hazard_p.get(c.name, []))
        if h.size:
            row.update(
                hazard_frames=int(h.size),
                hazard_detected=round(float(np.mean(h >= c.trip_above)), 4),
                hazard_median=round(float(np.median(h)), 3),
            )
        report["checks"][c.name] = row
        print(
            f"{c.name:15s} clean: median {row.get('clean_median')}, p99 {row.get('clean_p99')}, "
            f"max {row.get('clean_max')}, frames >= {c.trip_above}: "
            f"{row.get('clean_frames_above', 0):.1%}; false trips {row['false_trips']}"
            + (f" in {n_eps} episodes" if n_eps else "")
            + (
                f"; hazard frames detected {row['hazard_detected']:.0%} of {row['hazard_frames']}"
                if h.size
                else ""
            )
        )
        for t in trips[c.name]:
            print(f"    false trip {t['episode']} step {t['step']} ({t['phase']}): {t['history']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
