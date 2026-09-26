"""Record the rig's camera relay to disk in a separate process, then render a video.

    # while an experiment runs (Ctrl-C or SIGTERM to stop)
    python scripts/record_cameras.py record --host pantheon-gem13 --out runs/sweep_<ts>/video
    # afterwards: side-by-side MP4 with the safety events burned in
    python scripts/record_cameras.py render runs/sweep_<ts>/video --events runs/sweep_<ts>

``record`` subscribes to camera_relay.py's port on its own connection (the
control loop keeps its own), and writes every new frame's JPEG bytes as they
arrive, untouched, to <out>/<camera>/<n>.jpg with an index.jsonl line
(camera, file, rig send time, local arrival time). No decoding, so it keeps
up easily at the relay's rate.

``render`` pairs the cameras by arrival time into one wrist | overhead frame
per tick, stamps the time since the start, and, given the run directory,
overlays the monitors' events from contact_checks.jsonl and
hazard_checks.jsonl (CONTACT, BACKING OFF, PAUSE, Claude's decision, RESUME,
STOP) for a few seconds each, so the video lines up with reaction_report.json.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOPICS = {"wrist": "right_wrist_cam/rgb", "overview": "exo_cam/rgb"}


def record(args: argparse.Namespace) -> int:
    from teleop_sim.robots.real.rr_wire import BusSubscriber

    out = args.out
    for cam in TOPICS:
        (out / cam).mkdir(parents=True, exist_ok=True)
    sub = BusSubscriber(args.host, list(TOPICS.values()), port=args.port)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    last: dict[str, float] = {}
    counts = dict.fromkeys(TOPICS, 0)
    started = time.monotonic()
    print(f"recording {list(TOPICS.values())} from {args.host}:{args.port} to {out}", flush=True)
    with open(out / "index.jsonl", "a") as index:
        try:
            while not stop["now"] and time.monotonic() - started < args.max_seconds:
                for cam, topic in TOPICS.items():
                    rec = sub.get(topic)
                    if rec is None or rec.local_mono == last.get(cam):
                        continue
                    last[cam] = rec.local_mono
                    jpeg = rec.data.get("images", {}).get("rgb_jpeg")
                    if jpeg is None:
                        continue
                    counts[cam] += 1
                    rel = f"{cam}/{counts[cam]:06d}.jpg"
                    (out / rel).write_bytes(bytes(jpeg))
                    index.write(
                        json.dumps(
                            {
                                "camera": cam,
                                "file": rel,
                                "sent": rec.envelope.get("ts"),
                                "arrived": rec.local_time,
                            }
                        )
                        + "\n"
                    )
                index.flush()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
    sub.close()
    print(f"recorded {counts} frames in {time.monotonic() - started:.0f}s", flush=True)
    return 0


def _events(run: Path | None) -> list[tuple[float, str, tuple[int, int, int]]]:
    """(time, label, BGR colour) for every safety event in a run directory."""
    if run is None:
        return []
    red, amber, green, blue = (40, 40, 220), (0, 170, 255), (60, 180, 60), (200, 120, 30)
    out = []
    for name, src in (("contact_checks.jsonl", "contact"), ("hazard_checks.jsonl", "camera")):
        path = run / name
        if not path.exists():
            continue
        for line in path.open():
            e = json.loads(line)
            kind = e.get("event")
            if kind == "contact":
                out.append((e["t"], f"CONTACT J{e.get('joint', 0) + 1} ({e.get('rule')})", red))
            elif kind == "retreat":
                out.append((e["t"], "BACKING OFF", red))
            elif kind == "pause":
                out.append((e["t"], f"PAUSE: {e.get('check')}", amber))
            elif kind == "review":
                out.append(
                    (e["t"], f"Claude ({src}): {e['decision'].upper()} {e['confidence']:.2f}", blue)
                )
            elif kind == "resume":
                out.append((e["t"], f"RESUME ({src})", green))
            elif kind in ("stop", "trip"):
                out.append((e["t"], f"STOP ({src})", red))
    return sorted(out)


def _to_h264(path: Path) -> None:
    """Re-encode OpenCV's MPEG-4 Part 2 output as H.264, which browsers and most
    players play (OpenCV's own H.264 writer is not available in the pip wheels)."""
    import subprocess

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        print("imageio-ffmpeg not available: leaving the MPEG-4 file as written")
        return
    tmp = path.with_suffix(".h264.mp4")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            "23",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(path)


def render(args: argparse.Namespace) -> int:
    import cv2

    rows = [json.loads(line) for line in (args.video / "index.jsonl").open()]
    if not rows:
        raise SystemExit(f"{args.video}: no frames recorded")
    t0 = min(r["arrived"] for r in rows)
    by_cam = {cam: [r for r in rows if r["camera"] == cam] for cam in TOPICS}
    events = _events(args.events)
    fps = args.fps
    t_end = max(r["arrived"] for r in rows)
    if args.end is not None:
        t_end = min(t_end, t0 + args.end)
    # stretches where the arm is held: from a contact or pause to the next resume/stop
    starts = [e[0] for e in events if e[1].startswith(("CONTACT", "PAUSE"))]
    ends = [e[0] for e in events if e[1].startswith(("RESUME", "STOP"))]
    held = [(a, min([b for b in ends if b > a], default=float("inf"))) for a in starts]
    w, h = 640, 480
    top = 44 if args.title else 0
    out_path = args.out or args.video / "cameras.mp4"
    size = (2 * w, h + 60 + top)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    idx = dict.fromkeys(TOPICS, 0)
    frames = dict.fromkeys(TOPICS)
    n = 0
    t = t0
    start = t0 + (args.start or 0.0)
    while t <= t_end:
        for cam, lst in by_cam.items():
            while idx[cam] < len(lst) and lst[idx[cam]]["arrived"] <= t:
                img = cv2.imread(str(args.video / lst[idx[cam]]["file"]))
                if img is not None:
                    frames[cam] = cv2.resize(img, (w, h))
                idx[cam] += 1
        if t < start:
            t += 1.0 / fps
            continue
        panel = np.zeros((size[1], size[0], 3), np.uint8)
        if args.title:
            cv2.putText(panel, args.title[:90], (12, 30), 0, 0.8, (255, 255, 255), 2)
        for k, cam in enumerate(TOPICS):
            if frames[cam] is not None:
                panel[top : top + h, k * w : (k + 1) * w] = frames[cam]
            cv2.putText(panel, cam, (k * w + 10, top + 28), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, f"t = {t - t0:6.1f} s", (10, top + h + 40), 0, 0.9, (230, 230, 230), 2)
        live = [e for e in events if e[0] <= t < e[0] + args.hold]
        holding = next((a for a, b in held if a <= t < b), None)
        if live:
            _, label, colour = live[-1]
        elif holding is not None:
            label, colour = f"ARM HOLDING - Claude deciding ({t - holding:.0f} s)", (0, 150, 220)
        else:
            label, colour = "sweeping", (70, 70, 70)
        cv2.rectangle(panel, (230, top + h + 8), (2 * w - 10, top + h + 52), colour, -1)
        cv2.putText(panel, label[:60], (245, top + h + 40), 0, 0.9, (255, 255, 255), 2)
        writer.write(panel)
        n += 1
        t += 1.0 / fps
    writer.release()
    _to_h264(out_path)
    print(f"wrote {out_path}: {n} frames, {n / fps:.0f} s, {len(events)} events overlaid")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--host", required=True)
    r.add_argument("--port", type=int, default=5557, help="camera_relay.py's port")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--max-seconds", type=float, default=1800)
    v = sub.add_parser("render")
    v.add_argument("video", type=Path, help="the directory record wrote")
    v.add_argument("--events", type=Path, help="run directory with the monitors' logs")
    v.add_argument("--fps", type=float, default=10.0)
    v.add_argument("--hold", type=float, default=3.0, help="seconds each event label stays up")
    v.add_argument("--out", type=Path)
    v.add_argument("--start", type=float, help="seconds after the first frame to start")
    v.add_argument("--end", type=float, help="seconds after the first frame to stop")
    v.add_argument("--title", help="a banner across the top")
    args = ap.parse_args()
    return record(args) if args.cmd == "record" else render(args)


if __name__ == "__main__":
    raise SystemExit(main())
