"""Camera relay for a robots_realtime rig reached over a slow link.

A robots_realtime CameraNode publishes every frame at full rate with depth
attached: two 640x480 cameras are ~90 MB/s. Over gem13's gigabit cable that is
fine; over Tailscale it backs up, and joint state queued behind it arrived
3.6-4.9 s late (48 ms without the cameras). This relay runs ON the rig, reads
the camera topics from the local bus, and republishes each at a low rate as
JPEG, without depth, on its own port. Joint state and commands keep using the
rig's bus directly.

Standalone on purpose -- only zmq, msgpack, msgpack_numpy, cv2 and numpy, all
in robots_realtime's venv -- so it can be copied to the rig and run there:

    scp teleop_sim/robots/real/camera_relay.py pantheon@<rig>:/tmp/
    ssh pantheon@<rig> '~/robots_realtime_fleet/.venv/bin/python /tmp/camera_relay.py'

It only subscribes to the rig's bus and binds one new port; it never publishes
onto the rig's bus.
"""

from __future__ import annotations

import argparse
import time

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import zmq


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--topics", nargs="+", default=["right_wrist_cam/rgb", "exo_cam/rgb"])
    ap.add_argument("--bus-port", type=int, default=5556, help="the rig's XPUB port")
    ap.add_argument("--port", type=int, default=5557, help="port this relay binds")
    ap.add_argument("--hz", type=float, default=5.0, help="frames per second per camera")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--max-seconds", type=float, default=3600.0, help="exit after this long")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 4)  # never queue: only the newest frame matters
    sub.connect(f"tcp://127.0.0.1:{args.bus_port}")
    for topic in args.topics:
        sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)
    pub.bind(f"tcp://*:{args.port}")

    period = 1.0 / args.hz
    latest: dict[str, dict] = {}
    last_sent: dict[str, float] = {}
    started = time.monotonic()
    print(f"relaying {args.topics} at {args.hz:g} Hz on :{args.port}", flush=True)
    while time.monotonic() - started < args.max_seconds:
        while sub.poll(5):
            topic, raw = sub.recv_multipart()
            latest[topic.decode()] = msgpack.unpackb(
                raw, object_hook=msgpack_numpy.decode, raw=False
            )
        now = time.monotonic()
        for topic, envelope in latest.items():
            if now - last_sent.get(topic, 0.0) < period:
                continue
            data = envelope["data"]
            rgb = np.ascontiguousarray(data["images"]["rgb"])
            ok, jpeg = cv2.imencode(
                ".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, args.quality]
            )
            if not ok:
                continue
            out = {
                "ts": envelope["ts"],
                "src": "camera_relay",
                "data": {
                    "images": {"rgb_jpeg": jpeg.tobytes()},
                    "timestamp": data.get("timestamp"),
                    "intrinsics": data.get("intrinsics"),
                    "shape": list(rgb.shape),
                },
            }
            try:
                pub.send_multipart(
                    [
                        topic.encode(),
                        msgpack.packb(out, default=msgpack_numpy.encode, use_bin_type=True),
                    ],
                    zmq.NOBLOCK,
                )
            except zmq.Again:
                pass
            last_sent[topic] = now
    pub.close(linger=0)
    sub.close(linger=0)


if __name__ == "__main__":
    main()
