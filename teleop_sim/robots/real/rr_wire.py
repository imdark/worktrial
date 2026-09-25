"""robots_realtime's bus protocol, reimplemented so this repo does not import it.

Pinned down from robots_realtime/runtime/transport on gem13 (2026-09-24):

* A broker process binds an XSUB socket on tcp://*:5555 (publishers connect)
  and an XPUB socket on tcp://*:5556 (subscribers connect). It binds every
  interface, so a client on another machine connects directly.
* Every message is a two-frame multipart: ``[topic, payload]``. The topic is
  UTF-8 ``"{node}/{suffix}"``; the payload is msgpack of the envelope
  ``{"ts": float, "src": str, "data": dict}`` with numpy arrays encoded by
  msgpack_numpy.
* ``ts`` is the SENDER's ``time.time()``. A RobotNode drops any command whose
  ts is more than ``cmd_timeout_s`` (0.5 s) older than its own clock, so a
  client whose clock is skewed from the rig's by that much is silently
  ignored. The client measures the skew at connect time.
* Subscribers keep only the latest message per topic.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

DEFAULT_PUB_PORT = 5555  # publishers connect here (broker XSUB)
DEFAULT_SUB_PORT = 5556  # subscribers connect here (broker XPUB)


def pack(envelope: dict[str, Any]) -> bytes:
    import msgpack
    import msgpack_numpy

    return msgpack.packb(envelope, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(raw: bytes) -> dict[str, Any]:
    import msgpack
    import msgpack_numpy

    return msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False)


def encode(topic: str, data: dict[str, Any], src: str, ts: float | None = None) -> list[bytes]:
    envelope = {"ts": time.time() if ts is None else float(ts), "src": src, "data": data}
    return [topic.encode(), pack(envelope)]


def decode(parts: list[bytes]) -> tuple[str, dict[str, Any]]:
    return parts[0].decode(), unpack(parts[1])


@dataclass
class Received:
    envelope: dict[str, Any]
    local_time: float  # time.time() here when it arrived
    local_mono: float  # time.monotonic() here when it arrived

    @property
    def data(self) -> dict[str, Any]:
        return self.envelope["data"]

    def age(self) -> float:
        return time.monotonic() - self.local_mono


class BusPublisher:
    def __init__(self, host: str, port: int = DEFAULT_PUB_PORT, src: str = "teleop_sim") -> None:
        import zmq

        self.src = src
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SNDHWM, 10)  # never queue a backlog of stale commands
        self._sock.connect(f"tcp://{host}:{port}")
        self._lock = threading.Lock()

    def publish(self, topic: str, data: dict[str, Any], ts: float | None = None) -> None:
        import zmq

        with self._lock:
            try:
                self._sock.send_multipart(encode(topic, data, self.src, ts), zmq.NOBLOCK)
            except zmq.Again:
                pass  # link down: dropping is right, the rig holds position on silence

    def close(self) -> None:
        self._sock.close(linger=0)


class BusSubscriber:
    """Latest message per topic, drained on a background thread."""

    def __init__(self, host: str, topics: list[str], port: int = DEFAULT_SUB_PORT) -> None:
        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVHWM, 50)
        self._sock.connect(f"tcp://{host}:{port}")
        for topic in topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self._latest: dict[str, Received] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.errors = 0
        self._thread = threading.Thread(target=self._drain, daemon=True, name="rr-sub")
        self._thread.start()

    def _drain(self) -> None:
        import zmq

        while not self._stop.is_set():
            if not self._sock.poll(20):
                continue
            try:
                parts = self._sock.recv_multipart(zmq.NOBLOCK)
                topic, envelope = decode(parts)
            except zmq.Again:
                continue
            except Exception:
                self.errors += 1  # one malformed payload must not kill the feed
                continue
            with self._lock:
                self._latest[topic] = Received(envelope, time.time(), time.monotonic())

    def get(self, topic: str) -> Received | None:
        with self._lock:
            return self._latest.get(topic)

    def topics(self) -> list[str]:
        with self._lock:
            return sorted(self._latest)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close(linger=0)
