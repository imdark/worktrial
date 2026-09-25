"""A per-run directory of what happened: events, images, and every model call.

Deliberately dumb -- JSON lines and image files -- because its first reader
is a person debugging a failed grasp, and its second is the evaluation work
that follows the MVP. Every VLM call is kept with the image it saw and the
answer it gave, which is the raw material for scoring models against
ground truth later.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    return value


class RunLog:
    def __init__(self, root: str | Path | None) -> None:
        self.root = None if root is None else Path(root)
        self._n = 0
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def timestamped(cls, base: str | Path, tag: str = "") -> RunLog:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return cls(Path(base) / (f"{stamp}_{tag}" if tag else stamp))

    def event(self, kind: str, **fields: Any) -> None:
        line = {"t": time.time(), "event": kind, **_jsonable(fields)}
        print(f"[run] {kind}: " + ", ".join(f"{k}={v}" for k, v in fields.items() if k != "images"))
        self._append("events.jsonl", line)

    def call(self, record: dict[str, Any]) -> None:
        self._append("calls.jsonl", {"t": time.time(), **_jsonable(record)})

    def image(self, name: str, rgb: np.ndarray) -> str | None:
        """Save an RGB image; returns its file name relative to the run."""
        if self.root is None:
            return None
        self._n += 1
        fname = f"{self._n:04d}_{name}.png"
        try:
            import cv2

            cv2.imwrite(str(self.root / fname), np.ascontiguousarray(rgb[:, :, ::-1]))
        except ImportError:
            np.save(self.root / (fname + ".npy"), rgb)
        return fname

    def _append(self, fname: str, line: dict[str, Any]) -> None:
        if self.root is None:
            return
        with open(self.root / fname, "a") as fh:
            fh.write(json.dumps(line) + "\n")
